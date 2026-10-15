"""Stage2-only vLLM subprocess; original environment owns feature extraction.

The worker sleeps between requests to free its KV pool before HF hidden extraction.
No reasoning parser, text stop strings, prefix cache, or cross-engine token reuse.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .artifacts.io import atomic_json, read_json, safe_error, now
from . import ROOT


class DevicePeak:
    """Whole-device NVML peak sampled every 20 ms, including the HF process."""
    def __init__(self):
        import pynvml,threading
        pynvml.nvmlInit()
        self.nvml=pynvml;self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak=0;self.done=threading.Event()
        def poll():
            while not self.done.is_set():
                self.peak=max(self.peak,pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                self.done.wait(.02)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()

    def stop(self):
        self.done.set();self.thread.join();self.nvml.nvmlShutdown()
        return self.peak


class VLLMGenerator:
    def __init__(self, cfg, run):
        self.directory = Path(run)/'engine'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.counter=0
        config=self.directory/'config.json'
        atomic_json(config,cfg)
        (self.directory/'ready.json').unlink(missing_ok=True)
        self.log=open(self.directory/'worker.log','a',buffering=1)
        env={**os.environ,'VLLM_WORKER_MULTIPROC_METHOD':'spawn','TOKENIZERS_PARALLELISM':'false'}
        if cfg.get('vllm_diagnostic_callbacks'):
            # Trusted local worker callbacks only; this adapter starts no API server.
            env['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
        self.process=subprocess.Popen([cfg['vllm_python'],'-u','-m','veriserve_research.vllm_backend',
            str(config)],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True)
        try:
            self.ready=self._receive(self.directory/'ready.json',timeout=1800)
        except BaseException:
            self.close()
            raise

    def _receive(self,path,timeout=1800):
        start=time.monotonic()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM worker exited ({self.process.returncode}); see {self.directory}/worker.log')
            if time.monotonic()-start>timeout:
                raise TimeoutError(f'vLLM worker timed out; see {self.directory}/worker.log')
            time.sleep(.1)
        data=read_json(path)
        if data.get('error'):
            raise RuntimeError(data['error'])
        return data

    def generate(self,prompt_ids,budget,seed,eos_ids,*,ignore_eos=False,capture_path=None):
        import torch
        # HF's caching allocator otherwise retains workspaces while vLLM remaps
        # its sleeping KV pool. This frees only unused blocks, never live tensors.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.counter+=1
        target=self.directory/f'response-{os.getpid()}-{self.counter}.json'
        request=dict(prompt_ids=prompt_ids,budget=budget,seed=seed,eos_ids=eos_ids,
                     ignore_eos=ignore_eos,capture_path=str(capture_path) if capture_path else None,response=str(target.resolve()))
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        return self._receive(target,timeout=7200)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate();self.process.wait(timeout=30)
        self.log.close()


def worker(config_path):
    import torch, vllm, inspect
    from vllm import LLM, SamplingParams
    config_path=Path(config_path)
    cfg=read_json(config_path); directory=config_path.parent
    # Use the exact cached snapshot; no main revision lookup or extra download.
    snapshot=Path(cfg['model_cache_dir'])/('models--'+cfg['model'].replace('/','--'))/'snapshots'/cfg['revisions']['model']
    start=time.perf_counter()
    startup_peak=DevicePeak()
    try:
        llm=LLM(model=str(snapshot),tokenizer=str(snapshot),dtype='bfloat16',
            max_model_len=cfg['max_total_tokens'],max_num_seqs=1,
            gpu_memory_utilization=cfg.get('vllm_gpu_memory_utilization',.65),
            max_num_batched_tokens=cfg['prefill_chunk_size'],
            enable_chunked_prefill=True,mamba_ssm_cache_dtype='float32',
            enable_prefix_caching=False,enable_sleep_mode=True,
            seed=cfg['seed'],generation_config='vllm',
            language_model_only=True,limit_mm_per_prompt={'image':0,'video':0},
            async_scheduling=cfg.get('vllm_async_scheduling',True),
            enforce_eager=cfg.get('vllm_enforce_eager',False))
        init=time.perf_counter()-start
        torch.cuda.synchronize();llm.sleep(level=1)
        atomic_json(directory/'ready.json',dict(status='READY',created=now(),
            versions=dict(vllm=vllm.__version__,torch=torch.__version__),init_and_compile_seconds=init,
            startup_device_peak_bytes=startup_peak.stop(),device_peak_sampling_ms=20,
            sampling_params_signature=str(inspect.signature(SamplingParams)),
            hidden_api='LLM.generate returns token IDs/logprobs, no arbitrary-layer hidden; EAGLE aux hidden is internal',
            snapshot=str(snapshot)))
    except BaseException as exc:
        atomic_json(directory/'ready.json',dict(error=safe_error(exc)));raise
    for line in sys.stdin:
        req=json.loads(line);start=time.perf_counter()
        device_peak=DevicePeak()
        try:
            llm.wake_up();torch.cuda.synchronize();wake=time.perf_counter()-start
            if req.get('capture_path'):
                if not cfg.get('vllm_enforce_eager'):
                    raise RuntimeError('Diagnostic hooks require eager execution; captured CUDA graphs bypass Python hooks')
                llm.apply_model(install_capture)
            params=SamplingParams(temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],
                min_p=cfg['min_p'],presence_penalty=cfg['presence_penalty'],
                repetition_penalty=cfg['repetition_penalty'],frequency_penalty=0.,
                max_tokens=req['budget'],min_tokens=req['budget'] if req['ignore_eos'] else 0,
                ignore_eos=req['ignore_eos'],seed=req['seed'],stop_token_ids=req['eos_ids'],
                detokenize=False)
            torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            output=llm.generate([{'prompt_token_ids':req['prompt_ids']}],params,use_tqdm=False)[0].outputs[0]
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            peak=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
            result=dict(generated_ids=list(output.token_ids),finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,pure_generation_seconds=elapsed,wake_seconds=wake,
                gpu_reserved_mib=int(peak),engine='vllm',seed=req['seed'])
            if req.get('capture_path'):
                result['hidden_capture']=llm.apply_model(lambda model: save_capture(model,req['capture_path']))
            llm.sleep(level=1);torch.cuda.synchronize()
            result['generation_with_wake_and_sleep_seconds']=time.perf_counter()-start
            result['device_peak_bytes']=device_peak.stop()
            result['device_peak_sampling_ms']=20
            atomic_json(Path(req['response']),result)
        except BaseException as exc:
            atomic_json(Path(req['response']),dict(error=safe_error(exc)));raise


def install_capture(model):
    """Diagnostic only: overwrite one endpoint per layer, never retain all tokens."""
    language=getattr(model,'language_model',model)
    decoder=language.model
    model._probe_values={}
    model._probe_hooks=[]
    model._probe_last_input={}
    def input_token(module,args,kwargs):
        tokens=kwargs.get('input_ids',args[0] if args else None)
        model._probe_last_input['token_id']=int(tokens.reshape(-1)[-1])
    def input_position(module,args,kwargs):
        positions=kwargs.get('positions',args[0] if args else None)
        model._probe_last_input['position']=int(positions.reshape(-1)[-1])
    model._probe_hooks.append(decoder.embed_tokens.register_forward_pre_hook(input_token,with_kwargs=True))
    model._probe_hooks.append(decoder.layers[0].register_forward_pre_hook(input_position,with_kwargs=True))
    def hook(index,normalize=False):
        def capture(module,args,output):
            import torch
            if normalize:
                value=output[0] if isinstance(output,tuple) else output
            elif isinstance(output,tuple):
                value=output[0]+output[1] if output[1] is not None else output[0]
            else:value=output
            model._probe_values[index]=value[-1].detach().clone()
        return capture
    for index,module in enumerate(decoder.layers[:-1]):
        model._probe_hooks.append(module.register_forward_hook(hook(index)))
    model._probe_hooks.append(decoder.norm.register_forward_hook(hook(31,True)))
    return dict(layers=len(decoder.layers),decoder_class=type(decoder).__name__)


def save_capture(model,path):
    import numpy as np,torch
    try:
        if sorted(model._probe_values)!=list(range(32)):
            raise RuntimeError('vLLM endpoint hooks were bypassed or layer mapping differs')
        hidden=torch.stack([model._probe_values[i] for i in range(32)]).float().cpu().numpy()[:,None,:]
        np.savez_compressed(path,hidden=hidden)
        return dict(shape=list(hidden.shape),path=path,**model._probe_last_input)
    finally:
        for h in model._probe_hooks:h.remove()
        del model._probe_values,model._probe_hooks


if __name__=='__main__':
    worker(sys.argv[1])
