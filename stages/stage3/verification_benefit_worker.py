"""One vLLM request per subprocess; process exit releases the model before switching."""
import json
import sys
import time
from pathlib import Path


def main(request_path, output_path):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from veriserve_research.vllm_backend import DevicePeak
    from veriserve_research.artifacts.io import atomic_json

    request = json.loads(Path(request_path).read_text())
    if torch.cuda.device_count() != 1:
        raise RuntimeError('This experiment requires exactly one visible CUDA device')
    cfg = request['engine']
    started = time.perf_counter()
    peak = DevicePeak()
    if peak.nvml.nvmlDeviceGetCount() != 1:
        raise RuntimeError('Cannot safely attribute NVML peak memory to the sole CUDA device')
    device_info = peak.nvml.nvmlDeviceGetMemoryInfo(peak.handle)
    device_name = peak.nvml.nvmlDeviceGetName(peak.handle)
    if isinstance(device_name, bytes):
        device_name = device_name.decode('utf-8', errors='replace')
    model = LLM(model=cfg['snapshot'], tokenizer=cfg['snapshot'], dtype='bfloat16',
                max_model_len=cfg['max_model_len'], max_num_seqs=1,
                gpu_memory_utilization=cfg['gpu_memory_utilization'],
                max_num_batched_tokens=cfg['prefill_chunk_size'], enable_chunked_prefill=True,
                mamba_ssm_cache_dtype='float32', enable_prefix_caching=False,
                generation_config='vllm', language_model_only=True, disable_log_stats=False,
                limit_mm_per_prompt={'image': 0, 'video': 0},
                logits_processors=['stages.stage3.verification_benefit_logits.RestoredPresence'])
    torch.cuda.synchronize()
    init_seconds = time.perf_counter()-started
    sampling = request['sampling']
    params = SamplingParams(temperature=sampling['temperature'], top_p=sampling['top_p'],
        top_k=sampling['top_k'], min_p=sampling['min_p'], presence_penalty=sampling['presence_penalty'],
        repetition_penalty=sampling['repetition_penalty'], frequency_penalty=0.,
        max_tokens=request['max_tokens'], seed=request['seed'], detokenize=False,
        stop_token_ids=request['eos_ids'],
        extra_args={'restored_presence_history': request['presence_history']})
    start_generate = time.perf_counter()
    response = model.generate([{'prompt_token_ids': request['input_ids']}], params, use_tqdm=False)[0]
    result = response.outputs[0]
    torch.cuda.synchronize()
    generated_seconds = time.perf_counter()-start_generate
    stats = response.metrics
    prefill = (stats.first_token_ts - stats.scheduled_ts
               if stats and stats.first_token_ts and stats.scheduled_ts else None)
    atomic_json(Path(output_path), {'generated_ids': list(result.token_ids),
        'finish_reason': result.finish_reason, 'stop_reason': result.stop_reason,
        'init_seconds': init_seconds, 'prefill_and_generation_seconds': generated_seconds,
        'prefill_first_token_seconds': prefill,
        'device_peak_bytes': peak.stop(), 'device_peak_sampling_ms': 20,
        'device_total_bytes':device_info.total, 'device_name':device_name,
        'vllm_version': vllm.__version__, 'torch_version': torch.__version__})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
