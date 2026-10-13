"""Resident inference, original-prefix hidden extraction and native KV crop."""
from __future__ import annotations
import copy
import gc
import time
import numpy as np
from .artifacts.io import atomic_json
from .artifacts.legacy import prm_components
from .artifacts.provenance import source_hashes


def clock():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


SAMPLING_KEYS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


def generation_seed(seed, uid):
    from .artifacts.io import stable_hash
    return int(stable_hash([seed, uid])[:15], 16)


def sampling_config(cfg):
    defaults = dict(do_sample=False, temperature=1., top_p=1., top_k=0, min_p=0.,
                    presence_penalty=0., repetition_penalty=1.)
    return {k: cfg.get(k, v) for k, v in defaults.items()}


def prompt_tokens(cfg, tokenizer, problem):
    kwargs = {"enable_thinking": cfg["enable_thinking"]} if "enable_thinking" in cfg else {}
    messages = []
    if cfg.get('system_prompt'):
        messages.append({'role': 'system', 'content': cfg['system_prompt']})
    messages.extend(cfg.get('format_examples',[]))
    messages.append({"role": "user", "content": cfg["prompt"].replace("{problem}", problem)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs)


def sample_token(logits, ids, prompt_length, settings, rng):
    import torch
    from transformers import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper
    scores = logits.float()[None].clone()
    tokens = torch.tensor([ids], device=scores.device)
    if settings['repetition_penalty'] != 1.:
        scores = RepetitionPenaltyLogitsProcessor(settings['repetition_penalty'])(tokens, scores)
    # Presence applies once to each token in this request's generated/feedback suffix,
    # excluding the question prompt; shared by stage2, stage3 and the verifier.
    if settings['presence_penalty'] and len(ids) > prompt_length:
        seen = torch.tensor(ids[prompt_length:], device=scores.device).unique()
        scores[0, seen] -= settings['presence_penalty']
    if not settings['do_sample']:
        return int(scores[0].argmax())
    for processor in (TemperatureLogitsWarper(settings['temperature']),
                      TopKLogitsWarper(settings['top_k']) if settings['top_k'] else None,
                      TopPLogitsWarper(settings['top_p']) if settings['top_p'] < 1 else None,
                      MinPLogitsWarper(settings['min_p']) if settings['min_p'] else None):
        if processor is not None:
            scores = processor(tokens, scores)
    return int(torch.multinomial(torch.softmax(scores[0], dim=-1), 1, generator=rng))


class Session:
    """One request, independent RNG and immutable full-state checkpoints."""
    def __init__(self, model, ids, cfg=None, seed=0):
        import torch
        self.model, self.ids = model, list(ids)
        self.prompt_length = len(ids)
        self.settings = sampling_config(cfg or {'repetition_penalty': model.generation_config.repetition_penalty})
        self.chunk_size = (cfg or {}).get('prefill_chunk_size', max(1, len(ids)))
        self.rng = torch.Generator(device=model.device).manual_seed(seed)
        self.cache, self.logits = None, None
        self.cached_length = 0
        self.forward(self.ids)

    def forward(self, ids):
        import torch
        if not ids:
            return
        with torch.inference_mode():
            for lo in range(0, len(ids), self.chunk_size):
                part = ids[lo:lo+self.chunk_size]
                x = torch.tensor([part], device=self.model.device)
                output = self.model.model(input_ids=x, past_key_values=self.cache, use_cache=True,
                                          return_dict=True)
                self.cache = output.past_key_values
                self.logits = self.model.lm_head(output.last_hidden_state[:, -1:])[0, -1]
                self.cached_length += len(part)
                del output
        if self.cache.get_seq_length() != self.cached_length or self.cached_length != len(self.ids):
            raise RuntimeError('Cache/context length mismatch')

    def ensure(self):
        missing = self.ids[self.cached_length:]
        if missing:
            self.forward(missing)
        return len(missing)

    def sample(self):
        self.ensure()
        token = sample_token(self.logits, self.ids, self.prompt_length, self.settings, self.rng)
        self.ids.append(token)
        return token

    def snapshot(self):
        if self.cached_length != len(self.ids):
            raise RuntimeError('Ensure exact prefix before taking checkpoint; account for the forward')
        return dict(ids=self.ids.copy(), cache=copy.deepcopy(self.cache), logits=self.logits.clone(),
                    cached_length=self.cached_length)

    def restore(self, snapshot, feedback=()):
        # Release the revoked active state before copying; never alias snapshot tensors.
        self.cache, self.logits = None, None
        self.ids = snapshot['ids'].copy()
        self.cache = copy.deepcopy(snapshot['cache'])
        self.logits = snapshot['logits'].clone()
        self.cached_length = snapshot['cached_length']
        self.ids.extend(feedback)
        self.forward(list(feedback))
        # Deliberately do not rewind RNG or cumulative accounting on algorithmic FAIL.
        return len(feedback)

    def rollback(self, checkpoint, feedback):
        if any('linear' in kind for kind in getattr(self.model.config, 'layer_types', [])):
            raise RuntimeError('Hybrid caches require restore(full_snapshot), never cache.crop')
        checkpoint = list(checkpoint)
        if self.ids[:len(checkpoint)] != checkpoint:
            raise RuntimeError('Rollback checkpoint is not an original token prefix')
        self.cache.crop(min(self.cached_length, len(checkpoint)))
        self.cached_length = self.cache.get_seq_length()
        self.ids = checkpoint + list(feedback)
        missing = self.ids[self.cached_length:]
        self.forward(missing)
        return len(missing)

    def clone(self):
        import torch
        other = object.__new__(Session)
        other.__dict__ = self.__dict__.copy()
        other.ids = self.ids.copy()
        other.cache, other.logits = copy.deepcopy(self.cache), self.logits.clone()
        other.rng = torch.Generator(device=self.model.device)
        other.rng.set_state(self.rng.get_state())
        return other


def load_model(cfg):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for collection; use --fit-only on CPU")
    cache = cfg.get('model_cache_dir')
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revisions"]["tokenizer"], cache_dir=cache)
    config = AutoConfig.from_pretrained(cfg['model'], revision=cfg['revisions']['model'], cache_dir=cache)
    klass = AutoModelForCausalLM
    if config.model_type == 'qwen3_5':
        from transformers import Qwen3_5ForCausalLM
        klass, config = Qwen3_5ForCausalLM, config.text_config
    model, info = klass.from_pretrained(
        cfg["model"], config=config, revision=cfg["revisions"]["model"], dtype=torch.bfloat16,
        attn_implementation=cfg.get('attention', 'sdpa'), cache_dir=cache,
        output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f"Incomplete text checkpoint loading: {info}")
    unexpected = [k for k in info.get('unexpected_keys', [])
                  if not k.startswith(('model.visual.', 'mtp.'))]
    if unexpected:
        raise RuntimeError(f'Unexpected text weights: {unexpected}')
    model = model.to("cuda:0").eval()
    if cfg.get('enable_thinking'):
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        # Text subconfig may name endoftext; the native chat turn ends at im_end.
        chat_end = tokenizer.get_vocab().get('<|im_end|>')
        model.generation_config.eos_token_id = list(dict.fromkeys(
            token for token in [*eos, tokenizer.eos_token_id, chat_end] if token is not None))
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Model parameters are not uniformly BF16")
    with torch.inference_mode():
        test = model.model(input_ids=torch.tensor([[tokenizer.eos_token_id]], device="cuda"),
                           use_cache=False, output_hidden_states=True)
    if len(test.hidden_states) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Unexpected hidden_states index convention")
    for key, actual in [('expected_hidden_layers',model.config.num_hidden_layers),
                        ('expected_hidden_size',model.config.hidden_size)]:
        if key in cfg and cfg[key] != actual:
            raise RuntimeError(f'{key} differs: {actual}')
    final_norm = bool(torch.equal(test.hidden_states[-1],test.last_hidden_state))
    if not final_norm:
        raise RuntimeError('Last hidden output differs from post-norm decoder output')
    model.research_loading_info = dict(class_name=type(model).__name__, missing_keys=sorted(info.get('missing_keys', [])),
        unexpected_keys=sorted(info.get('unexpected_keys', [])), layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size, final_hidden_includes_norm=final_norm,
        actual_revision=cfg['revisions']['model'])
    return model, tokenizer


def full_forward(model, ids, positions):
    import torch
    with torch.inference_mode():
        inputs = torch.tensor([ids], device="cuda:0")
        out = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                          use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = np.stack([h[0, positions].float().cpu().numpy() for h in out.hidden_states[1:]])
    return hidden


def extract(model, ids, positions, layer=None):
    # BF16 kernels depend on total sequence shape. Re-forward each original prefix
    # so the saved endpoint has the same computation as an independent prefix.
    hidden = np.stack([full_forward(model, ids[:p + 1], [p])[:, 0] for p in positions], axis=1)
    return hidden if layer is None else hidden[layer-1:layer]


def causal_check(model, record, cfg):
    import torch
    steps = record["boundaries"]["steps"]
    if not steps:
        return {"passed": None, "applicable": False, "reason": "no intermediate position; trajectory retained"}
    ids = record["prompt_ids"] + record["generated_ids"]
    position = len(record["prompt_ids"]) + steps[0]["end_token_index"]
    original = extract(model, ids, [position])
    changed = ids[:position + 1] + [model.config.vocab_size // 2] * (len(ids) - position - 1)
    future = extract(model, changed, [position])
    # Independent decoder call and explicit last-token indexing catch offset bugs.
    with torch.inference_mode():
        inputs = torch.tensor([ids[:position + 1]], device="cuda:0")
        reference = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                use_cache=False, output_hidden_states=True, return_dict=True)
        prefix = np.stack([h[0, -1:].float().cpu().numpy() for h in reference.hidden_states[1:]])
        previous = np.stack([h[0, -2:-1].float().cpu().numpy() for h in reference.hidden_states[1:]])
    del inputs, reference
    single_full = full_forward(model, ids, [position])
    single_future = full_forward(model, changed, [position])
    checks = {}
    for name, left, other in (("changed_future", original, future),
                               ("independent_prefix", original, prefix),
                               ("single_full_changed_future", single_full, single_future),
                               ("single_full_vs_prefix_diagnostic", single_full, prefix),
                               ("previous_token_negative_control", original, previous)):
        diff = np.abs(left - other)
        checks[name] = {"max_abs": float(diff.max()), "rms": float(np.sqrt(np.mean(diff ** 2))),
                        "max_abs_per_layer": diff.max(axis=(1, 2)).tolist(),
                        "passed": bool(np.allclose(left, other, atol=cfg["causal_atol"],
                                                   rtol=cfg["causal_rtol"]))}
    checks["single_full_vs_prefix_diagnostic"].update(
        gating=False, used_for_features=False,
        note="Retained full-once BF16 shape discrepancy; features use independent original prefixes")
    checks["previous_token_negative_control"]["detected"] = not checks["previous_token_negative_control"]["passed"]
    checks.update(passed=all(checks[k]["passed"] for k in (
                      "changed_future", "independent_prefix", "single_full_changed_future"))
                  and checks["previous_token_negative_control"]["detected"],
                  applicable=True, position=position, extraction="independent_original_prefixes",
                  atol=cfg["causal_atol"], rtol=cfg["causal_rtol"],
                  tolerance_note="BF16 rounding: atol=1/32, rtol=one BF16 relative ULP; report raw errors")
    torch.cuda.empty_cache()
    return checks


def load_resident(cfg, run):
    import torch
    from huggingface_hub import snapshot_download
    ProcessRewardModel, SYSTEM, _, _ = prm_components()
    from .intervention import protocol
    from .artifacts import storage
    class ResidentPRM(ProcessRewardModel):
        def input_ids(self,question,accepted,new):
            steps,prior=protocol.prm_steps(accepted,new)
            if not steps:
                return [],[],0
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':question},
                      {'role':'assistant','content':'<extra_0>'.join(body for _,body in steps)+'<extra_0>'}]
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
            if sum(t==self.separator_id for t in ids)!=len(steps):
                raise RuntimeError('PRM separator/step count mismatch')
            return ids,[n for n,_ in steps],prior
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Required BF16 CUDA unavailable; no model/precision substitution')
    generator, tokenizer = load_model(cfg)
    if cfg.get('verifier', {}).get('kind') == 'generative':
        verifier = GenerativeVerifier(cfg)
        generative_capacity_check(cfg,generator,tokenizer,verifier,run)
        return generator, tokenizer, verifier
    path = snapshot_download(cfg['verifier']['id'], revision=cfg['revisions']['verifier'])
    verifier = ResidentPRM(path, threshold=cfg['verifier']['threshold'])
    if verifier.tokenizer.name_or_path != path:
        raise RuntimeError('Verifier tokenizer revision not fixed')
    start = clock()
    warm = Session(generator, [tokenizer.eos_token_id] * 16)
    warm.sample()
    warm.ensure()
    verifier.verify('Compute 1+1.', '', 'Step 1: 1+1=2.\n\nFinal answer: \\boxed{2}')
    del warm
    gc.collect()
    torch.cuda.empty_cache()
    # Real simultaneous residency and cache/headroom at the frozen limits.
    torch.cuda.reset_peak_memory_stats()
    big = Session(generator, [tokenizer.eos_token_id] * cfg['max_total_tokens'])
    prm_ids = [verifier.tokenizer.eos_token_id] * cfg['max_prm_input_tokens']
    with torch.inference_mode():
        output = verifier.model(input_ids=torch.tensor([prm_ids], device=verifier.device), use_cache=False)
        if not torch.isfinite(output.logits).all():
            raise RuntimeError('PRM capacity forward produced non-finite logits')
    capacity = {'status': 'PASS', 'warmup_capacity_seconds': clock()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'free_total_bytes': list(torch.cuda.mem_get_info()),
                'generator_context_tokens': cfg['max_total_tokens'],
                'prm_input_tokens': cfg['max_prm_input_tokens'],
                'simultaneous_residency': True, 'source_hashes': source_hashes(
                    "high-low" if cfg.get("experiment_type") == "within_question_high_low" else "fixed-step")}
    storage.archive_diagnostic(run, 'capacity_check.json')
    atomic_json(run / 'capacity_check.json', capacity)
    del big, output, prm_ids
    gc.collect()
    torch.cuda.empty_cache()
    return generator, tokenizer, verifier


def feature(model, ids, layer=None):
    """Save only the selected layer; hook layer 18 gives HF hidden_states[19]."""
    if layer is not None:
        return extract(model, ids, [len(ids)-1], layer=layer)[0, 0]
    import torch
    values = []
    def capture(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        values.append(h[0, -1].float().cpu().numpy().copy())
    hook = model.model.layers[18].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            x = torch.tensor([ids], device='cuda:0')
            model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                        output_hidden_states=False, return_dict=True)
    finally:
        hook.remove()
    if len(values) != 1:
        raise RuntimeError('Selected hidden layer was not captured exactly once')
    return values[0]


class GenerativeVerifier:
    """Resident text model; quoted problem/prefix only, independent request RNG."""
    def __init__(self, cfg):
        self.cfg = cfg
        vc = cfg['verifier']
        model_cfg = {**cfg, 'model': vc['id'], 'expected_hidden_layers': 32,
                     'expected_hidden_size': 4096,
                     'revisions': {'model': cfg['revisions']['verifier'],
                                   'tokenizer': cfg['revisions']['verifier_tokenizer']}}
        self.model, self.tokenizer = load_model(model_cfg)

    def input_ids(self, problem, prefix, current_step, feedback, kind):
        import json
        from .intervention.protocol import VERIFIER_PROMPT
        from .trajectory import verification_units, PARAGRAPHS
        mode = getattr(self,'cfg',{}).get('boundary_mode','steps')
        prompt = VERIFIER_PROMPT
        if mode == PARAGRAPHS:
            numbers, units = verification_units(prefix,mode,terminal=kind=='terminal')
            if not numbers or current_step != numbers[-1]:
                raise ValueError('Verifier checkpoint numbering differs from the exact prefix')
            prefix = units
            prompt += ('\nThe quoted prefix is an ordered array of externally numbered checkpoint units. '
                       'Concatenating quoted_text restores the original prefix exactly. Numbers are metadata, '
                       'not model-generated Step markers. A paragraph need not be one complete mathematical step. '
                       'Use checkpoint_number as first_error_step; do not confuse it with numbers inside the quoted text.')
        message = json.dumps(dict(problem=problem, quoted_prefix=prefix, current_step=current_step,
                                  retained_feedback=feedback, check_type=kind), ensure_ascii=False)
        return self.tokenizer.apply_chat_template(
            [{'role':'system','content':prompt}, {'role':'user','content':message}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)

    def verify(self, ids, uid, numbers, checkpoint_step, kind, audit_repeat=0):
        from .artifacts.io import stable_hash
        from .trajectory import decode
        from .intervention.protocol import parse_verdict
        vc = self.cfg['verifier']
        seed_input = [uid, ids, kind]
        if audit_repeat:
            seed_input.append(['audit_repeat',audit_repeat])
        seed = generation_seed(self.cfg['seed'], stable_hash(seed_input))
        session = Session(self.model, ids, self.cfg, seed)
        output = []
        completion = 'LENGTH_TRUNCATED'
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        start = clock()
        try:
            for _ in range(min(vc['max_new_tokens'], vc['max_total_tokens']-len(ids))):
                token = session.sample()
                output.append(token)
                if token in eos:
                    completion = 'EOS'
                    break
            raw = decode(self.tokenizer, output[:-1] if output and output[-1] in eos else output)
            result = parse_verdict(raw, numbers, checkpoint_step, completion)
            result.update(raw_text=raw, generated_ids=output, generation_seed=seed,
                          input_tokens=len(ids), output_tokens=len(output), sampling=sampling_config(self.cfg),
                          decode_seconds=clock()-start, verifier_output_kind='generative_native_thinking_json')
            return result
        finally:
            del session
            gc.collect()


def migration_self_check():
    """CPU checks of the shared sampler, full snapshots and native Thinking parsing."""
    import json
    import torch
    from types import SimpleNamespace
    from .trajectory import boundaries, thinking_view
    from .intervention.protocol import parse_verdict
    from .artifacts.io import read_json
    class Tokenizer:
        all_special_ids = [999, 998]
        pieces = ['Step 1:', ' x', '.\n\nSt', 'ep 2:', ' y', '</think>', '\nFinal answer: \\boxed{2}']
        def get_vocab(self):
            return {'</think>': 998}
        def decode(self, ids, **kwargs):
            return ''.join('</think>' if i == 998 else '<eos>' if i == 999 else self.pieces[i] for i in ids)
    tok = Tokenizer()
    text, bounds = boundaries(tok, [0,1,2,3,4,998,6,999])
    assert bounds['steps'][0]['prefix_token_count'] == 2 and bounds['steps'][0]['boundary_aligned_back']
    assert len(bounds['steps']) == 2 and not bounds['format_errors']
    assert text.count('Final answer:') == 1 and len(text) == len(tok.decode([0,1,2,3,4,998,6]))
    checks = ['Thinking tags permitted; cross-boundary endpoints preserve original token offsets']
    good = dict(verdict='FAIL',first_error_step=2,diagnosis='Unresolved sign error.',hint='Recompute the sign.')
    raw = 'Thinking includes an irrelevant {"verdict":"PASS"}</think>' + json.dumps(good)
    assert parse_verdict(raw,[1,2])['verdict'] == 'FAIL'
    assert parse_verdict(raw,[1,2],2)['effective_verdict'] == 'UNCERTAIN'
    for bad in ('{}', '</think>{broken}', '</think>'+json.dumps({**good,'first_error_step':True}),
                '</think>'+json.dumps({**good,'first_error_step':3}), '</think>'+json.dumps({**good,'hint':'x'*301})):
        assert parse_verdict(bad,[1,2])['parser_error']
    assert parse_verdict(raw,[1,2],completion='LENGTH_TRUNCATED')['verdict'] == 'UNCERTAIN'
    checks.append('strict final JSON only; malformed/truncated/range/type/conflict cases cannot PASS')
    scores = torch.tensor([4.,3.,2.,1.])
    settings = sampling_config(dict(do_sample=True,temperature=1.,top_p=.95,top_k=2,presence_penalty=1.5))
    a = torch.Generator().manual_seed(42)
    b = torch.Generator().manual_seed(42)
    assert [sample_token(scores,[0,1],1,settings,a) for _ in range(20)] == [sample_token(scores,[0,1],1,settings,b) for _ in range(20)]
    base = sampling_config({})
    assert sample_token(scores,[0,0],1,base,None) == 0
    assert sample_token(scores,[0,0],1,{**base,'presence_penalty':2.},None) == 1
    assert sample_token(scores,[0],1,{**base,'presence_penalty':2.},None) == 0
    checks.append('independent sampling RNG; presence penalty works once on suffix only')
    class Cache:
        def __init__(self):
            self.layers = [SimpleNamespace(keys=torch.ones(1),values=torch.ones(1),
                conv_states=torch.ones(1),recurrent_states=torch.ones(1,dtype=torch.float32),has_previous_state=True)]
    session = object.__new__(Session)
    session.model = SimpleNamespace(device=torch.device('cpu'))
    session.ids, session.cached_length = [1,2], 2
    session.logits, session.cache = scores.clone(), Cache()
    session.rng = torch.Generator().manual_seed(99)
    snapshot = session.snapshot()
    session.cache.layers[0].recurrent_states.add_(10)
    session.cache.layers[0].conv_states.add_(20)
    assert snapshot['cache'].layers[0].recurrent_states.item() == 1
    session.rng.set_state(torch.Generator().manual_seed(123).get_state())
    rng_state = session.rng.get_state().clone()
    assert session.restore(snapshot) == 0
    assert torch.equal(rng_state,session.rng.get_state())
    session.cache.layers[0].keys.add_(10)
    assert snapshot['cache'].layers[0].keys.item() == 1
    assert session.cache.layers[0].recurrent_states.dtype == torch.float32
    checks.append('full tensor snapshots immutable, preserve recurrent dtype and do not rewind FAIL RNG')
    cfg = read_json(__import__('pathlib').Path(__file__).resolve().parent.parent/'stages/stage2/qwen35_4b_omni_config.json')
    assert generation_seed(cfg['seed'],'question') == generation_seed(cfg['seed'],'question')
    assert generation_seed(cfg['seed'],'question') != generation_seed(cfg['seed'],'other')
    return dict(status='PASS',checks=checks)


def cache_state(cache):
    """Expose all tensor/flag state in the locked cache implementation for diagnostics."""
    import torch
    result = {}
    def visit(value, path):
        if isinstance(value, torch.Tensor):
            result[path] = value.detach().clone()
        elif isinstance(value, (list,tuple)):
            for i,v in enumerate(value):
                visit(v,f'{path}/{i}')
        elif isinstance(value, dict):
            for k,v in value.items():
                visit(v,f'{path}/{k}')
        elif isinstance(value,(bool,int,float,str,type(None))):
            result[path] = value
        elif hasattr(value,'__dict__'):
            for k,v in vars(value).items():
                visit(v,f'{path}/{k}')
    visit(cache,'cache')
    return result


def compare_state(left, right, atol=0., rtol=0.):
    import torch
    if set(left) != set(right):
        raise AssertionError('Cache field set changed')
    details = {}
    for key,a in left.items():
        b = right[key]
        if isinstance(a,torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise AssertionError('Cache shape/dtype mismatch: '+key)
            diff = (a.float()-b.float()).abs()
            details[key] = dict(max_abs=float(diff.max()) if diff.numel() else 0.,
                                rms=float(diff.square().mean().sqrt()) if diff.numel() else 0.,
                                dtype=str(a.dtype),shape=list(a.shape),
                                passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
        else:
            details[key] = dict(passed=a==b, left=a, right=b)
    return dict(passed=all(v['passed'] for v in details.values()), fields=details,atol=atol,rtol=rtol)


def snapshot_gpu_check(session, feedback_ids, test_ids, cfg):
    """Compare revoked paths with pristine and independent original-schedule replays."""
    import torch
    session.ensure()
    snapshot = session.snapshot()
    pristine = cache_state(snapshot['cache'])
    def append(active, ids):
        active.ids.extend(ids)
        active.forward(ids)
    def logit_error(left,right):
        delta=(left.logits.float()-right.logits.float()).abs()
        return dict(max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),
                    passed=bool(torch.allclose(left.logits,right.logits,atol=cfg['causal_atol'],rtol=cfg['causal_rtol'])))
    control=session.clone()
    append(control,feedback_ids)
    append(control,test_ids)
    # Record normal forward-schedule differences BEFORE creating either failed suffix.
    # BF16 chunk prefill and recurrent decode are distinct numerical computations.
    one_shot=Session(session.model,snapshot['ids']+feedback_ids+test_ids,cfg,0)
    prefill_baseline=dict(cache=compare_state(cache_state(control.cache),cache_state(one_shot.cache),
                                             cfg['causal_atol'],cfg['causal_rtol']),
                          logits=logit_error(control,one_shot),gating=False,
                          measured_before_revoked_suffix=True,
                          note='One-shot prefill vs original prompt-prefill/token-decode schedule; raw failures retained')
    del one_shot
    # Independent exact-prefix reconstruction repeats the frozen generation schedule,
    # without sampling, a saved cache, or ANY revoked token. No tolerance is changed.
    recomputed=Session(session.model,snapshot['ids'][:session.prompt_length],cfg,0)
    for token in snapshot['ids'][session.prompt_length:]:
        append(recomputed,[token])
    append(recomputed,feedback_ids)
    append(recomputed,test_ids)
    trials=[]
    for suffix in ([7,8,9],[13,14,15,16]):
        active=session.clone()
        append(active,suffix)
        immutable=compare_state(pristine,cache_state(snapshot['cache']))
        assert immutable['passed'],immutable
        before_rng=active.rng.get_state().clone()
        charged=active.restore(snapshot,feedback_ids)
        assert charged==len(feedback_ids) and torch.equal(before_rng,active.rng.get_state())
        append(active,test_ids)
        restored=compare_state(cache_state(active.cache),cache_state(control.cache))
        logits=float((active.logits.float()-control.logits.float()).abs().max())
        assert restored['passed'] and logits==0.,(restored,logits)
        replay=compare_state(cache_state(active.cache),cache_state(recomputed.cache),cfg['causal_atol'],cfg['causal_rtol'])
        replay_logits=logit_error(active,recomputed)
        trials.append(dict(failed_suffix=suffix,immutable=immutable,restored=restored,
                           restored_logit_max_abs=logits,recompute=replay,recompute_logits=replay_logits,
                           recompute_mode='fresh original prompt prefill then teacher-forced per-token prefix',
                           feedback_forward_tokens=charged))
        del active
        if not replay['passed'] or not replay_logits['passed']:
            return dict(status='FAILED',trials=trials,one_shot_prefill_baseline=prefill_baseline,
                        reason='Independent exact-prefix replay failed at original fixed tolerances')
    return dict(status='PASS',trials=trials,one_shot_prefill_baseline=prefill_baseline,
                synthetic=True,excluded_from_statistics=True)


def generative_capacity_check(cfg, generator, tokenizer, verifier, run):
    import torch
    from .artifacts.io import read_json,stable_hash,now,safe_error
    path = run/'capacity_check.json'
    identity = dict(config_hash=stable_hash(cfg),source_hashes=source_hashes('high-low'))
    saved = read_json(path,{})
    if saved.get('status') == 'PASS' and all(saved.get(k)==v for k,v in identity.items()):
        return saved
    torch.cuda.reset_peak_memory_stats()
    start = clock()
    result = dict(status='RUNNING',created=now(),**identity)
    atomic_json(path,result)
    prompt=active=verify_session=prompt_snapshot=pass_snapshot=None
    try:
        prompt = Session(generator,prompt_tokens(cfg,tokenizer,'Compute 1 + 1.'),cfg,0)
        prompt_snapshot = prompt.snapshot()
        del prompt
        prompt = None
        active = Session(generator,[tokenizer.eos_token_id]*cfg['max_total_tokens'],cfg,0)
        pass_snapshot = active.snapshot()
        vc=cfg['verifier']
        verify_session = Session(verifier.model,[verifier.tokenizer.eos_token_id]*vc['max_input_tokens'],cfg,0)
        for _ in range(vc['max_new_tokens']):
            verify_session.sample()
        result.update(status='PASS',seconds=clock()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      free_total_bytes=list(torch.cuda.mem_get_info()),generator_context=cfg['max_total_tokens'],
                      verifier_input=vc['max_input_tokens'],verifier_output=vc['max_new_tokens'],
                      prefill_chunk_size=cfg['prefill_chunk_size'],simultaneous_residency=True,
                      two_immutable_checkpoints=True,precision='bfloat16')
        atomic_json(path,result)
        return result
    except BaseException as exc:
        result.update(status='FAILED',error=safe_error(exc),seconds=clock()-start)
        atomic_json(path,result)
        raise
    finally:
        del prompt,active,verify_session,prompt_snapshot,pass_snapshot
        gc.collect()
        torch.cuda.empty_cache()
