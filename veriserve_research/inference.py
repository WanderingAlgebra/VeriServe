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


class Session:
    """cache holds ids[:cache_length]; logits are valid only once all ids are fed."""
    def __init__(self, model, ids):
        self.model, self.ids = model, list(ids)
        self.cache, self.logits = None, None
        self.forward(self.ids)

    def forward(self, ids):
        import torch
        if not ids:
            return
        with torch.inference_mode():
            x = torch.tensor([ids], device='cuda:0')
            output = self.model.model(input_ids=x, past_key_values=self.cache, use_cache=True,
                                      return_dict=True)
            self.cache = output.past_key_values
            self.logits = self.model.lm_head(output.last_hidden_state[:, -1:])[0, -1]
        if self.cache.get_seq_length() != len(self.ids):
            raise RuntimeError('KV/context length mismatch')

    def ensure(self):
        missing = self.ids[self.cache.get_seq_length():]
        if missing:
            self.forward(missing)

    def sample(self):
        self.ensure()
        from transformers import RepetitionPenaltyLogitsProcessor
        import torch
        scores=self.logits.float()[None]
        penalty=self.model.generation_config.repetition_penalty
        if penalty!=1.0:
            scores=RepetitionPenaltyLogitsProcessor(penalty)(torch.tensor([self.ids],device=scores.device),scores)
        token = int(scores[0].argmax())
        self.ids.append(token)  # last sampled token may still be absent from KV
        return token

    def rollback(self, checkpoint, feedback):
        checkpoint = list(checkpoint)
        if self.ids[:len(checkpoint)] != checkpoint:
            raise RuntimeError('Rollback checkpoint is not an original token prefix')
        self.cache.crop(min(self.cache.get_seq_length(), len(checkpoint)))
        self.ids = checkpoint + list(feedback)
        missing = self.ids[self.cache.get_seq_length():]
        self.forward(missing)
        return len(missing)

    def clone(self):
        other = object.__new__(Session)
        other.model, other.ids = self.model, self.ids.copy()
        other.cache, other.logits = copy.deepcopy(self.cache), self.logits.clone()
        return other


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for collection; use --fit-only on CPU")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revisions"]["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revisions"]["model"], dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda:0").eval()
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("Model parameters are not uniformly BF16")
    with torch.inference_mode():
        test = model.model(input_ids=torch.tensor([[tokenizer.eos_token_id]], device="cuda"),
                           use_cache=False, output_hidden_states=True)
    if len(test.hidden_states) != model.config.num_hidden_layers + 1:
        raise RuntimeError("Unexpected hidden_states index convention")
    return model, tokenizer


def full_forward(model, ids, positions):
    import torch
    with torch.inference_mode():
        inputs = torch.tensor([ids], device="cuda:0")
        out = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                          use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = np.stack([h[0, positions].float().cpu().numpy() for h in out.hidden_states[1:]])
    return hidden


def extract(model, ids, positions):
    # BF16 kernels depend on total sequence shape. Re-forward each original prefix
    # so the saved endpoint has the same computation as an independent prefix.
    return np.stack([full_forward(model, ids[:p + 1], [p])[:, 0] for p in positions], axis=1)


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


def feature(model, ids):
    """Save only the selected layer; hook layer 18 gives HF hidden_states[19]."""
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
