import json
import sys
import gc
from pathlib import Path

root = Path('/home/asus/projects/Ongoing_project/VeriServe')
sys.path.insert(0, str(root / 'stages/stage1'))
import torch
from veriserve.model import LanguageModel, feedback_text
from veriserve.exp3 import _append_feedback, _consistency_case

run = root / 'stages/stage1/results/current'
snapshot = json.loads((run/'states/exp2b_eval/k2/test-525.json').read_text())['fail_snapshots'][0]
asset = json.loads((run/'config.json').read_text())['assets']['generator']
model = LanguageModel(asset['path'])
output = {'label': 'DIAGNOSTIC_ONLY', 'question_id': 'test-525', 'revision': asset['revision'],
          'torch': torch.__version__, 'attention': model.model.config._attn_implementation,
          'original_case': json.loads((run/'exp3/cache_cases/test-525.json').read_text())}

@torch.inference_mode()
def diagnose():
    raw = snapshot['state']
    prefix = raw['checkpoint_ids']
    rejected = raw['context_ids'][len(prefix):]
    feedback = model.text_ids(feedback_text(snapshot['diagnosis'], snapshot['hint'], raw['checkpoint_step']))
    caches = {}
    logits = {}
    caches['A'], _, _ = model._prefill(prefix + rejected)
    saved = [(layer.keys[:, :, :len(prefix), :].clone(), layer.values[:, :, :len(prefix), :].clone()) for layer in caches['A'].layers]
    caches['A'].crop(-len(rejected))
    crop_exact = all(torch.equal(k, layer.keys) and torch.equal(v, layer.values)
                     for (k, v), layer in zip(saved, caches['A'].layers))
    del saved
    caches['D'], _, _ = model._prefill(prefix)
    prefix_cache_difference = max(float((a.keys.float()-d.keys.float()).abs().max())
                                 for a, d in zip(caches['A'].layers, caches['D'].layers))
    logits['A'] = _append_feedback(model, caches['A'], feedback)
    logits['D'] = _append_feedback(model, caches['D'], feedback)
    caches['B'], logits['B'], _ = model._prefill(prefix + feedback)
    pairs = [('A', 'B'), ('D', 'B'), ('A', 'D')]
    result = {'dtype': str(next(model.model.parameters()).dtype),
              'lengths': {'prefix': len(prefix), 'rejected': len(rejected), 'feedback': len(feedback)},
              'crop_exactly_kept_prefix_tensors': crop_exact,
              'cropped_vs_fresh_prefix_key_max_abs': prefix_cache_difference,
              'initial_logit_differences': {a+b: float((logits[a].float()-logits[b].float()).abs().max()) for a,b in pairs},
              'first_divergence': {}, 'tokens': {name: [] for name in caches}}
    for i in range(32):
        choices = {name: int(torch.argmax(value, dim=-1).item()) for name, value in logits.items()}
        for a,b in pairs:
            if a+b not in result['first_divergence'] and choices[a] != choices[b]:
                record = {'token_number': i+1, 'max_abs_logit_difference': float((logits[a].float()-logits[b].float()).abs().max())}
                for name in (a,b):
                    values, ids = logits[name][0].float().topk(3)
                    record[name] = {'chosen': choices[name], 'chosen_text': model.tokenizer.decode([choices[name]]),
                                    'top3': [{'id': int(t), 'text': model.tokenizer.decode([int(t)]), 'score': float(v)} for t,v in zip(ids,values)],
                                    'top1_top2_margin': float(values[0]-values[1])}
                result['first_divergence'][a+b] = record
        for name, token in choices.items():
            result['tokens'][name].append(token)
            logits[name] = model._next_logits(token, caches[name])
    result['decoded'] = {name: model.tokenizer.decode(tokens) for name,tokens in result['tokens'].items()}
    result['greedy_equal'] = {a+b: result['tokens'][a] == result['tokens'][b] for a,b in pairs}
    return result

try:
    output['repeat_original'] = _consistency_case(model, snapshot, 32, .5)
    print('ORIGINAL_REPRO:', json.dumps(output['repeat_original']), flush=True)
    output['bf16'] = diagnose()
    print('BF16:', json.dumps(output['bf16'], ensure_ascii=False), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.model.float()
    output['fp32'] = diagnose()
    print('FP32:', json.dumps(output['fp32'], ensure_ascii=False), flush=True)
finally:
    model.close()
    Path('/tmp/veriserve_cache_525_diagnosis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
