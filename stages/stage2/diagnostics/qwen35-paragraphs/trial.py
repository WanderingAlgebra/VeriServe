"""Three manifest-first smoke prompts, short diagnostic only; no formal grading/features."""
import copy
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from veriserve_research import ROOT
from veriserve_research.artifacts.io import atomic_json, digest, now, read_json, git, safe_error
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.inference import Session, generation_seed, load_model, prompt_tokens
from veriserve_research.trajectory import decode, thinking_error
from check import paragraphs, self_check, lexical_flags, token_endpoint


def main():
    folder = Path(__file__).resolve().parent
    run = ROOT/'stages/stage2/runs/v2-20261013-2f15553ff1e5'
    manifest = read_json(run/'manifest.json')
    cfg = copy.deepcopy(manifest['config'])
    cfg['prompt'] = ('Please reason step by step. After thinking, output only one line: '
                     'Final answer: \\boxed{your final answer}.\n\n{problem}')
    cfg['max_new_tokens'] = 2048
    result = dict(validation_only=True,scientific_results=False,new_smoke_passed=False,
        created=now(),status='RUNNING',config=cfg,script_sha256=digest(__file__),
        boundary_script_sha256=digest(folder/'check.py'),source_manifest_sha256=digest(run/'manifest.json'),
        code_commit=git('rev-parse','HEAD'),execution_source_hashes=source_hashes('probe'),
        selection='First three frozen smoke questions in manifest order; no outcome selection.',
        limitations='Short diagnostic limit; no claim of full-budget completion, reasoning correctness or human audit.',
        questions=[])
    atomic_json(folder/'short_trial.json',result)
    model, tokenizer = load_model(cfg)
    self_check(tokenizer)
    result['loading'] = model.research_loading_info
    result['eos_token_ids'] = model.generation_config.eos_token_id
    for row in manifest['splits']['smoke'][:3]:
        prompt = prompt_tokens(cfg,tokenizer,row['problem'])
        seed = generation_seed(cfg['seed'],row['unique_id'])
        session = Session(model,prompt,cfg,seed)
        generated = []
        start = time.monotonic()
        for _ in range(cfg['max_new_tokens']):
            token = session.sample()
            generated.append(token)
            if token in model.generation_config.eos_token_id:
                break
        raw = decode(tokenizer,generated)
        record = dict(unique_id=row['unique_id'],generation_seed=seed,prompt_ids=prompt,
            generated_ids=generated,raw_text=raw,seconds=time.monotonic()-start,
            stage='GENERATED',completion='EOS' if generated[-1] in model.generation_config.eos_token_id else 'SHORT_DIAGNOSTIC_LIMIT')
        result['questions'].append(record)
        atomic_json(folder/'short_trial.json',result)
        spans, tail = paragraphs(raw)
        points = []
        for a,b,c,d in spans:
            point = token_endpoint(tokenizer,generated,raw,(a,b,c,d))
            point['flags'] = lexical_flags(raw[:b])
            points.append(point)
        record.update(closed_paragraphs=len(spans),points=points,stage='PARSED',
            thinking_closed='</think>' in raw,thinking_error=thinking_error(raw,terminal=True),
            undelimited_tail_chars=len(tail))
        atomic_json(folder/'short_trial.json',result)
        print(f"{row['unique_id'][:12]} tokens={len(generated)} paragraphs={len(spans)} completion={record['completion']}",flush=True)
        del session
        gc.collect()
    result.update(status='DIAGNOSTIC_COMPLETE',finished=now())
    atomic_json(folder/'short_trial.json',result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = Path(__file__).with_name('short_trial.json')
        result = read_json(path,{})
        result.update(status='FAILED',error=safe_error(exc),finished=now())
        atomic_json(path,result)
        raise
