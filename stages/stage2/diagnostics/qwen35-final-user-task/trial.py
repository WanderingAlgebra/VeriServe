"""One fixed, well-posed smoke question; original strict final-line rule retained."""
import copy
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from veriserve_research import ROOT
from veriserve_research.artifacts.backup import backup_probe
from veriserve_research.artifacts.io import read_json,atomic_json,now,digest,safe_error
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.inference import load_model,Session,prompt_tokens,generation_seed,sampling_config
from veriserve_research.trajectory import boundaries,decode,thinking_error,PARAGRAPHS
from veriserve_research.intervention.protocol import terminal

HERE=Path(__file__).resolve().parent
RUN=ROOT/'stages/stage2/runs/v2-20261013-da5c802342ee'
OUTPUT=HERE/'trial.json'


def main():
    global HERE, OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['user_suffix','channel_contract'],default='user_suffix')
    args=parser.parse_args()
    if args.variant=='channel_contract':
        HERE=HERE/'channel-contract'
        OUTPUT=HERE/'trial.json'
    if OUTPUT.exists():
        raise FileExistsError('Preserve the existing diagnostic; do not repeat to select an outcome')
    manifest=read_json(RUN/'manifest.json')
    row=manifest['splits']['smoke'][1]
    cfg=copy.deepcopy(manifest['config'])
    cfg.update(system_prompt=None,format_examples=[],max_new_tokens=49152,
        prompt=('Solve the following mathematics problem. Use blank lines between reasoning stages.\n\n'
                '{problem}\n\nYour final response must be only this single line, replacing ANSWER with your answer:\n'
                'Final answer: \\boxed{ANSWER}\nDo not include explanations in the final response.'))
    if args.variant=='channel_contract':
        cfg['prompt']=('{problem}\n\n'
            'Solve this problem in your reasoning channel. Separate reasoning stages with blank lines. '
            'The final answer channel is consumed by an exact-line parser. Its entire content must be '
            'exactly one line: Final answer: \\boxed{ANSWER}, replacing ANSWER with your answer. '
            'All derivations and explanations belong in the reasoning channel. '
            'Do not repeat or summarize the solution in the final answer channel.')
    assert sampling_config(cfg)==sampling_config(manifest['config'])
    atomic_json(HERE/'trial_config.json',cfg)
    result=dict(status='PREPARED',created=now(),pid=os.getpid(),validation_only=True,scientific_results=False,
        new_smoke_passed=False,config=cfg,source_hashes=source_hashes('probe'),driver_sha256=digest(__file__),
        source_manifest_sha256=digest(RUN/'manifest.json'),unique_id=row['unique_id'],problem=row['problem'],
        selection='Original second smoke question: a well-posed complex-format failure; no test/audit or correctness selection.',
        prompt_variant=args.variant,
        limitations='Single-question prompt/budget diagnosis, not ten fresh smoke or formal data; strict final-only parsing remains required.')
    atomic_json(OUTPUT,result)
    backup_probe(RUN,f'Prepare strict final-line {args.variant} prompt diagnosis at 49152 tokens')
    model,tokenizer=load_model(cfg)
    prompt=prompt_tokens(cfg,tokenizer,row['problem'])
    seed=generation_seed(cfg['seed'],row['unique_id'])
    session=Session(model,prompt,cfg,seed)
    generated=[];start=time.monotonic()
    for _ in range(min(cfg['max_new_tokens'],cfg['max_total_tokens']-len(prompt))):
        token=session.sample();generated.append(token)
        if len(generated)%2048==0 or token in model.generation_config.eos_token_id:
            result.update(status='GENERATING',prompt_ids=prompt,generated_ids=generated.copy(),
                          generation_seed=seed,raw_text=decode(tokenizer,generated),seconds=time.monotonic()-start)
            atomic_json(OUTPUT,result)
            print('TOKENS',len(generated),flush=True)
        if token in model.generation_config.eos_token_id:
            break
    eos=generated[-1] in model.generation_config.eos_token_id
    raw=decode(tokenizer,generated)
    result.update(status='GENERATED',prompt_ids=prompt,generated_ids=generated,generation_seed=seed,
                  raw_text=raw,seconds=time.monotonic()-start,completion='EOS' if eos else 'LENGTH_TRUNCATED')
    atomic_json(OUTPUT,result)
    _,bounds=boundaries(tokenizer,generated,PARAGRAPHS)
    endpoint=terminal(tokenizer,generated,eos,mode=PARAGRAPHS)
    result.update(status='DIAGNOSTIC_COMPLETE',boundaries=bounds,finished=now(),
        terminal={k:(endpoint or {}).get(k) for k in ('reason','answer','current_step')},
        thinking_error=thinking_error(raw,terminal=True,mode=PARAGRAPHS),
        strict_final_format_valid=(endpoint or {}).get('reason')=='FINAL')
    atomic_json(OUTPUT,result)
    backup_probe(RUN,f'Save strict final-line {args.variant} prompt diagnosis without changing official protocol')


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        result=read_json(OUTPUT,{})
        atomic_json(OUTPUT,dict(result,status='FAILED',error=safe_error(exc),finished=now()))
        raise
