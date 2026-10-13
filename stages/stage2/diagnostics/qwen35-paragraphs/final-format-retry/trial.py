"""Full-budget format calibration on the first four frozen smoke questions."""
import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from veriserve_research import ROOT
from veriserve_research.artifacts.backup import backup_probe
from veriserve_research.artifacts.io import atomic_json, digest, git, now, read_json, safe_error
from veriserve_research.artifacts.provenance import source_hashes
from veriserve_research.inference import Session, generation_seed, load_model, prompt_tokens
from veriserve_research.intervention.protocol import terminal
from veriserve_research.probe.materials import record_path
from veriserve_research.trajectory import PARAGRAPHS, boundaries, decode, thinking_error

FOLDER = Path(__file__).resolve().parent
RUN = ROOT/'stages/stage2/runs/v2-20261013-da5c802342ee'
STATE = FOLDER/'trial_status.json'


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    manifest = read_json(RUN/'manifest.json')
    frozen = source_hashes('probe')
    assert frozen == manifest['source_hashes']
    cfg = copy.deepcopy(manifest['config'])
    cfg['system_prompt'] = (
        'Solve each problem using your native thinking block. Keep all derivations and '
        'explanations inside that block. Separate major reasoning stages with blank lines. '
        'Once you close the thinking block, output exactly one line of the form: '
        'Final answer: \\boxed{answer}. Do not repeat the solution after the thinking block.')
    cfg['format_examples'] += [
        {'role':'user','content':'Determine all real solutions of x^4 - 13x^2 + 36 = 0.'},
        {'role':'assistant','content':r'Final answer: \boxed{\{-3,-2,2,3\}}'},
        {'role':'user','content':'A triangle has side lengths 17, 25, and 28. Find its inradius.'},
        {'role':'assistant','content':r'Final answer: \boxed{6}'},
    ]
    rows = manifest['splits']['smoke'][:4]
    metadata = dict(validation_only=True, scientific_results=False, new_smoke_passed=False,
        created=now(), pid=os.getpid(), wait_pid=args.wait_pid,
        source_manifest_sha256=digest(RUN/'manifest.json'), execution_source_hashes=frozen,
        driver_sha256=digest(__file__), ordered_ids=[r['unique_id'] for r in rows],
        selection='Original first four smoke questions in manifest order; format-only calibration.',
        limitations='No correctness tuning or formal labels; does not replace ten fresh smoke or resume checks.')
    atomic_json(FOLDER/'trial_config.json', cfg)
    atomic_json(STATE, dict(metadata, status='WAITING_FOR_PRIOR_SMOKE'))
    # Start time distinguishes PID reuse; zombies have already released GPU resources.
    prior = process_state(args.wait_pid)
    if prior:
        while ((state := process_state(args.wait_pid)) and state[19] == prior[19]
               and state[0] not in ('Z','X')):
            time.sleep(30)
    assert source_hashes('probe') == frozen
    assert all(read_json(record_path(RUN,'smoke',r['unique_id']),{}).get('stage')
               in ('FEATURES','NO_FEATURE_POSITIONS') for r in manifest['splits']['smoke'])
    backup_probe(RUN, 'Prepare separate final-format calibration after complete smoke')
    metadata['execution_commit'] = git('rev-parse','HEAD')
    atomic_json(STATE, dict(metadata, status='LOADING_MODEL', completed=0))
    model, tokenizer = load_model(cfg)
    for index, row in enumerate(rows):
        path = FOLDER/f'{index+1}-{row["unique_id"][:12]}.json'
        if path.exists():
            raise FileExistsError(f'Preserve existing diagnostic: {path}')
        atomic_json(STATE, dict(metadata,status='GENERATING',completed=index,current_id=row['unique_id']))
        prompt = prompt_tokens(cfg,tokenizer,row['problem'])
        seed = generation_seed(cfg['seed'],row['unique_id'])
        session = Session(model,prompt,cfg,seed)
        generated = []
        started = time.monotonic()
        for _ in range(min(cfg['max_new_tokens'],cfg['max_total_tokens']-len(prompt))):
            token = session.sample()
            generated.append(token)
            if token in model.generation_config.eos_token_id:
                break
        eos = generated[-1] in model.generation_config.eos_token_id
        raw = decode(tokenizer,generated)
        record = dict(metadata,unique_id=row['unique_id'],config_sha256=digest(FOLDER/'trial_config.json'),
            generation_seed=seed,prompt_ids=prompt,generated_ids=generated,raw_text=raw,
            seconds=time.monotonic()-started,stage='GENERATED',completion='EOS' if eos else 'LENGTH_TRUNCATED')
        atomic_json(path,record)
        _, bounds = boundaries(tokenizer,generated,PARAGRAPHS)
        endpoint = terminal(tokenizer,generated,eos,mode=PARAGRAPHS)
        record.update(stage='PARSED',boundaries=bounds,terminal=endpoint,
            thinking_error=thinking_error(raw,terminal=True,mode=PARAGRAPHS),
            final_format_valid=endpoint.get('reason')=='FINAL',finished=now())
        atomic_json(path,record)
        print(row['unique_id'][:12],len(generated),record['completion'],
              'final_format_valid',record['final_format_valid'],flush=True)
        del session
        gc.collect()
        backup_probe(RUN,f'Final-format calibration question {index+1}/4')
    atomic_json(STATE,dict(metadata,status='DIAGNOSTIC_COMPLETE',completed=len(rows),finished=now()))
    backup_probe(RUN,'Save complete four-question final-format calibration')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        state = read_json(STATE,{})
        atomic_json(STATE,dict(state,status='FAILED',error=safe_error(exc),finished=now()))
        raise
