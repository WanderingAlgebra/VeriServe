"""Read-only joins against the frozen second review; incomplete runs retain all57 rows."""
import csv
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[4]
BASE=ROOT/'stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-unbounded-v2'
REF=BASE/'full-review-v2-20261101'
MAJOR=[18,19,23,24,25,33,52]


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def costs(records):
    def describe(key):
        v=[r[key] for r in records if r.get(key) is not None]
        return dict(n=len(v),total=sum(v),mean=statistics.mean(v),median=statistics.median(v),
                    maximum=max(v)) if v else dict(n=0)
    return dict(records=len(records),request_seconds=describe('request_seconds'),
                output_tokens=describe('output_tokens'),input_tokens=describe('input_tokens'),
                peak_device_bytes=max((r.get('device_peak_bytes',0) for r in records),default=0))


def main():
    labels=read(REF/'labels.json');inventory={r['index']:r for r in read(RUN/'input_inventory.json')}
    rows=[];new_records=[];old_records=[];matrix=Counter();by_question=defaultdict(list)
    for ref in labels:
        i=ref['source_index'];p=RUN/'comparison'/f'{i:03}.json'
        r=read(p) if p.exists() else dict(status='in_progress' if p.with_suffix('.intent.json').exists() else 'not_run',judgment=None)
        old=read(BASE/'first'/f'{i:03}'/'attempt_01.json');old_records.append(old)
        if p.exists():new_records.append(r)
        verdict=(r.get('judgment') or {}).get('verdict','UNAVAILABLE')
        matrix[(ref['label'],verdict)]+=1
        by_question[ref['unique_id']].append(int(verdict==ref['label']))
        j=r.get('judgment') or {}
        rows.append(dict(index=i,question_number=ref['question_number'],unique_id=ref['unique_id'],
            review_key=ref['review_key'],check_type=ref['check_type'],severity=ref['severity'],
            provisional_codex=ref['label'],codex_reason=ref['reason'],
            codex_error_checkpoints=ref['quoted_checkpoint_numbers'],
            nine_b=old['judgment']['verdict'],nine_b_error_step=old['judgment']['first_error_step'],
            nine_b_diagnosis=old['judgment']['diagnosis'],nine_b_hint=old['judgment']['hint'],
            twentyseven_b=verdict,status=r['status'],first_error_step=j.get('first_error_step'),
            diagnosis=j.get('diagnosis',''),hint=j.get('hint',''),
            agrees_with_reference=verdict==ref['label'],original_19=i in [x['source_index'] for x in read(REF/'disagreements.json')],
            input_tokens=inventory[i]['input_tokens'],output_tokens=r.get('output_tokens'),
            request_seconds=r.get('request_seconds'),device_peak_bytes=r.get('device_peak_bytes'),
            finish_reason=r.get('finish_reason'),error=r.get('error')))
    save(RUN/'comparison.json',rows)
    with (RUN/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    correct=sum(x['agrees_with_reference'] for x in rows)
    attempted=sum(x['status'] not in ['not_run','in_progress'] for x in rows)
    def indices(pred):return [x['index'] for x in rows if pred(x)]
    capacity=[read(p) for p in (RUN/'capacity').glob('*.json') if p.stem.isdigit()]
    old_sessions=[read(p) for p in (BASE/'sessions').glob('*/completed.json') if read(p)['phase']=='first']
    old_cost=costs(old_records)
    old_cost['peak_device_bytes']=max((s['device_peak_bytes'] for s in old_sessions),default=None)
    old_cost['model_load_seconds']=sum(s['model_init_seconds'] for s in old_sessions)
    summary=dict(planned_prefixes=57,attempted=attempted,original_questions=len(by_question),
        completed=sum(x['status']=='completed' for x in rows),
        status_counts=dict(Counter(x['status'] for x in rows)),verdict_counts=dict(Counter(x['twentyseven_b'] for x in rows)),
        confusion_matrix={gold:{v:matrix[(gold,v)] for v in ['PASS','FAIL','UNAVAILABLE']} for gold in ['PASS','FAIL']},
        agreement=dict(numerator=correct,denominator=57,
                       fraction=correct/57 if all(x['status']=='completed' for x in rows) else None),
        successful_reference_matches_over_all_planned=dict(numerator=correct,denominator=57,
                       fraction=correct/57 if attempted==57 else None,
                       meaning='Operational successful matching fraction; unavailable is not a successful match, not converted to PASS or FAIL.'),
        question_equal_weight_agreement=(sum(statistics.mean(v) for v in by_question.values())/len(by_question)
                                         if all(x['status']=='completed' for x in rows) else None),
        question_equal_weight_successful_match_fraction=(sum(statistics.mean(v) for v in by_question.values())/len(by_question)
                                         if attempted==57 else None),
        question_equal_weight_successful_match_policy='All20 questions equally weighted, prefixes within question equally weighted; unavailable is not a successful match.',
        old_19_recovered=indices(lambda x:x['original_19'] and x['twentyseven_b']=='FAIL'),
        old_19_still_pass=indices(lambda x:x['original_19'] and x['twentyseven_b']=='PASS'),
        old_19_unavailable=indices(lambda x:x['original_19'] and x['twentyseven_b']=='UNAVAILABLE'),
        new_sample_false_alarms=indices(lambda x:x['provisional_codex']=='PASS' and x['twentyseven_b']=='FAIL'),
        old_5_regressed=indices(lambda x:x['nine_b']=='FAIL' and x['twentyseven_b']=='PASS'),
        all_new_fail=indices(lambda x:x['twentyseven_b']=='FAIL'),
        major_7=[x for x in rows if x['index'] in MAJOR],
        nine_b_cost=old_cost,twentyseven_b_cost=costs(new_records),capacity_cost=costs(capacity),
        model_loads=[dict(session=p.parent.name,**read(p)) for p in (RUN/'sessions').glob('*/engine.json')],
        labels_are_provisional=True,independent_expert_gold=False,rework_executed=False,
        warnings=['Descriptive paired comparison on20 questions, with nested prefixes; not57 independent questions.',
          'No capacity/error/truncation exclusions from the57-row denominator; unavailable predictions shown separately.',
          'Quantization, prefill chunking, eager execution and context differ; latency is measured deployment cost, not a pure model-size effect.',
          'FAIL label agreement does not establish correct location, reason, or feedback. See qualitative review.'])
    save(RUN/'summary.json',summary)
    prior=RUN.parent/'20261101-awq-v1'
    if (prior/'manifest.json').exists():
        prior_records=[read(p) for p in (prior/'comparison').glob('*.json') if p.stem.isdigit()]
        summary['preliminary_unbounded_run']=dict(
            path=str(prior),cost=costs(prior_records),
            statuses=dict(Counter(r['status'] for r in prior_records)),
            unknown_output_tokens=[r['index'] for r in prior_records if r.get('output_tokens') is None],
            interrupted_timing_is_approximate=True,
            old_partial_predictions_not_merged_into_this_run=True)
        summary['real_capacity_cost']=costs([read(p) for p in (prior/'capacity').glob('*.json') if p.stem.isdigit()])
        summary['capacity_cost']=summary['real_capacity_cost']
        summary['capacity_cost_accounting']='Reused3 prior capacity requests, counted once in overall experiment cost; not new calls in this57-run.'
        summary['prior_model_loads']=[dict(session=p.parent.name,**read(p)) for p in (prior/'sessions').glob('*/engine.json')]
        summary['identical_tokens_on_overlapping_returned_requests']=[r['index'] for r in prior_records
            if r.get('generated_ids') is not None and (RUN/'comparison'/f'{r["index"]:03}.json').exists()
            and read(RUN/'comparison'/f'{r["index"]:03}.json').get('generated_ids')==r['generated_ids']]
        summary['nine_b_reference_matches']=sum(r['nine_b']==r['provisional_codex'] for r in rows)
        summary['nine_b_confusion_matrix']={g:{v:sum(r['provisional_codex']==g and r['nine_b']==v for r in rows)
                                               for v in ['PASS','FAIL']} for g in ['PASS','FAIL']}
        summary['nine_b_question_equal_weight_match_fraction']=statistics.mean(
            statistics.mean(r['nine_b']==r['provisional_codex'] for r in rows if r['unique_id']==qid)
            for qid in by_question)
        save(RUN/'summary.json',summary)
    save(RUN/'fail_outputs.json',[x for x in rows if x['twentyseven_b']=='FAIL'])
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save(RUN/'analysis_provenance.json',dict(analysis_code_sha256=sha(Path(__file__)),
        new_manifest_sha256=sha(RUN/'manifest.json'),reference_labels_sha256=sha(REF/'labels.json'),
        baseline_manifest_sha256=sha(BASE/'manifest.json'),
        new_records_sha256={p.name:sha(p) for p in (RUN/'comparison').glob('*.json') if p.stem.isdigit()},
        no_labels_modified=True,reference_is_provisional=True))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['major_7','model_loads','warnings']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
