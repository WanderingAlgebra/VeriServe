"""Read-only joins against the frozen second review; incomplete runs retain all57 rows."""
import csv
from collections import Counter, defaultdict
import json
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
        i=ref['source_index'];p=RUN/'first'/f'{i:03}.json'
        r=read(p) if p.exists() else dict(status='not_run',judgment=None)
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
    def indices(pred):return [x['index'] for x in rows if pred(x)]
    capacity=[read(p) for p in (RUN/'capacity').glob('*.json') if p.stem.isdigit()]
    old_sessions=[read(p) for p in (BASE/'sessions').glob('*/completed.json') if read(p)['phase']=='first']
    old_cost=costs(old_records)
    old_cost['peak_device_bytes']=max((s['device_peak_bytes'] for s in old_sessions),default=None)
    old_cost['model_load_seconds']=sum(s['model_init_seconds'] for s in old_sessions)
    summary=dict(planned_prefixes=57,original_questions=len(by_question),
        completed=sum(x['status']=='completed' for x in rows),
        status_counts=dict(Counter(x['status'] for x in rows)),verdict_counts=dict(Counter(x['twentyseven_b'] for x in rows)),
        confusion_matrix={gold:{v:matrix[(gold,v)] for v in ['PASS','FAIL','UNAVAILABLE']} for gold in ['PASS','FAIL']},
        agreement=dict(numerator=correct,denominator=57,
                       fraction=correct/57 if all(x['status']=='completed' for x in rows) else None),
        question_equal_weight_agreement=(sum(statistics.mean(v) for v in by_question.values())/len(by_question)
                                         if all(x['status']=='completed' for x in rows) else None),
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
    save(RUN/'fail_outputs.json',[x for x in rows if x['twentyseven_b']=='FAIL'])
    print(json.dumps({k:v for k,v in summary.items() if k not in ['major_7','model_loads','warnings']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
