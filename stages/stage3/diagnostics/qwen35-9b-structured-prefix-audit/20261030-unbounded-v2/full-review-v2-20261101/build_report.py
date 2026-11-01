"""Join direct-review records to frozen sources; never derive mathematical labels."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE.parent
ROOT=HERE.parents[5]
SOURCE=ROOT/'stages/stage3/runs/verification-benefit-c905eb4ccd85/verifier_audit_review_inputs.json'
def read(p): return json.loads(p.read_text())
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def md(s): return str(s).replace('|','\\|').replace('\n',' ')
manifest=read(RUN/'manifest.json')
old_path=Path(manifest['codex_csv'])
assert sha(SOURCE)==manifest['source_inputs_sha256']
assert sha(old_path)==manifest['codex_csv_sha256']
source=read(SOURCE)
old=list(csv.DictReader(old_path.open(encoding='utf-8-sig')))
verifier=read(RUN/'comparison.json')
items=[read(HERE/'items'/f'{i:03}.json') for i in range(1,58)]
assert len(source)==len(old)==len(verifier)==len(items)==57
rows=[];groups=defaultdict(list);disagreements=[];changes=[]
for i,(x,a,b,n) in enumerate(zip(source,old,verifier,items),1):
    assert int(a['source_index'])==b['source_index']==n['index']==i
    assert x['review_key']==a['review_key']==b['review_key']==n['review_key']
    assert x['unique_id']==a['unique_id']==b['unique_id']==n['unique_id']
    assert n['reviewed_chars']==len(x['quoted_prefix'])
    assert n['label'] in ('PASS','FAIL')
    assert n['reason'].strip()
    assert (n['label']=='PASS' and n['first_error_quote']=='') or (n['label']=='FAIL' and n['first_error_quote'] in x['quoted_prefix'])
    q=int(a['question_number']);groups[q].append(i)
    quote=n['first_error_quote']
    locations=[u['checkpoint_number'] for u in x['checkpoint_units'] if quote and quote in u['quoted_text']]
    assert not quote or locations
    attempt=read(RUN/'first'/f'{i:03}'/'attempt_01.json')
    r=dict(source_index=i,question_number=q,unique_id=x['unique_id'],review_key=x['review_key'],check_type=x['check_type'],checkpoint_count=len(x['checkpoint_units']),reviewed_chars=len(x['quoted_prefix']),prefix_sha256=hashlib.sha256(x['quoted_prefix'].encode()).hexdigest(),label=n['label'],severity=n['severity'],first_error_quote=quote,quoted_checkpoint_numbers=locations,reason=n['reason'],superseded_local_errors=n.get('superseded_local_errors',''),old_codex_label=a['label'],old_9b=b['old_9b'],structured_9b=b['final_9b'],structured_9b_reason=attempt['judgment']['diagnosis'],structured_9b_hint=attempt['judgment']['hint'],label_changed=n['label']!=a['label'],agrees_with_structured_9b=n['label']==b['final_9b'],is_human_gold=False,reading_scope='full_text_shared_prefix')
    rows.append(r)
    if r['label_changed']:changes.append({k:r[k] for k in ['source_index','old_codex_label','label','reason']})
    if not r['agrees_with_structured_9b']:disagreements.append(r)
coverage=[]
for q,indices in groups.items():
    full=source[indices[-1]-1]['quoted_prefix']
    assert all(full.startswith(source[i-1]['quoted_prefix']) for i in indices)
    coverage.append(dict(question_number=q,unique_id=source[indices[0]-1]['unique_id'],indices=indices,read_chars_once=len(full),cutoffs={str(i):len(source[i-1]['quoted_prefix']) for i in indices},coverage='all characters of longest prefix read; each cutoff judged separately'))
assert sorted(i for g in coverage for i in g['indices'])==list(range(1,58))
matrix={label:{v:sum(r['label']==label and r['structured_9b']==v for r in rows) for v in ('PASS','FAIL','UNFINISHED')} for label in ('PASS','FAIL')}
assert sum(sum(x.values()) for x in matrix.values())==57
summary=dict(status='complete',reviewer='Codex direct second full-text review',new_model_or_api_calls=0,is_human_gold=False,blind_review=False,source_sha256=sha(SOURCE),old_codex_csv_sha256=sha(old_path),count=57,questions=len(groups),prefix_characters=sum(len(x['quoted_prefix']) for x in source),shared_text_read_characters=sum(g['read_chars_once'] for g in coverage),labels=dict(Counter(r['label'] for r in rows)),fail_severity=dict(Counter(r['severity'] for r in rows if r['label']=='FAIL')),fail_indices=[r['source_index'] for r in rows if r['label']=='FAIL'],changes_from_frozen_codex=changes,structured_9b_cross_table=matrix,agreement_count=sum(r['agrees_with_structured_9b'] for r in rows),disagreement_indices=[r['source_index'] for r in disagreements],focus_indices=[10,20,21,23,25,42,48,51,52,54],major_focus_indices=[23,25,52,54])
save(HERE/'labels.json',rows);save(HERE/'coverage.json',coverage);save(HERE/'summary.json',summary);save(HERE/'disagreements.json',disagreements)
with (HERE/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
progress=read(HERE/'progress.json');assert progress['completed']==list(range(1,58));progress.update(status='complete',source_sha256=sha(SOURCE),old_codex_csv_sha256=sha(old_path));save(HERE/'progress.json',progress)
lines=['# 全部57条前缀：第二轮全文复核','', '本次已完成57/57条的全文复核。直接阅读原题和完整 quoted_prefix，包括思考过程及已有最终答复；同一道题完全相同的重叠文本读一次，在各自截断位置分别判断。没有新增模型/API调用。', '',f"结果：**PASS {summary['labels']['PASS']}条，FAIL {summary['labels']['FAIL']}条**。FAIL中局部错误{summary['fail_severity'].get('局部',0)}条、主线错误{summary['fail_severity'].get('主线',0)}条、逻辑错误{summary['fail_severity'].get('逻辑',0)}条。局部指没有据此否定主要结论；逻辑指已明确采用无效限制或推导，可能仍在探索。",'',f"覆盖20道原题，57条前缀累计{summary['prefix_characters']:,}字符；按完全相同的嵌套文本去重，全文阅读{summary['shared_text_read_characters']:,}字符。编号、review_key、unique_id、截断长度及原文对应见[覆盖记录](coverage.json)和[逐条表](labels.csv)。原输入及旧Codex标签的SHA256均与原运行manifest一致。",'', '## 判定口径与限制','', '只因截至当前仍被采用的明确数学错误判FAIL。未完成、探索失败、没有最终答案均不单独判错。已纠正、明确放弃，或已被后文独立成立的论证替代的支路不延续判错；仍用于当前核对的新增局部错误单独标记。PASS表示在此口径下未确认保留错误，不表示逐句原文没有写错。', '', '第43、46、49条的旧辅助检查确有错误，但后文已独立替代或明确选择其他证明，因此仍为PASS；这些问题另保存在superseded_local_errors字段。第15条新增迭代表仍被用于确认收敛，表内f(4)错误未修正；正确的固定点7不证明该表每一行正确，也不替代那项收敛检查。', '', '本次已经知道旧9B及旧Codex结论，属于非盲复核；这些是Codex暂定对照判断，未经独立专家逐条确认，不能称为绝对真值或据此直接报告模型准确率。错误引文定位到checkpoint；它表示本次确认的判错位置，多错误、反复改写的长前缀中，最早仍有效错误的位置仍可能需进一步裁定。', '', '## 相对旧Codex表的修订','', '|编号|旧标签|本次标签|复核依据|','|---|---|---|---|']
for c in changes:lines.append(f"|{c['source_index']}|{c['old_codex_label']}|{c['label']}|{md(c['reason'])}|")
lines+=['', '其余54条标签保持一致。旧labels.csv、旧报告及此前6条定向复核记录均未覆盖。本次57条都有独立条目文件，不再只有6条有复核记录。', '', '## 与结构化9B结果比较','', '|本次Codex暂定对照|9B PASS|9B FAIL|9B未完成|','|---|---:|---:|---:|']
for label in ('PASS','FAIL'):
    m=matrix[label];lines.append(f"|{label}|{m['PASS']}|{m['FAIL']}|{m['UNFINISHED']}|")
lines += ['',f"标签一致{summary['agreement_count']}/57（{summary['agreement_count']/57:.2%}）；不一致{len(disagreements)}条。双方完整理由见[不一致清单](disagreements.json)。本次没有重跑9B，52 PASS、5 FAIL及原输出token/耗时记录保持原样。本次一致性变化来自对照标签修订，不是9B输出变化。",'', '9B在第50条给FAIL，但其理由主要是“没完成构造”；本次依据是3阶棋盘奇偶分配错误，两者理由不同。第53条同样不能因为没写出最大值就判错，本次依据是(2,1)度数误判。第54条9B提出的6×6剩余棋盘全为偶度也不成立：满6×6的非角边界车度数3。标签相同不代表解释或返工提示可靠。', '', '重点10条：','', '|编号|旧9B|结构化9B|本次Codex|类别|','|---|---|---|---|---|']
for i in summary['focus_indices']:
    r=rows[i-1];lines.append(f"|{'**'+str(i)+'**' if i in summary['major_focus_indices'] else i}|{r['old_9b']}|{r['structured_9b']}|{r['label']}|{r['severity']}|")
lines += ['', '主要推导/结论重点23、25、52仍是9B PASS / 本次FAIL；54为双方FAIL。这里列为待复核分歧，不自行将Codex当作已核实真值。', '', '## 数学复算与后续','', '[mathematical_checks.py](mathematical_checks.py)用标准库复算新增算术错误及若干关键反例，已运行通过；结果见[mathematical_checks.json](mathematical_checks.json)。这些检查只支持具体数学事实，不自动生成标签，也不证明全文审查绝无遗漏。几何及棋盘不可达性说明见[mathematical_notes.md](mathematical_notes.md)。', '', '下一步优先独立复核主线分歧23、24、25、33、52，以及对判定口径敏感的15、43、46、49；确认后再冻结新版本对照标签。不要用本次结果覆盖旧表或把局部错误数量解释为最终答案错误数量。', '', '## 逐条结果','', '|编号|题号|类型|旧Codex|本次Codex|9B结构化|错误类别|理由|','|---|---|---|---|---|---|---|---|']
for r in rows:lines.append(f"|{r['source_index']}|{r['question_number']}|{r['check_type']}|{r['old_codex_label']}|{r['label']}|{r['structured_9b']}|{r['severity'] or '—'}|{md(r['reason'])}|")
(HERE/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['count','labels','fail_severity','agreement_count','structured_9b_cross_table','changes_from_frozen_codex']},ensure_ascii=False,indent=2))
