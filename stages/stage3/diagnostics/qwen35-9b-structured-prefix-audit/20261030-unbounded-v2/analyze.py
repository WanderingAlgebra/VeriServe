"""Rebuild diagnostic tables from preserved first attempts and separate retries."""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUN=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
manifest=read(RUN/'manifest.json')


def records(phase,index):
    return [read(p) for p in sorted((RUN/phase/f'{index:03}').glob('attempt_*.json'))]


def label(r):
    return (r.get('judgment') or {}).get('verdict','未完成') if r else '未完成'


def describe(values):
    values=[x for x in values if x is not None]
    return dict(count=len(values),total=sum(values),mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,max=max(values) if values else None)


def matrix(items,reference,new):
    return {a:{b:sum(reference(x)==a and new(x)==b for x in items) for b in ['PASS','FAIL','未完成']}
            for a in ['PASS','FAIL']}


def save(name,value):
    (RUN/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


rows=[];firsts=[];retries=[]
for source in manifest['rows']:
    i=source['source_index'];fs=records('first',i);rs=records('retry',i)
    first=fs[0] if fs else None
    final=next((x for x in fs+rs if x['status']=='completed'), (rs or fs or [None])[-1])
    if first:firsts.append(first)
    retries.extend(rs)
    old=source['original_result'];judgment=(final or {}).get('judgment') or {}
    row=dict(source_index=i,unique_id=source['unique_id'],review_key=source['review_key'],
             check_type=source['check_type'],codex_label=source['codex_label'],
             old_9b=old['effective_verdict'],old_parser_error=old['parser_error'],
             first_9b=label(first),first_status=(first or {}).get('status','not_run'),
             configured_max_tokens=None,
             context_output_capacity=manifest['original_config']['max_verifier_model_len']-
                 source['original_result']['input_tokens']-source['prompt_token_difference'],
             final_9b=label(final),final_status=(final or {}).get('status','not_run'),
             retry_attempts=len(rs),is_focus=i in manifest['focus_indices'],
             is_major=i in manifest['major_indices'],
             first_output_tokens=(first or {}).get('output_tokens'),
             first_input_tokens=(first or {}).get('input_tokens'),
             first_error=(first or {}).get('error'),
             first_request_seconds=(first or {}).get('request_seconds'),
             first_finish_reason=(first or {}).get('finish_reason'),
             first_raw_output=(first or {}).get('raw_output'),
             final_output_tokens=(final or {}).get('output_tokens'),
             final_error=(final or {}).get('error'),
             final_request_seconds=(final or {}).get('request_seconds'),
             final_finish_reason=(final or {}).get('finish_reason'),
             final_raw_output=(final or {}).get('raw_output'),
             new_first_error_step=judgment.get('first_error_step'),
             new_diagnosis=judgment.get('diagnosis',''),new_hint=judgment.get('hint',''),
             codex_reason=source['codex_reason'],codex_quote=source['codex_quote'],
             codex_caveat=source['codex_caveat'],old_raw_output=old.get('raw_text',''),
             old_output_tokens=old['output_tokens'],old_worker_seconds=old['seconds'],
             old_generation_seconds=old.get('engine',{}).get('prefill_and_generation_seconds'),
             prompt_token_difference=source['prompt_token_difference'])
    rows.append(row)

smoke=[read(p) for p in (RUN/'smoke').glob('*/attempt_*.json')]
old_counts=Counter(r['old_9b'] for r in rows)
old_raw_states=Counter()
for source in manifest['rows']:
    old=source['original_result']
    if not old['parser_error']:continue
    raw=old.get('raw_text','')
    for eos in ['<|im_end|>','<|endoftext|>']:
        if raw.endswith(eos):raw=raw[:-len(eos)]
    try:
        value=json.loads(raw)
        old_raw_states[old['completion']+' / valid_json / '+str(value.get('verdict'))]+=1
    except ValueError:
        old_raw_states[old['completion']+' / invalid_json']+=1
failures=Counter(r['first_status'] for r in rows if r['first_status']!='completed')
changes={a:{b:sum(r['old_9b']==a and r['first_9b']==b for r in rows) for b in ['PASS','FAIL','未完成']}
         for a in ['PASS','FAIL','UNCERTAIN']}
disagreements=[r for r in rows if r['final_9b'] in ['PASS','FAIL'] and r['final_9b']!=r['codex_label']]
groups=defaultdict(list)
for r in rows:
    if r['final_9b']!='未完成':groups[r['unique_id']].append(r['final_9b']==r['codex_label'])
old_length_rejects=[]
for r in firsts:
    value=r.get('judgment') or {}
    if any(len(value.get(k,''))>300 for k in ['diagnosis','hint']):old_length_rejects.append(r['source_index'])
summary=dict(total=57,output_budget_policy='max_tokens=None; no fixed output cap',
             model_context_tokens=manifest['original_config']['max_verifier_model_len'],
             first_attempted=len(firsts),first_status_counts=dict(Counter(r['first_status'] for r in rows)),
             first_valid=sum(r['first_status']=='completed' for r in rows),first_failures=dict(failures),
             first_verdicts=dict(Counter(r['first_9b'] for r in rows)),
             first_codex_matrix=matrix(rows,lambda r:r['codex_label'],lambda r:r['first_9b']),
             final_codex_matrix=matrix(rows,lambda r:r['codex_label'],lambda r:r['final_9b']),
             original_verdicts=dict(old_counts),old_to_first_matrix=changes,
             retries=dict(attempts=len(retries),status_counts=dict(Counter(r['status'] for r in retries)),
                          indices=sorted(set(r['source_index'] for r in retries))),
             final_valid=sum(r['final_status']=='completed' for r in rows),
             final_verdicts=dict(Counter(r['final_9b'] for r in rows)),
             disagreement_indices=[r['source_index'] for r in disagreements],
             focus=[{k:r[k] for k in ['source_index','is_major','old_9b','first_9b','final_9b','codex_label']}
                    for r in rows if r['is_focus']],
             completed_only_agreement=(sum(r['final_9b']==r['codex_label'] for r in rows)/
                                       sum(r['final_status']=='completed' for r in rows)
                                       if any(r['final_status']=='completed' for r in rows) else None),
             question_equal_weight_agreement=statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None,
             evaluated_question_count=len(groups),
             costs=dict(first_input_tokens=describe([r.get('input_tokens') for r in firsts]),
                        first_output_tokens=describe([r.get('output_tokens') for r in firsts]),
                        first_request_seconds=describe([r.get('request_seconds') for r in firsts]),
                        retry_output_tokens=describe([r.get('output_tokens') for r in retries]),
                        retry_request_seconds=describe([r.get('request_seconds') for r in retries]),
                        old_output_tokens=describe([r['old_output_tokens'] for r in rows]),
                        old_worker_seconds=describe([r['old_worker_seconds'] for r in rows]),
                        old_generation_seconds=describe([r['old_generation_seconds'] for r in rows])),
             new_completed_exceeding_old300_character_cap=old_length_rejects,
             first_finish_reasons=dict(Counter(str(r.get('finish_reason')) for r in firsts)),
             original_failures=dict(Counter(r['old_parser_error'] for r in rows if r['old_parser_error'])),
             original_rejected_raw_output_audit=dict(old_raw_states),
             first_engine_initializations=[read(p) for p in (RUN/'sessions').glob('*/engine.json')
                 if read(p.parent/'start.json')['phase']=='first'],
             smoke_costs=dict(attempts=len(smoke),output_tokens=describe([r.get('output_tokens') for r in smoke]),
                              request_seconds=describe([r.get('request_seconds') for r in smoke])),
             documentation=manifest['docs'])
save('summary.json',summary);save('comparison.json',rows);save('disagreements.json',disagreements)
save('focus_10.json',[r for r in rows if r['is_focus']])
with (RUN/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table_matrix(m):
    return ['| Codex暂时对照 | 9B PASS | 9B FAIL | 未完成 |','|---|---:|---:|---:|']+[
        f"| {a} | {m[a]['PASS']} | {m[a]['FAIL']} | {m[a]['未完成']} |" for a in ['PASS','FAIL']]


lines=['# Qwen3.5-9B：57前缀结构化输出重跑','',
       f"首轮已运行{len(firsts)}/57条，可用PASS/FAIL共{summary['first_valid']}条；失败/未运行分类：{dict(failures)}。",
       f"首轮判定分布：{summary['first_verdicts']}。重试共{len(retries)}次，状态为{summary['retries']['status_counts']}；合并后可用{summary['final_valid']}/57条。",'',
       '原首检为39 PASS、18 UNCERTAIN、0 FAIL；18条均伴随格式/反馈长度或截断问题。新结果的格式成功率与数学判断一致性分开报告。', '',
       '## 首轮与Codex暂时对照','']+table_matrix(summary['first_codex_matrix'])
lines+=['','## 含重试的最终对照','']+table_matrix(summary['final_codex_matrix'])
lines+=['',f"仅在有效新判断上，与Codex标签的一致率为{summary['completed_only_agreement']}；按题等权为{summary['question_equal_weight_agreement']}（{len(groups)}题）。这不是相对绝对真值的准确率。57个前缀来自20题，存在前缀包含关系。",'',
        '## 新旧9B首轮判断','',
        '| 原9B首检 | 新PASS | 新FAIL | 未完成 |','|---|---:|---:|---:|']
for old in ['PASS','FAIL','UNCERTAIN']:
    a=changes[old];lines.append(f"| {old} | {a['PASS']} | {a['FAIL']} | {a['未完成']} |")
lines += ['',f"原39条有效PASS中，{changes['PASS']['FAIL']}条在新首轮变为FAIL，{changes['PASS']['PASS']}条仍为PASS，{changes['PASS']['未完成']}条未完成。原UNCERTAIN到新二分类是从旧不可用结果获得新判断，不称为原有效数学标签的改判。",
        f"原18条不可用结果，新首轮不可用为{57-summary['first_valid']}条（未运行也计入这一描述，实际状态分类见开头）。程序可用性与数学标签一致性分开评估；其差值同时受结构化约束、取消输出上限及输出规则变化影响。",'',
        '## 原先10条分歧','', '| 编号 | 主要推导/结论 | 原9B | 新首轮 | 重试后 | Codex |','|---|---|---|---|---|---|']
for r in rows:
    if r['is_focus']:lines.append(f"| {r['source_index']} | {'是' if r['is_major'] else '局部问题'} | {r['old_9b']} | {r['first_9b']} | {r['final_9b']} | {r['codex_label']} |")
def cost_text(key,unit):
    c=summary['costs'][key]
    return '未发生' if not c['count'] else f"总计{c['total']:.2f}{unit}，平均{c['mean']:.2f}{unit}，最大{c['max']:.2f}{unit}"


lines+=['','## 成本和异常','',f"首轮输出：{cost_text('first_output_tokens',' token')}。",f"首轮请求耗时：{cost_text('first_request_seconds','秒')}。",f"首轮结束原因：{summary['first_finish_reasons']}。原首检失败分类：{summary['original_failures']}。",f"重试输出：{cost_text('retry_output_tokens',' token')}；重试耗时：{cost_text('retry_request_seconds','秒')}。",f"原首检输出：{cost_text('old_output_tokens',' token')}；原首检生成耗时：{cost_text('old_generation_seconds','秒')}。",'',
        '新运行按阶段保持模型驻留，原运行每条启动并退出模型进程。原seconds包含进程启动、模型加载等，新request_seconds不包含阶段级模型初始化，不能直接据二者之差宣称结构化输出加速。模型初始化、显存峰值和版本保存在sessions/；所有原始输出、token ID、结束原因和每次尝试保存在各阶段目录。',
        f"无上限smoke另计{len(smoke)}次请求，未混入57条首轮；实际输出token合计{summary['smoke_costs']['output_tokens']['total']}。中途停止的1024-token版本保存在相邻20261030-v1目录，也没有混入本轮数学或成本均值。",
        f"新完成输出中超过旧300字符上限的条目：{old_length_rejects}。这部分可用性改善也受放宽字符上限影响，不能全部归因于语法约束。",'',
        '## 配置与归因限制','',
        '模型Qwen/Qwen3.5-9B，revision=c202236235762e1c871ad0ccb60c8ee5ba337b9a。复用原始57条请求中的完整用户JSON、原题、检查点编号和逐条随机种子，未输入参考答案、原9B结论、Codex标签或后续轨迹。prepare逐条检查了原始token往返、JSON载荷及Codex编号/ID/review_key对应。',
        '原采样保持temperature=0.7、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1.0；关闭Thinking，根据用户最新要求，max_tokens=None，不设置固定输出token上限。系统提示改动使每条输入统一增加19 token，用户JSON保持相同。BF16、148480上下文、2048预填充分块及0.8显存比例均沿用原请求。',
        f"实际vLLM版本{manifest['vllm_version']}。接口经本机签名检查，并查阅[vLLM对应版本文档]({manifest['docs']})，采用SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))和xgrammar后端。",'',
        '以下变化均已在manifest记录：']
lines+=['- '+x for x in manifest['differences']]
lines+=['','原核验标准中的当前采用推导、自我纠正和中途/终点区别保留。旧协议终点额外要求答案完整且受推理支持；Codex标准主要检查当前未纠正数学错误，两者存在口径差别。',
        '取消输出上限的直接例子：前缀1在中途停止的1024-token结构化版本中截断；本轮输出1286 token后闭合，前1024个token ID完全相同。这个样本的完成率改善可直接归因于延长输出预算，不应记成单纯JSON语法修复。证据见cap_removal_example.json；没有为此新增模型调用。',
        '原18条不可用结果中，9条原始输出可解析为JSON：其中5条原文为UNCERTAIN、4条原文为FAIL，均被旧反馈长度/字段规则拒绝。另4条有非法JSON转义、5条截断。本次保留旧解析结果，不把原文FAIL重新算作原有效FAIL。此次也按用户要求取消原1024-token输出上限，只有148480总上下文的物理边界。新的二选一约束会影响原本想表达数学不确定的样本，不能把这类变化称为单纯修复转义。',
        '二选一移除了原数学不确定出口；PASS增加诊断理由；删除300字符限制；加入语法约束改变可选token及随机采样路径。因此数学标签变化不能单独归因于“修好了JSON”。重试结果不能替换首轮统计；取消原1024上限带来的完成率变化也无法仅归因于JSON Schema。',
        '此次使用既有57条进行探索性诊断，Codex为暂时对照且自身存在复核范围和题意歧义限制。此次不创建或更新原实验质量放行文件。', '',
        '## 新9B与Codex不一致的条目','',f"共{len(disagreements)}条：{summary['disagreement_indices']}。双方理由如下，供复核，不裁定哪方一定正确。"]
lines[-1]=f"共{len(disagreements)}条：{summary['disagreement_indices']}。双方完整理由见[分歧明细](disagreements.md)及comparison.csv，供复核，不裁定哪方一定正确。"
details=['# 新9B与Codex分歧明细','', 'Codex仅为暂时对照；以下逐条保留双方理由，不裁定哪方一定正确。']
for r in disagreements:
    details+=['',f"### 前缀{r['source_index']}：9B {r['final_9b']} / Codex {r['codex_label']}",
            '',f"9B指向检查点：{r['new_first_error_step']}",
            '', '9B理由：'+r['new_diagnosis'],'', '9B返工提示：'+(r['new_hint'] or '无'),
            '', 'Codex理由：'+r['codex_reason']]
    if r['codex_quote']:details+=['','Codex引用原文：'+r['codex_quote']]
    if r['codex_caveat']:details+=['','Codex限制说明：'+r['codex_caveat']]
(RUN/'disagreements.md').write_text('\n'.join(details)+'\n')
lines+=['','## 下一步','',
        '先复核23、25、52、54的双方具体推导，再处理局部笔误和其他分歧。若需要量化纯语法约束的因果作用，应另行保持提示词、判定空间、解析规则和引擎生命周期完全相同作对照；本次不把混合配置变化当作该因果实验。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['first_valid','first_verdicts','first_failures','first_codex_matrix','retries','focus']},ensure_ascii=False,indent=2))
