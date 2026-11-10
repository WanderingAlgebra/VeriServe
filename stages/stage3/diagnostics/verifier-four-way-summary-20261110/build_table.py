"""Join saved verdicts by review_key. Does not rerun or relabel any sample."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
GPT_ROOT = ROOT.parent / 'VeriServe'
GPT = GPT_ROOT / 'stages/stage3/diagnostics/gpt61sol-prefix-audit/full-57-v3/labels.json'
NINE = ROOT / 'stages/stage3/diagnostics/qwen35-9b-structured-prefix-audit/20261030-unbounded-v2'
CODEX = NINE / 'full-review-v2-20261101/labels.json'
TWENTYSEVEN = ROOT / 'stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261101-awq-v2-cap4096'


def read(p):
    return json.loads(p.read_text())


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    gpt_rows, codex_rows = read(GPT), read(CODEX)
    gpt = {x['review_key']: x for x in gpt_rows}
    assert len(gpt) == len(codex_rows) == 57
    records, sources = [], {str(GPT), str(CODEX)}
    for c in codex_rows:
        i = c['source_index']
        g = gpt[c['review_key']]
        ng = NINE / 'first' / f'{i:03}' / 'attempt_01.json'
        nq = TWENTYSEVEN / 'comparison' / f'{i:03}.json'
        n, q = read(ng), read(nq)
        gp = GPT_ROOT / g['result_path']
        gr = read(gp)
        assert gr['status'] == 'completed' and gr['judgment']['label'] == g['label']
        assert g['source_index'] == i
        assert g['unique_id'] == c['unique_id'] and g['check_type'] == c['check_type']
        assert c['review_key'] == n['review_key'] == q['review_key'] == gr['review_key']
        assert n['status'] == 'completed'
        label27 = (q.get('judgment') or {}).get('verdict')
        display27 = label27 or {'format_error': '格式错误', 'truncation': '截断'}[q['status']]
        records.append(dict(index=i, question_number=c['question_number'],
            check_type=c['check_type'], review_key=c['review_key'], unique_id=c['unique_id'],
            gpt61sol=g['label'], codex=c['label'], nine_b=n['judgment']['verdict'],
            twentyseven_b=label27, twentyseven_b_status=q['status'],
            twentyseven_b_display=display27,
            gpt61sol_result_path=str(gp), nine_b_result_path=str(ng), twentyseven_b_result_path=str(nq)))
        sources.update(map(str, [gp, ng, nq]))
    assert [r['index'] for r in records] == list(range(1, 58))
    assert len({r['unique_id'] for r in records}) == 20
    keys = ['gpt61sol', 'codex', 'nine_b', 'twentyseven_b']
    names = ['GPT6.1Sol', 'Codex（第二轮）', '9B（结构化）', '27B AWQ 4-bit']
    counts = {name: dict(Counter(r[k] or 'UNAVAILABLE' for r in records)) for name, k in zip(names, keys)}
    differences = [r['index'] for r in records if r['gpt61sol'] != r['codex']]
    save(OUT / 'verdicts.json', records)
    save(OUT / 'summary.json', dict(prefixes=57, questions=20, counts=counts,
        gpt61sol_codex_disagreements=differences,
        twentyseven_unavailable=[r['index'] for r in records if r['twentyseven_b'] is None],
        original_labels_changed=False, new_model_calls=False, expert_gold=False))
    headers = ['编号', '题号', '类型', *names, '27B技术状态', 'review_key']
    def cells(r):
        return [r['index'], r['question_number'], '终点' if r['check_type']=='terminal' else '中途',
                r['gpt61sol'], r['codex'], r['nine_b'], r['twentyseven_b_display']]
    with (OUT / 'verdicts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in records:
            writer.writerow(cells(r) + [r['twentyseven_b_status'], r['review_key']])
    lines = ['# 四套验证判定总表（57条前缀，20道原题）', '',
        'PASS＝通过，FAIL＝未通过。格式错误/截断＝没有有效数学判决，不能当作PASS或FAIL。所有结果按review_key及样本身份对齐，未重新判题、未修改标签。', '',
        '|结果来源|PASS|FAIL|无有效判决|', '|---|---:|---:|---:|']
    for name in names:
        count = counts[name]
        lines.append(f'|{name}|{count.get("PASS",0)}|{count.get("FAIL",0)}|{count.get("UNAVAILABLE",0)}|')
    lines += ['', '## 逐条结果', '', '|编号|题号|类型|GPT6.1Sol|Codex第二轮|9B结构化|27B 4-bit|',
              '|---:|---:|---|---|---|---|---|']
    lines += ['|' + '|'.join(map(str, cells(r))) + '|' for r in records]
    lines += ['', '## 来源与限制', '',
        '- GPT6.1Sol：2026-10-30完成的`full-57-v3`完整汇总，含独立保存的技术重试/续跑结果，不混用pilot标签。',
        '- Codex：`full-review-v2-20261101`第二轮全文复核，33 PASS、24 FAIL；不是旧版36 PASS、21 FAIL。',
        '- 9B：`20261030-unbounded-v2/first`结构化首轮，52 PASS、5 FAIL；不是更早的39 PASS、18 UNCERTAIN版本。',
        '- 27B：`20261101-awq-v2-cap4096/comparison`完整57条运行，非Thinking、AWQ 4-bit、4096输出token上限。21、22格式错误，27输出截断。',
        '- GPT6.1Sol与Codex标签不同的编号：' + '、'.join(map(str, differences)) + '。此处只展示既有分歧，不作新的裁定。',
        '- 这些均为模型判断，Codex仅是暂定对照；PASS数量多不表示模型更准确，同题前缀也不是独立原题。', '',
        'CSV：[verdicts.csv](verdicts.csv)；完整来源：[sources.json](sources.json)。']
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    save(OUT / 'sources.json', dict(gpt61sol=str(GPT), codex=str(CODEX), nine_b=str(NINE/'first'),
        twentyseven_b=str(TWENTYSEVEN/'comparison'),
        sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(sources)},
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(counts, ensure_ascii=False))
    print('GPT/Codex differences:', differences)
    print('\n'.join(lines[lines.index('## 逐条结果')+2:lines.index('## 来源与限制')]))


if __name__ == '__main__':
    main()
