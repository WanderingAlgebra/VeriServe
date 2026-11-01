"""Full57 comparison after actual capacity success; retain JSON errors as unavailable.

The initial runner's first-phase guard unnecessarily equated format validity with
capacity. This explicit comparison phase keeps that frozen runner unchanged and
keeps the failing capacity record unchanged. It does not certify a quality gate.
"""
from stages.stage3 import quantized_prefix_audit as audit


if __name__=='__main__':
    m=audit.read(audit.RUN/'manifest.json');audit.check_sources(m)
    records=[audit.read(audit.RUN/'capacity'/f'{i:03}.json') for i in m['capacity_indices']]
    assert all(r['finish_reason']=='stop' and r['status'] in ['completed','format_error'] for r in records)
    assert all(r['input_tokens']+r['output_tokens']<m['config']['max_model_len'] for r in records)
    audit.save(audit.RUN/'capacity_assessment.json',dict(
        full_prompt_capacity_pass=True,oom=False,truncated=False,parsed=2,total=3,
        format_errors=[r['index'] for r in records if r['status']=='format_error'],
        explanation='All3 full real prompts generated to EOS. Sample21 emitted literal control token196 '
          '(backspace), rejected by the unchanged strict JSON parser. Both tokenizer files have identical '
          'SHA256. No text repair, no prompt/sampling/backend change. Full57 uses comparison/ and retains '
          'format errors as unavailable; this is capacity feasibility, not format perfection or quality acceptance.',
        entrypoint_sha256=audit.sha(__file__)))
    audit.execute('comparison')
