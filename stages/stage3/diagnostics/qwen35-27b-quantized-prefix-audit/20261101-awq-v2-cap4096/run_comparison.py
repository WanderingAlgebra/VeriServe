"""Consistent full57 run with explicit4096-token technical output cap."""
from stages.stage3 import quantized_prefix_audit_bounded as audit


if __name__=='__main__':
    prior=audit.RUN.parent/'20261101-awq-v1'
    m=audit.read(audit.RUN/'manifest.json');audit.check_sources(m)
    records=[audit.read(prior/'capacity'/f'{i:03}.json') for i in m['capacity_indices']]
    assert all(r['finish_reason']=='stop' and r['status'] in ['completed','format_error'] for r in records)
    assert all(r['output_tokens']<4096 for r in records)
    assert all(r['input_tokens']==len(audit.read(audit.RUN/'requests'/f'{r["index"]:03}.json')['input_ids']) for r in records)
    audit.save(audit.RUN/'capacity_reuse.json',dict(source=str(prior/'capacity'),
        records_sha256={f'{r["index"]:03}':audit.sha(prior/'capacity'/f'{r["index"]:03}.json') for r in records},
        rationale='Same model revision, full input token IDs,81920 context,prefill,dtype and engine. '
          'Only output budget reduced from None to4096; all3 actual capacity outputs used less than4096. '
          'No additional capacity rerun needed. Prior format error remains uncorrected.',
        full_57_restarted=True,old_partial_run_not_merged=True,entrypoint_sha256=audit.sha(__file__)))
    audit.execute('comparison')
