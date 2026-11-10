"""CPU-only reproduction of xgrammar minLength string grammar defect.

Reads saved outputs; never loads model weights or changes the installed backend.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import xgrammar as xgr
from xgrammar.testing import _json_schema_to_ebnf
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'20261101-awq-v2-cap4096'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    snapshot=read(RUN/'manifest.json')['snapshot']
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    info=xgr.TokenizerInfo.from_huggingface(tok,vocab_size=248320)
    compiler=xgr.GrammarCompiler(info,max_threads=4)
    def accepts(compiled,text):
        m=xgr.GrammarMatcher(compiled)
        return m.accept_string(text) and m.is_completed()
    tests={'plain':'{"x":"abc"}', 'empty':'{"x":""}',
        'raw_backspace':'{"x":"a'+chr(8)+'b"}', 'raw_tab':'{"x":"a\tb"}',
        'escaped_backslash':r'{"x":"\\binom{7}{k}"}',
        'escaped_quote':r'{"x":"a\"b"}', 'escaped_newline':r'{"x":"a\nb"}'}
    # The repair is a diagnostic prototype, not a production private-API dependency.
    bad=r'[^"\\\r\n]{1,}'
    good=r'([^\0-\x1f"\\] | "\\" basic_escape){1,}'
    minimal={}
    for mode in ['plain_string','minLength1','repaired_minLength1']:
        prop={'type':'string'}
        if mode!='plain_string':prop['minLength']=1
        schema={'type':'object','properties':{'x':prop},'required':['x'],'additionalProperties':False}
        grammar=_json_schema_to_ebnf(schema,any_whitespace=True)
        if mode=='repaired_minLength1':
            assert bad in grammar
            grammar=grammar.replace(bad,good)
        compiled=compiler.compile_grammar(grammar)
        (HERE/(mode+'.ebnf')).write_text(grammar)
        minimal[mode]={}
        for name,text in tests.items():
            try:json.loads(text);valid=True
            except json.JSONDecodeError:valid=False
            minimal[mode][name]=dict(xgrammar_accepts=accepts(compiled,text),strict_json_valid=valid)
    assert minimal['minLength1']['raw_backspace']['xgrammar_accepts']
    assert not minimal['plain_string']['raw_backspace']['xgrammar_accepts']
    assert not minimal['repaired_minLength1']['raw_backspace']['xgrammar_accepts']
    assert not minimal['repaired_minLength1']['empty']['xgrammar_accepts']
    assert minimal['repaired_minLength1']['escaped_backslash']['xgrammar_accepts']
    actual=[]
    for index in [21,22]:
        rp=RUN/'comparison'/f'{index:03}.json';qp=RUN/'requests'/f'{index:03}.json'
        record,request=read(rp),read(qp)
        grammar=_json_schema_to_ebnf(request['schema'],any_whitespace=True)
        (HERE/f'{index:03}-original.ebnf').write_text(grammar)
        assert grammar.count(bad)==3
        repaired=grammar.replace(bad,good)
        (HERE/f'{index:03}-repaired-prototype.ebnf').write_text(repaired)
        variants={}
        for name,g in [('original',grammar),('repaired_prototype',repaired)]:
            compiled=compiler.compile_grammar(g);m=xgr.GrammarMatcher(compiled)
            control_mask=None;rejected_at=None
            mask=xgr.allocate_token_bitmask(1,248320)
            for position,token in enumerate(record['generated_ids']):
                if token==196:
                    m.fill_next_token_bitmask(mask)
                    control_mask=bool((int(mask[0,196//32])>>(196%32))&1)
                if not m.accept_token(token):
                    rejected_at=dict(position=position,token=token);break
            variants[name]=dict(backspace_token_allowed_by_bitmask=control_mask,
                rejected_at=rejected_at,terminated=m.is_terminated())
        assert variants['original']['backspace_token_allowed_by_bitmask'] is True
        assert variants['original']['terminated'] is True
        assert variants['repaired_prototype']['backspace_token_allowed_by_bitmask'] is False
        assert variants['repaired_prototype']['rejected_at']['token']==196
        actual.append(dict(index=index,record_sha256=sha(rp),request_sha256=sha(qp),
            replay=variants,strict_json_error=record['error']))
    result=dict(versions={p:importlib.metadata.version(p) for p in ['vllm','xgrammar','transformers']},
        tokenizer_type=type(tok).__name__,vocab_type=str(info.vocab_type),
        token196_grammar_bytes=repr(info.decoded_vocab[196]),token196_actual_decode=repr(tok.decode([196])),
        minimal_cases=minimal,original_saved_output_replays=actual,
        root_cause='xgrammar0.2.3 length-constrained string production omits C0 exclusion and JSON escapes',
        production_environment_modified=False,old_results_modified=False,new_gpu_calls=0,
        repair_status='Only CPU grammar/mask prototype verified; not installed or GPU-retested',
        upstream_issue='https://github.com/mlc-ai/xgrammar/issues/800',script_sha256=sha(Path(__file__)))
    (HERE/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
