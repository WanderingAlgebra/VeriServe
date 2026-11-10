"""Synthetic controls for the missed division error; never use audit-source data."""
import json
import os
from pathlib import Path
import time

from stages.stage3.thinkprm_prefix_trial import (
    RUN, MODEL, SCHEMA, prompt, parse_final, inspect_quantized_model, read, save, sha)


def main():
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.reasoning import ReasoningParserManager
    from veriserve_research.vllm_backend import DevicePeak
    m=read(RUN/'manifest.json')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    parser=ReasoningParserManager.get_reasoning_parser('deepseek_r1')(tok)
    save(RUN/'controls/manifest.json',dict(config=m['config'],sampling=m['sampling'],
        implementation_sha256=sha(__file__),base_implementation_sha256=m['implementation_sha256'],
        research_sample=False,reason='The synthetic incorrect division received PASS.'))
    peak=DevicePeak();start=time.perf_counter()
    llm=LLM(**m['config']);torch.cuda.synchronize()
    save(RUN/'controls/engine.json',dict(seconds=time.perf_counter()-start,
        actual_quantization=llm.apply_model(inspect_quantized_model),device_peak_bytes=peak.stop()))
    problem='Solve for x: 2x + 3 = 7'
    prefix='Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 1'
    text,ids=prompt(tok,problem,prefix)
    card=f'''You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{prefix}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps
'''
    card_text=tok.apply_chat_template([dict(role='user',content=card)],tokenize=False,
        add_generation_prompt=True)+"\nLet's verify step by step:"
    card_ids=tok.encode(card_text,add_special_tokens=False)
    for name,text,ids,constrained in [
        ('fail_first_guidance',text,ids,True),
        ('fail_no_constraint',text,ids,False),
        ('fail_model_card',card_text,card_ids,False)]:
        save(RUN/'controls'/f'{name}.request.json',dict(prompt=text,input_ids=ids,
            structured=constrained,input_tokens=len(ids),expected='FAIL',research_sample=False))
        options=dict(m['sampling'],skip_special_tokens=False)
        if constrained:
            options['structured_outputs']=StructuredOutputsParams(json=SCHEMA)
        peak=DevicePeak();start=time.perf_counter()
        response=llm.generate([dict(prompt_token_ids=ids)],SamplingParams(**options),use_tqdm=False)[0]
        torch.cuda.synchronize();out=response.outputs[0];generated=list(out.token_ids)
        raw=tok.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        clean=generated[:-1] if generated and generated[-1]==tok.eos_token_id else generated
        reasoning,final=parser.extract_reasoning(tok.decode(clean,skip_special_tokens=False,
            clean_up_tokenization_spaces=False),None)
        judgment,status=parse_final(final,out.finish_reason) if name!='fail_model_card' else (None,'model_card_control')
        record=dict(name=name,input_tokens=len(ids),output_tokens=len(generated),raw_output=raw,
            generated_ids=generated,reasoning=reasoning,final_output=final,judgment=judgment,status=status,
            finish_reason=out.finish_reason,stop_reason=out.stop_reason,
            eos_observed=bool(generated and generated[-1]==tok.eos_token_id),
            reasoning_end_observed=parser.end_token_id in generated,
            returned_prompt_ids_match=list(response.prompt_token_ids)==ids,
            seconds=time.perf_counter()-start,device_peak_bytes=peak.stop())
        save(RUN/'controls'/f'{name}.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in
            ['generated_ids','raw_output','reasoning']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION','1')
    main()
