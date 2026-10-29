"""Offline checks for selection, accounting, and paired summaries."""
import unittest
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from stages.stage3.verification_benefit_core import (LIMIT, arm_rule, choose_checkpoints,
                                                     paired_seed, summarize, token_accounting)


def point(n, step=1, back=False):
    return {'prefix_token_count': n, 'step_number': step, 'char_end': 10,
            'aligned_char_end': 9 if back else 10, 'boundary_aligned_back': back}


class BenefitCPU(unittest.TestCase):
    def test_deduplicates_and_uses_only_complete_points(self):
        record = {'unique_id': 'question', 'generated_ids': [0]*40000,
                  'boundaries': {'mode': 'thinking_paragraphs', 'format_errors': [],
                                 'steps': [point(9000, back=True), point(17000, 2), point(33000, 3)]}}
        choice = choose_checkpoints(record, 4)
        self.assertEqual([x['prefix_token_count'] for x in choice['checkpoints']], [17000, 33000])
        self.assertEqual(choice['checkpoints'][0]['thresholds'], [8192, 16384])
        self.assertEqual(choice['checkpoints'][0]['pair_seed'], paired_seed(4, 'question', 17000))

    def test_short_or_invalid_reference(self):
        record = {'unique_id':'x', 'generated_ids':[1]*7000,
                  'boundaries':{'mode':'thinking_paragraphs', 'format_errors':[], 'steps':[]}}
        result = choose_checkpoints(record, 0)
        self.assertEqual(len(result['checkpoints']), 0)
        self.assertEqual(result['thresholds']['8192']['reason'], 'ENDED_BEFORE_THRESHOLD')
        record['boundaries']['format_errors'] = ['bad']
        self.assertEqual(choose_checkpoints(record, 0)['thresholds']['8192']['reason'], 'INVALID_BOUNDARIES')

    def test_budget_and_full_answer_accounting(self):
        self.assertEqual(arm_rule(16384, 'PASS')['budget'], 49152)
        self.assertEqual(arm_rule(16384, 'UNCERTAIN')['budget'], 49152)
        self.assertEqual(arm_rule(16384, 'FAIL')['budget'], LIMIT)
        resumed = token_accounting(30000, 16384, 'NO_CHECK', 20000)
        self.assertEqual(resumed['answer_length_sum'], 66384)
        self.assertEqual(resumed['actual_4b_generated_tokens'], 50000)
        self.assertTrue(resumed['answer_length_sum_exceeds_limit'])
        restarted = token_accounting(30000, 16384, 'FAIL', 50000)
        self.assertEqual(restarted['answer_length_sum'], 80000)
        self.assertEqual(restarted['revoked_tokens'], 16384)
        with self.assertRaises(ValueError):
            token_accounting(30000, 16384, 'PASS', 49153)
        with self.assertRaises(ValueError):
            arm_rule(LIMIT, 'FAIL')

    def test_question_weighted_summary_and_unscorable(self):
        pairs = [{'unique_id':'a','A':{'label':1},'B':{'label':0}},
                 {'unique_id':'a','A':{'label':1},'B':{'label':0}},
                 {'unique_id':'b','A':{'label':0},'B':{'label':1}},
                 {'unique_id':'c','A':{'label':None},'B':{'label':0}}]
        out = summarize(pairs, bootstrap=100)
        self.assertEqual(out['paired_questions'], 2)
        self.assertEqual(out['accuracy_difference'], 0.)
        self.assertEqual(out['paired_checkpoints'], 3)

    def test_branch_uses_identical_seed_and_fail_restarts_from_prompt(self):
        from stages.stage3 import verification_benefit_run as run
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs): return [20, 21]
            def encode(self, text, add_special_tokens=False): return [70, 71]
            def convert_ids_to_tokens(self, token): return '<EOS>'
        tokenizer = Tokenizer()
        calls = []
        def worker(_run, _cfg, kind, ids, history, budget, seed, _tokenizer, key):
            calls.append((kind, ids, history, budget, seed))
            return {'generated_ids':[99], 'finish_reason':'stop', '_reused_after_interruption':False}
        def fake_decode(_tokenizer, ids):
            if ids == [99]:
                return '{"verdict":"FAIL","first_error_step":1,"diagnosis":"error","hint":"retry"}<EOS>'
            return 'reasoning'
        cfg = {'seed':3, 'max_verifier_input_tokens':100, 'max_verifier_output_tokens':10,
               'max_verifier_model_len':110}
        first = {'generated_ids':[11]*30000, 'prompt_ids':[1,2], 'token_sha256':'saved'}
        point = {'prefix_token_count':16384, 'pair_seed':42, 'thresholds':[8192], 'step_number':1}
        row = {'unique_id':'q', 'problem':'p', 'answer':'2'}
        with TemporaryDirectory() as temp, patch.object(run, 'worker_call', side_effect=worker), \
             patch.object(run, 'decode', side_effect=fake_decode), \
             patch.object(run, 'verification_units', return_value=([1], None)), \
             patch.object(run, 'grade', return_value={'label':0}):
            a = run.branch(Path(temp), cfg, 'pilot', row, first, point, 'A', tokenizer, tokenizer)
            b = run.branch(Path(temp), cfg, 'pilot', row, first, point, 'B', tokenizer, tokenizer)
            saved = next((Path(temp)/'branches').glob('*-B.json'))
            altered = json.loads(saved.read_text())
            altered['generated_ids'] = [123]
            saved.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                run.branch(Path(temp), cfg, 'pilot', row, first, point, 'B', tokenizer, tokenizer)
        self.assertEqual(a['verdict'], 'NO_CHECK')
        self.assertEqual(b['verdict'], 'FAIL')
        self.assertEqual(calls[0][1], [1,2]+[11]*16384)
        self.assertEqual(calls[0][3:], (49152, 42))
        self.assertEqual(calls[-1][1:5], ([1,2,70,71], [70,71], LIMIT, 42))
        self.assertEqual(b['tokens']['revoked_tokens'], 16384)

    def test_unverified_audit_stops_before_gpu(self):
        from stages.stage3 import verification_benefit_run as run
        with TemporaryDirectory() as temp, patch.object(run, 'worker_call') as worker:
            with self.assertRaisesRegex(RuntimeError, 'quality audit'):
                run.execute(Path(temp), {}, {}, 'pilot')
            worker.assert_not_called()

    def test_grade_uses_answer_and_completion_not_verdict(self):
        from veriserve_research.trajectory import grade
        text = 'reasoning</think>\nFinal answer: \\boxed{2}'
        self.assertEqual(grade(text, '2', 'EOS', 5, 'thinking_paragraphs',
                               'explanation_then_final_line')['label'], 0)
        self.assertIsNone(grade(text, '2', 'LENGTH_TRUNCATED', 5, 'thinking_paragraphs',
                                'explanation_then_final_line')['label'])

    def test_audit_accepts_model_labels_and_preserves_provenance(self):
        from stages.stage3 import verification_benefit_run as run
        from veriserve_research.artifacts.io import atomic_json, stable_hash
        cfg = {'generator_revision':'g', 'verifier_revision':'v', 'verifier_sampling':{},
               'max_verifier_output_tokens':1024}
        rows = [{'unique_id':str(n)} for n in range(20)]
        manifest = {'splits':{'verifier_audit':rows}}
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            records = folder/'audit_checks'
            records.mkdir()
            labels = []
            for n in range(40):
                key = str(n)
                correct = n < 20
                ids = [n, 10]
                record = {'unique_id':str(n % 20), 'review_key':key,
                    'input_ids':ids, 'input_sha256':stable_hash(ids), 'checkpoint_numbers':[1],
                    'results':[{'verdict':'PASS' if correct else 'FAIL', 'parser_error':None, 'completion':'EOS'}]}
                record['record_sha256'] = stable_hash(record)
                atomic_json(records/(key+'.json'), record)
                labels.append({'review_key':key, 'label':'CORRECT' if correct else 'INCORRECT',
                               'first_error_step':None if correct else 1, 'label_source':'model', 'labeler':'test-model'})
            atomic_json(folder/'verifier_audit_collection.json', {'status':'AWAITING_LABELS',
                'ordered_ids':[r['unique_id'] for r in rows], 'checks':40, 'config_hash':stable_hash(cfg)})
            atomic_json(folder/'verifier_audit_cohort.json', rows)
            label_path = folder/'input-labels.json'
            atomic_json(label_path, labels)
            report = run.audit_report(folder, cfg, manifest, label_path)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['label_sources'], {'human':0, 'model':40})
            run.require_audit(folder, cfg)
            altered = json.loads((folder/'verifier_audit_labels.json').read_text())
            altered[0]['label'] = 'UNDETERMINABLE'
            atomic_json(folder/'verifier_audit_labels.json', altered)
            with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                run.require_audit(folder, cfg)

    def test_model_label_parser_rejects_truncation_and_bad_checkpoints(self):
        from stages.stage3.verification_benefit_run import parse_model_label
        valid = '{"label":"INCORRECT","first_error_step":2}'
        self.assertEqual(parse_model_label(valid, True, [1,2])[0]['label'], 'INCORRECT')
        self.assertEqual(parse_model_label(valid, False, [1,2])[0]['label'], 'UNDETERMINABLE')
        self.assertEqual(parse_model_label(valid, True, [1])[0]['label'], 'UNDETERMINABLE')
        duplicated = '{"label":"CORRECT","label":"INCORRECT","first_error_step":null}'
        self.assertEqual(parse_model_label(duplicated, True, [1])[0]['label'], 'UNDETERMINABLE')

    def test_model_labeling_uses_blind_inputs_and_records_model_source(self):
        from stages.stage3 import verification_benefit_run as run
        from veriserve_research.artifacts.io import atomic_json, digest
        class Tokenizer:
            messages = None
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return [7, 8]
            def convert_ids_to_tokens(self, token): return '<EOS>'
        tokenizer = Tokenizer()
        cfg = {'generator_model':'Qwen/test', 'generator_revision':'revision', 'max_model_len':100,
               'audit_labeler_max_output_tokens':10, 'seed':3}
        review = {'review_key':'blind', 'problem':'problem', 'checkpoint_units':[{'checkpoint_number':1,'quoted_text':'work'}],
                  'check_type':'intermediate', 'reference_answer':'2', 'reference_solution':'proof'}
        out = {'generated_ids':[99], 'finish_reason':'stop'}
        with TemporaryDirectory() as temp, patch.object(run, 'load_tokenizer', return_value=tokenizer), \
             patch.object(run, 'backup'), patch.object(run, 'worker_call', return_value=out), \
             patch.object(run, 'decode', return_value='{"label":"CORRECT","first_error_step":null}<EOS>'):
            folder = Path(temp)
            atomic_json(folder/'verifier_audit_review_inputs.json', [review])
            atomic_json(folder/'verifier_audit_collection.json', {'status':'AWAITING_LABELS', 'checks':1,
                'review_inputs_sha256':digest(folder/'verifier_audit_review_inputs.json')})
            labels = run.model_label_audit(folder, cfg)
            self.assertEqual(labels[0]['label_source'], 'model')
            self.assertEqual(labels[0]['labeler'], 'Qwen/test@revision')
            self.assertEqual(labels[0]['label'], 'CORRECT')
            self.assertNotIn('verdict', tokenizer.messages[1]['content'])
            self.assertIn('proof', tokenizer.messages[1]['content'])

    def test_audit_reservations_append_without_reuse(self):
        from stages.stage3 import verification_benefit_run as run
        from veriserve_research.artifacts.io import atomic_json, digest
        with TemporaryDirectory() as temp, patch.object(run, 'ROOT', Path(temp)):
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            rows = [{'problem':f'question-{i}', 'answer':'1', 'solution':'proof'} for i in range(26)]
            for row in rows:
                row['unique_id'] = hashlib.sha256(row['problem'].encode()).hexdigest()
            data = source/'omni_math_rule.jsonl'
            data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            atomic_json(source/'manifest.json', {'splits':{'verifier_audit':rows[:20]},
                'unused_ids':[r['unique_id'] for r in rows[20:]], 'dataset_integrity':{'sha256':digest(data)}})
            cfg = {'question_manifest':'source/manifest.json'}
            manifest = {'identity':{'version':1}}
            folder = root/'new-run'
            folder.mkdir()
            first = run.reserve_audit(folder, cfg, manifest, 0)
            extended = run.reserve_audit(folder, cfg, manifest, 1)
            self.assertEqual([r['unique_id'] for r in first], [r['unique_id'] for r in extended[:20]])
            self.assertEqual(extended[-1]['unique_id'], rows[21]['unique_id'])
            other = root/'other-run'
            other.mkdir()
            with self.assertRaisesRegex(ValueError, 'Not enough'):
                run.reserve_audit(other, cfg, {'identity':{'version':2}}, 0)


if __name__ == '__main__':
    unittest.main()
