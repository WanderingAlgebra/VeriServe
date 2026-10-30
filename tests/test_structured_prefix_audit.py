"""Small failure-path checks; GPU smoke is required separately."""
import json
import tempfile
import unittest
from pathlib import Path
from stages.stage3.structured_prefix_audit import attempts, make_prompt, parse_output
from veriserve_research.intervention.protocol import VERIFIER_PROMPT


class StructuredAuditCheck(unittest.TestCase):
    def test_completed_payloads_and_failures(self):
        good={'verdict':'PASS','first_error_step':None,'diagnosis':'Checked the displayed arithmetic.','hint':''}
        self.assertEqual(parse_output(json.dumps(good),'stop',[1])[1],'completed')
        self.assertEqual(parse_output(json.dumps(good),'length',[1])[:2],(None,'truncation'))
        bad={**good,'verdict':'UNCERTAIN'}
        self.assertEqual(parse_output(json.dumps(bad),'stop',[1])[1],'format_error')
        fail=dict(verdict='FAIL',first_error_step=1,diagnosis='a'*500,hint='Recompute this sum.')
        self.assertEqual(parse_output(json.dumps(fail),'stop',[1])[1],'completed')
        self.assertEqual(parse_output(json.dumps(fail),'stop',[2])[1],'format_error')
        self.assertEqual(parse_output('{"verdict":','stop',[1])[1],'format_error')
        self.assertEqual(parse_output('{"verdict":"PASS","verdict":"FAIL"}','stop',[1])[1],'format_error')
        fail['hint']=' '
        self.assertEqual(parse_output(json.dumps(fail),'stop',[1])[1],'format_error')

    def test_prompt_and_durable_attempt_discovery(self):
        old=VERIFIER_PROMPT.replace('After your own thinking, emit','Emit')
        new=make_prompt(old)
        self.assertEqual(old.splitlines()[:3],new.splitlines()[:3])
        self.assertNotIn('UNCERTAIN',new)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'first/001';folder.mkdir(parents=True)
            (folder/'intent_01.json').write_text('{}')
            self.assertEqual(attempts(Path(tmp),'first',1),[])
            (folder/'attempt_01.json').write_text('{}')
            self.assertEqual(len(attempts(Path(tmp),'first',1)),1)


if __name__=='__main__':unittest.main()
