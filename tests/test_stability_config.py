import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts.compare_lfba_auto_vote import parse_options,dry_run


class StabilityConfigTests(unittest.TestCase):
    def test_default_matrix_and_order(self):
        from lfba_comparison.stability_config import build_cases,case_options,stability_protocol
        options=parse_options(['--stability'])
        cases=build_cases(options)
        self.assertEqual(len(cases),6)
        self.assertEqual(options.seeds,[2004,592,3383,100,101])
        self.assertEqual(options.anchor_source,'fixed')
        self.assertEqual(options.output.name,'lfba_stability_full')
        self.assertEqual(stability_protocol(options)['fixed_lfba_anchor'],23470)
        self.assertEqual(stability_protocol(options)['version'],2)
        self.assertEqual(cases[0]['id'],'p_0p1_s_0p5')
        self.assertEqual(len({c['id'] for c in cases}),6)
        child=case_options(options,cases[1])
        self.assertEqual(child.output,options.output/cases[1]['id'])
        self.assertTrue(child.defer_cleanup)
        self.assertEqual(stability_protocol(options)['methods'],['auto_vote','lfba_original'])

    def test_invalid_settings_rejected(self):
        invalid=[['--stability','--anchor-source','auto_vote'],['--stability','--dataset','UCIHAR'],
                 ['--poison-rate','.5'],['--poison-rate','nan'],['--seeds','4294967296'],
                 ['--stability','--poison-rates','.1','.10'],['--poison-rates','.1'],
                 ['--stability','--select-rate','.7'],['--smoke','--poison-rate','.001']]
        for flags in invalid:
            with self.subTest(flags=flags),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):parse_options(flags)

    def test_smoke_matrix_feasible_and_no_output_created(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new'
            options=parse_options(['--stability','--smoke','--output',str(output)])
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture):dry_run(options)
            import json
            payload=json.loads(capture.getvalue())
            self.assertEqual(payload['tasks'],60)
            self.assertEqual(payload['pairs'],30)
            self.assertEqual(payload['train_size'],450)
            self.assertEqual(payload['execution_order'],['auto_vote','lfba_original'])
            self.assertTrue(all(c['replacements_per_epoch']>=1 for c in payload['cases']))
            self.assertFalse(output.exists())

    def test_budget(self):
        from lfba_comparison.stability_config import validate_budget
        result=validate_budget(45000,.1,.5)
        self.assertEqual(result['candidates'],4500)
        self.assertEqual(result['budgets'],{'lfba_original':87750,'auto_vote':83250})

    def test_old_smoke_protocol_cannot_silently_resume_with_new_sample_limit(self):
        from unittest.mock import patch
        from lfba_comparison.runner import frozen_protocol,run_experiment
        from lfba_comparison.common import write_json
        with tempfile.TemporaryDirectory() as directory:
            options=parse_options(['--run','--resume','--smoke','--output',directory])
            protocol=frozen_protocol(options)
            self.assertEqual(protocol['train_source_limit'],500)
            protocol['version']=3
            write_json(Path(directory)/'protocol.json',protocol)
            with patch('lfba_comparison.runner.assemble',side_effect=AssertionError('no training')):
                with self.assertRaisesRegex(ValueError,'identical frozen protocol'):
                    run_experiment(options)

    def test_close_parameters_do_not_alias_output_directories(self):
        from lfba_comparison.stability_config import build_cases
        options=parse_options(['--stability','--poison-rates','.1','.10000000000001'])
        cases=build_cases(options)
        self.assertEqual(len(cases),len({c['id'] for c in cases}))

    def test_explicit_execution_and_report_dispatch(self):
        from unittest.mock import patch,MagicMock
        from types import ModuleType
        from scripts.compare_lfba_auto_vote import main
        module=ModuleType('lfba_comparison.stability')
        module.run_stability=MagicMock(return_value=0)
        module.report_stability=MagicMock(return_value=0)
        with patch.dict('sys.modules',{'lfba_comparison.stability':module}), \
             patch('scripts.compare_lfba_auto_vote.dry_run',return_value=0):
            self.assertEqual(main(['--stability']),0)
            module.run_stability.assert_not_called()
            self.assertEqual(main(['--stability','--run']),0)
            module.run_stability.assert_called_once()
            self.assertEqual(main(['--stability','--report-only']),0)
            module.report_stability.assert_called_once()
            module.report_stability.return_value=1
            self.assertEqual(main(['--stability','--report-only']),1)


if __name__=='__main__':unittest.main()
