import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfba_comparison.common import write_json
from lfba_comparison.stability_report import write_stability_report,statistics_record


class StabilityReportTests(unittest.TestCase):
    def test_sample_standard_deviation(self):
        result=statistics_record([1,3])
        self.assertEqual(result['mean'],2)
        self.assertAlmostEqual(result['std'],2**.5)
        self.assertEqual((result['min'],result['max']),(1,3))

    def fixture(self,root):
        write_json(root/'protocol.json',dict(kind='stability',seeds=[1,2],epochs=5,smoke=True,
            dataset='CIFAR10',baseline=dict(poison_rate=.1,select_rate=.5),
            cases=[dict(id='base',poison_rate=.1,select_rate=.5)]))

    def test_all_missing_has_no_fake_zero_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            out=write_stability_report(root)
            self.assertFalse(out['all_complete'])
            self.assertEqual(out['missing_pairs'],2)
            self.assertIsNone(out['cases'][0]['paired']['delta_asr']['mean'])
            for name in ('results.md','summary.json','all_epoch_metrics.csv','stability_parameters.png'):
                self.assertTrue((root/name).exists())

    def test_paired_statistics_and_partial_failure(self):
        def group(folder,method,epochs,seed,rate,strict=True):
            if seed==2 and method=='lfba_original':raise ValueError('missing')
            gain=.04 if seed==1 else .1
            result=dict(seed=seed,method=method,anchor=7,target_class=1,
                        test=dict(acc=.8,asr=.6+(gain if method=='auto_vote' else 0)))
            return result,dict(history=[dict(epoch=1,validation=dict(acc=.8,asr=.6))]),[dict(epoch=1)]
        def pair(directory,epochs,seed,p,s):
            return {m:group(directory/m,m,epochs,seed,s)[0] for m in ('auto_vote','lfba_original')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            write_json(root/'base/failures.json',[dict(seed=2,error='failed')])
            with patch('lfba_comparison.stability_report.verified_group',side_effect=group), \
                 patch('lfba_comparison.stability_report._pair_records',side_effect=pair):
                out=write_stability_report(root)
            self.assertEqual(out['completed_pairs'],1)
            self.assertEqual(out['failed_pairs'],1)
            self.assertEqual(out['cases'][0]['methods']['auto_vote']['acc']['n'],2)
            self.assertAlmostEqual(out['cases'][0]['paired']['delta_asr']['mean'],.04)
            self.assertIsNone(out['cases'][0]['paired']['delta_asr']['std'])

    def test_configuration_narrative_describes_reversal_and_missing(self):
        from lfba_comparison.stability_report import _describe_case,_describe_trends
        protocol=dict(baseline=dict(poison_rate=.1,select_rate=.5),epochs=40)
        def case(p,gain):
            a=statistics_record([.8,.9]);b=statistics_record([.7,.8])
            return dict(poison_rate=p,select_rate=.5,completed_pairs=2,expected_pairs=5,
                failed_pairs=1,missing_pairs=2,
                methods={m:dict(acc=a,asr=b) for m in ('auto_vote','lfba_original')},
                paired=dict(delta_asr=statistics_record([gain,gain]),delta_acc=statistics_record([0,0]),positive_fraction=float(gain>0)))
        cases=[case(.1,.05),case(.2,-.02)]
        prose='\n'.join(_describe_case(cases[1],protocol)+_describe_trends(cases,protocol))
        for word in ('更低','阶段性','最差','较基准','反转','标准差'):
            self.assertIn(word,prose)

    def test_anchor_mismatch_is_failed_pair_not_success(self):
        def group(folder,method,epochs,seed,rate,strict=True):
            r=dict(seed=seed,method=method,anchor=1 if method=='auto_vote' else 2,
                   target_class=0,test=dict(acc=.8,asr=.7))
            return r,dict(history=[]),[]
        def pair(directory,epochs,seed,p,s):
            return {m:group(directory/m,m,epochs,seed,s)[0] for m in ('auto_vote','lfba_original')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('lfba_comparison.stability_report.verified_group',side_effect=group), \
                 patch('lfba_comparison.stability_report._pair_records',side_effect=pair):
                out=write_stability_report(root)
            self.assertEqual(out['completed_pairs'],0)
            self.assertEqual(out['failed_pairs'],2)


if __name__=='__main__':unittest.main()
