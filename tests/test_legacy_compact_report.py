import tempfile
import unittest
from pathlib import Path
from lfba_comparison.common import digest,read_json,write_json
from lfba_comparison.report import write_report

class LegacyReportTests(unittest.TestCase):
    def fixture(self, root, methods):
        write_json(root/'protocol.json',dict(seeds=[1],epochs=5,smoke=True,methods=methods))
        for method in methods:
            folder=root/'seed_1'/method
            choice=dict(anchor_id=0,candidates=[1,2,3,4])
            clean=1 if method.startswith('lfba_') else 3
            counts=([0,1,1,1,1] if method=='lfba_matched' else
                    [0,2,2,2,2] if method=='lfba_original' else [0,0,0,2,2])
            history=[dict(epoch=i+1,validation=dict(acc=.7,asr=.8 if i>=clean else None)) for i in range(5)]
            audit=[dict(epoch=i+1,recipient_count=n,cumulative_replacements=sum(counts[:i+1]),
                        candidate_count=0 if i<clean else 4,
                        candidate_digest=digest([] if i<clean else choice['candidates'])) for i,n in enumerate(counts)]
            result=dict(seed=1,method=method,epoch=5,anchor=0,target_class=1,test=dict(acc=.7,asr=.8,poison_acc=.2),
                        candidate_purity=.9,replacement_purity=.8,cumulative_replacements=sum(counts),
                        initial_digest='i',data_digest='d',prefix=[dict(epoch=1)],start_state='s',
                        final_model_digest='model',training_seconds=2,shared_clean_seconds=1,
                        peak_host_memory_bytes=1,peak_cuda_memory_bytes=0)
            result['independent_evaluation']=dict(epoch=5,test=result['test'],final_model_digest='model')
            write_json(folder/'metrics.json',result)
            write_json(folder/'experiment_data.json',dict(version=1,history=history,choice=choice,audit=audit,plans=[]))

    def test_legacy_matched_remains_in_report_and_metrics_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,['lfba_original','lfba_matched','auto_vote'])
            before=(root/'seed_1'/'lfba_matched'/'metrics.json').read_bytes()
            write_report(root)
            self.assertIn('lfba_matched',read_json(root/'summary.json'))
            self.assertIn('lfba_matched',(root/'results.md').read_text(encoding='utf-8'))
            self.assertEqual(before,(root/'seed_1'/'lfba_matched'/'metrics.json').read_bytes())

    def test_legacy_neighbor_methods_all_reported_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,['auto_single','auto_mean','auto_vote'])
            write_report(root)
            summary=read_json(root/'summary.json')
            self.assertIn('auto_single',summary);self.assertIn('auto_mean',summary)
            self.assertTrue((root/'validation_curves_lfba_vs_vote.png').exists())
            self.assertIn('—',(root/'results.md').read_text(encoding='utf-8'))

    def test_strict_records_require_config_and_valid_replacement_plans(self):
        from lfba_comparison.records import verified_group
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root,['lfba_original','auto_vote'])
            folder=root/'seed_1'/'auto_vote'
            metrics=read_json(folder/'metrics.json');metrics['status']='complete';write_json(folder/'metrics.json',metrics)
            with self.assertRaisesRegex(ValueError,'configuration'):
                verified_group(folder,'auto_vote',5,1,strict=True)
            write_json(folder/'config.json',dict(seed=1,comparison_method='auto_vote',anchor_idx=0,target_label=1,
                        trigger_dimensions=[],data_state=dict(train_ids=list(range(10)))))
            data=read_json(folder/'experiment_data.json')
            data['plans']=[dict(epoch=i,recipients=[1,2],donors=[5,6]) for i in (4,5)]
            write_json(folder/'experiment_data.json',data)
            verified_group(folder,'auto_vote',5,1,strict=True)
            data['plans'][0]['donors']=[1,2]
            write_json(folder/'experiment_data.json',data)
            with self.assertRaisesRegex(ValueError,'replacement'):
                verified_group(folder,'auto_vote',5,1,strict=True)
