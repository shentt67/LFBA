import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from lfba_comparison.common import read_json, write_json
from scripts.compact_experiment_outputs import run, safe_path


class CompactTests(unittest.TestCase):
    def test_duplicate_comparison_does_not_read_locked_batch_lock(self):
        import shutil
        from scripts import compact_experiment_outputs as module
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            primary=root/'results/lfba_auto_vote_formal';batch.rename(primary)
            (primary/'batch.lock').write_bytes(b'0')
            duplicate=root/'.worktrees/lfba-vs-auto-vote/results/lfba_auto_vote_formal'
            shutil.copytree(primary,duplicate)
            original=module.same_file_contents
            def compare(a,b):
                if a.name=='batch.lock' or b.name=='batch.lock':raise PermissionError('locked')
                return original(a,b)
            with patch.object(module,'same_file_contents',side_effect=compare):result=run(root,True)
            self.assertFalse(result['errors'])
            self.assertTrue(primary.exists());self.assertFalse(duplicate.exists())

    def test_delete_failure_is_durably_journaled_before_unlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            original=Path.unlink
            observed=[]
            def unlink(path,*args,**kwargs):
                if path==folder/'history.json':
                    journals=list((root/'note').glob('artifact-cleanup-*.json'))
                    self.assertEqual(len(journals),1)
                    observed.append(read_json(journals[0])['pending_deletion'])
                    raise PermissionError('simulated deletion failure')
                return original(path,*args,**kwargs)
            with patch.object(Path,'unlink',new=unlink):result=run(root,True)
            saved=read_json(Path(result['log_path']))
            self.assertTrue(saved['errors']);self.assertTrue(observed)
            self.assertEqual(Path(saved['pending_deletion']['path']),Path('results/test/seed_1/auto_vote/history.json'))
            self.assertTrue((folder/'history.json').exists())

    def test_main_checkpoint_preserves_only_recoverable_final_selection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'results'/'native'/'seed_1';folder.mkdir(parents=True)
            write_json(folder/'config.json', {'seed':1})
            validation={'acc':.5}
            write_json(folder/'history.json', [{'epoch':1,'validation':validation}])
            write_json(folder/'metrics.json', {'checkpoint_epoch':1,'validation':validation,'test':validation})
            torch.save(dict(epoch=1,config={'anchor_idx':3},data_state={'train_ids':[1,2,3]},
                            trigger_dimensions=[5],attack_state={'poison_indexes':[1,3], 'current_plan':{1:2}}),
                       folder/'last_checkpoint.pth.tar')
            result=run(root,True)
            self.assertFalse(result['errors'])
            data=read_json(folder/'experiment_data.json')
            self.assertEqual(data['choice'],{'anchor_id':3,'candidates':[1,3]})
            self.assertEqual(data['plans'],[{'epoch':1,'recipients':[1],'donors':[2],
                                           'record_scope':'final_epoch_only'}])
            self.assertEqual(read_json(folder/'config.json')['data_state'],{'train_ids':[1,2,3]})
            self.assertFalse((folder/'last_checkpoint.pth.tar').exists())

    def fixture(self, root, complete=True):
        batch = root / 'results' / 'test'
        folder = batch / 'seed_1' / 'auto_vote'
        folder.mkdir(parents=True)
        write_json(batch / 'protocol.json', dict(seeds=[1], methods=['auto_vote'], epochs=1))
        write_json(folder / 'config.json', dict(seed=1, data_state={'train_ids':[1]}, trigger_dimensions=[]))
        write_json(folder / 'history.json', [dict(epoch=1, validation={'acc':.5})])
        write_json(folder / 'choice.json', dict(candidates=[1], anchor_id=2))
        write_json(folder / 'replacement_ids.json', [{'epoch':1,'recipients':[1],'donors':[2]}])
        (folder / 'epoch_metrics.csv').write_text('epoch,recipient_count\n1,1\n', encoding='utf-8')
        write_json(folder / 'metrics.json', dict(epoch=1, test={'acc':.5}, final_model_digest='same'))
        if complete:
            write_json(folder / 'independent_evaluation.json', dict(test={'acc':.5},final_model_digest='same'))
        (folder / 'last_checkpoint.pth.tar').write_bytes(b'fake-checkpoint')
        (folder / 'experiment.log').write_text('log')
        (folder / 'unknown.bin').write_bytes(b'unknown')
        (batch / 'results.md').write_text('report')
        (batch / 'curve.png').write_bytes(b'image')
        return batch, folder

    def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=run(root)
            after={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after);self.assertFalse(result['deleted'])

    def test_completed_migration_retains_data_images_unknown_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            result=run(root,True)
            self.assertFalse(result['errors'])
            self.assertFalse((folder/'last_checkpoint.pth.tar').exists())
            self.assertFalse((folder/'history.json').exists())
            self.assertEqual(read_json(folder/'experiment_data.json')['plans'][0]['donors'],[2])
            self.assertEqual(read_json(folder/'metrics.json')['status'],'complete')
            for path in (batch/'results.md',batch/'curve.png',folder/'unknown.bin',folder/'experiment.log'):
                self.assertTrue(path.exists())
            self.assertTrue(Path(result['log_path']).exists())

    def test_incomplete_checkpoint_and_data_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root,False)
            run(root,True)
            self.assertTrue((folder/'last_checkpoint.pth.tar').exists())
            self.assertTrue((folder/'history.json').exists())

    def test_failed_write_never_deletes_old_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            from scripts import compact_experiment_outputs as module
            original=module.write_json
            def fail(path,value):
                if Path(path).name=='experiment_data.json':raise OSError('write failed')
                return original(path,value)
            with patch.object(module,'write_json',side_effect=fail):result=run(root,True)
            self.assertTrue(result['errors'])
            self.assertTrue((folder/'last_checkpoint.pth.tar').exists())
            self.assertTrue((folder/'history.json').exists())

    def test_outside_workspace_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):safe_path(Path(tmp).parent/'outside',Path(tmp))

    def test_conflicting_old_and_new_data_is_never_deleted(self):
        for field, value in [('history', [{'epoch':1,'validation':{'acc':.9}}]),
                             ('choice', {'candidates':[99]}),
                             ('plans', [{'recipients':[99]}]),
                             ('audit', [{'epoch':'1','recipient_count':'99'}])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);batch,folder=self.fixture(root)
                write_json(folder/'experiment_data.json', {'version':1,field:value})
                result=run(root,True)
                self.assertTrue(result['errors'])
                for name in ('history.json','choice.json','replacement_ids.json','epoch_metrics.csv',
                             'last_checkpoint.pth.tar','independent_evaluation.json'):
                    self.assertTrue((folder/name).exists(), name)

    def test_conflicting_evaluation_proof_preserves_both_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,folder=self.fixture(root)
            record=read_json(folder/'metrics.json')
            proof=read_json(folder/'independent_evaluation.json')
            record['independent_evaluation']={**proof,'extra':'conflict'}
            write_json(folder/'metrics.json',record)
            result=run(root,True)
            self.assertTrue(result['errors'])
            self.assertTrue((folder/'independent_evaluation.json').exists())
            self.assertTrue((folder/'last_checkpoint.pth.tar').exists())
            self.assertEqual(read_json(folder/'metrics.json'),record)


if __name__=='__main__':unittest.main()
