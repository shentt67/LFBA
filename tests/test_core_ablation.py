"""Core ablation invariants, independent of the legacy two-arm protocol."""
import unittest
import tempfile
from pathlib import Path
import torch

from lfba_comparison.core_policy import METHODS, select_core, schedule


def observations():
    generator = torch.Generator().manual_seed(34)
    return [dict(epoch=i+1, ids=torch.arange(40),
                 gradients=torch.randn(40, 7, generator=generator)) for i in range(3)]


class CorePolicyTests(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(schedule(40, 4500), [0]*3+[2250]*37)
        self.assertEqual(sum(schedule(40, 4500)), 83250)
        self.assertEqual(schedule(5, 45), [0, 0, 0, 22, 22])

    def test_shared_anchor_and_last_round_bridge(self):
        from attack.neighbor_vote import select_all, ordered_candidates
        obs = observations()
        original = select_all(obs, 4)
        selected = select_core(obs, 4)
        for method in ('auto_single', 'auto_mean', 'auto_vote'):
            self.assertEqual(selected['results'][method], original['results'][method])
        bridge = selected['results']['auto_rank_last']
        anchor = original['results']['auto_vote']['anchor_id']
        self.assertEqual(bridge['anchor_id'], anchor)
        scores = original['evidence']['auto_vote']['similarities'][-1].numpy()
        ids = original['ids'].numpy()
        expected = sorted(ids[ordered_candidates(ids, scores, 4, anchor)].tolist())
        self.assertEqual(bridge['candidates'], expected)
        for result in selected['results'].values():
            self.assertEqual(len(set(result['candidates'])), 4)
            self.assertNotIn(result['anchor_id'], result['candidates'])
        # This fixture must actually discriminate the three neighborhood rules.
        self.assertNotEqual(bridge['candidates'], selected['results']['auto_mean']['candidates'])
        self.assertNotEqual(selected['results']['auto_mean']['candidates'], selected['results']['auto_vote']['candidates'])

    def test_id_permutation_and_invalid_observations(self):
        obs = observations()
        before = select_core(obs, 4)['results']
        for item in obs:
            item['ids'] = item['ids'].flip(0)
            item['gradients'] = item['gradients'].flip(0)
        self.assertEqual(before, select_core(obs, 4)['results'])
        obs[0]['ids'][0] = obs[0]['ids'][1]
        with self.assertRaises(ValueError): select_core(obs, 4)
        obs = observations()
        obs[1]['gradients'].zero_()
        with self.assertRaises(ValueError): select_core(obs, 4)

    def test_vote_ties(self):
        import numpy as np
        from attack.neighbor_vote import ordered_candidates
        ids = np.array([1, 2, 3, 4])
        scores = np.array([1., .2, .3, .3])
        votes = np.array([3, 2, 2, 2])
        self.assertEqual(ordered_candidates(ids, scores, 2, 1, votes).tolist(), [2, 3])


class CoreInterfaceTests(unittest.TestCase):
    def test_result_metadata_rejects_corrupt_fields(self):
        import copy
        from lfba_comparison.core_runner import verify_result_metadata
        from lfba_comparison.common import digest
        choice = dict(anchor_id=7, candidates=[0, 1, 2, 3])
        quality = dict(full=.75, full_denominator=4, without_anchor=.75,
                       without_anchor_denominator=4, anchor_in_pool=False)
        final = dict(config=dict(anchor_idx=7, target_label=2), data_state={'train_ids':list(range(10))},
                     comparison=dict(audit=[dict(recipient_count=2, recipient_purity=.5),
                         dict(recipient_count=2, recipient_purity=1., candidate_purity=.75,
                              candidate_purity_without_anchor=.75)]))
        result = dict(anchor=7, target_class=2, data_digest=digest(final['data_state']),
                      candidate_digest=digest(choice['candidates']), candidate_purity=quality,
                      recipient_purity=.75, wrong_candidates=1)
        verify_result_metadata(result, final, choice)
        for key, bad_value in [('anchor', 8), ('target_class', 99), ('candidate_purity', 0),
                               ('recipient_purity', 0), ('wrong_candidates', 0),
                               ('data_digest', 'wrong'), ('candidate_digest', 'wrong')]:
            changed = copy.deepcopy(result); changed[key] = bad_value
            with self.subTest(field=key), self.assertRaises(ValueError):
                verify_result_metadata(changed, final, choice)

    def test_dry_run_and_report_only_do_not_train(self):
        from scripts.run_core_ablation import parse_options, preflight
        from lfba_comparison.core_runner import protocol
        args = parse_options([])
        self.assertFalse(args.run)
        self.assertEqual(args.seeds, [2004, 592, 3383, 100])
        self.assertEqual(sum(schedule(protocol(args)['epochs'], 4500)), 83250)
        args = parse_options(['--smoke'])
        self.assertEqual(args.seeds, [2004])
        self.assertTrue(preflight(args)['training_started'] is False)

    def test_report_missing_is_not_zero(self):
        from lfba_comparison.core_report import write_report
        from lfba_comparison.common import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/'protocol.json', dict(seeds=[2004], epochs=5, smoke=True))
            result = write_report(root)
            self.assertEqual(result['complete'], 0)
            self.assertFalse(result['all_complete'])
            self.assertEqual(result['methods']['auto_vote']['n'], 0)


class CoreStateTests(unittest.TestCase):
    def test_shared_prefix_isolation_and_interrupted_resume(self):
        from tests.helpers import make_trainer
        from lfba_comparison.core_trainer import CoreTrainer
        from lfba_comparison.common import digest
        from utils.utils import set_seed
        from utils.checkpoint import load_checkpoint, cpu_tree
        from utils.artifacts import runtime_checkpoint

        def make(path, method, checkpoint=None, choice=None):
            base = make_trainer(path, attack='LFBA', epoch=3 if method=='capture' else 5,
                                early_stop=0, pretrain_stage=3, comparison_method=method)
            return CoreTrainer(base.device, base.model_list, [], None, base.optimizer_list,
                               base.criterion, base.train_loader, base.test_loader,
                               base.test_asr_loader, base.trigger_dimensions, base.logger,
                               base.args, checkpoint, base.val_loader, base.val_asr_loader,
                               choice=choice)

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            set_seed(100)
            prefix_trainer = make(root/'prefix', 'capture')
            prefix_trainer.train()
            prefix = load_checkpoint(runtime_checkpoint(root/'prefix'))
            before = digest(prefix)
            selected = select_core(prefix['comparison']['observations'], 4)
            starts = []
            for method in METHODS:
                choice = selected['results'][method]
                obj = make(root/method, method, prefix, choice)
                obj._restore()
                starts.append(digest(obj.state_identity()))
                self.assertEqual(len(obj.observations), 3)
                obj.train()
                self.assertEqual(obj.cumulative, 4)
                self.assertEqual(before, digest(prefix))
                continuous = cpu_tree([m.state_dict() for m in obj.model_list])
                plans = cpu_tree(obj.plans)
                metrics = obj.test(4)
                # Simulate interruption after epoch 4, preserving epoch=5 in saved config.
                partial = make(root/'partial', method, prefix, choice)
                original_snapshot = partial._snapshot
                def interrupt_snapshot():
                    snap = original_snapshot()
                    if partial.completed_epoch == 4:
                        from utils.checkpoint import atomic_save
                        atomic_save(snap, runtime_checkpoint(root/'partial'))
                        raise InterruptedError('test interruption')
                    return snap
                partial._snapshot = interrupt_snapshot
                with self.assertRaises(InterruptedError): partial.train()
                resumed = make(root/'resume', method, load_checkpoint(runtime_checkpoint(root/'partial')), choice)
                resumed.train()
                self.assertEqual(resumed.plans, plans)
                self.assertEqual(digest([m.state_dict() for m in resumed.model_list]), digest(continuous))
                self.assertEqual(resumed.test(4), metrics)
                self.assertEqual(before, digest(prefix))
            self.assertEqual(len(set(starts)), 1)


if __name__ == '__main__': unittest.main()
