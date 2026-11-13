"""Budget-independent label-free selection and predeclared acceptance contracts."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from attack.neighbor_vote import select_all
from lfba_comparison.common import digest
from utils.checkpoint import capture_rng


def observations():
    generator = torch.Generator().manual_seed(34)
    return [dict(epoch=i + 1, ids=torch.arange(40),
                 gradients=torch.randn(40, 7, generator=generator)) for i in range(3)]


class SelectionTests(unittest.TestCase):
    def test_budget_does_not_change_anchor_ranking_or_rng(self):
        from lfba_comparison.decoupled_policy import build_ranking, choose
        obs = observations()
        before = digest(capture_rng({}))
        ranking = build_ranking(obs, 4)
        for method in ('decoupled_mean', 'decoupled_vote'):
            small, large = choose(ranking, method, 2), choose(ranking, method, 12)
            self.assertEqual(small['anchor_id'], large['anchor_id'])
            self.assertEqual(small['ranking_digest'], large['ranking_digest'])
            self.assertTrue(set(small['candidates']) < set(large['candidates']))
            self.assertNotIn(small['anchor_id'], large['candidates'])
        self.assertEqual(before, digest(capture_rng({})))
        self.assertNotIn('labels', inspect.signature(build_ranking).parameters)
        self.assertNotIn('labels', inspect.signature(choose).parameters)

    def test_reference_budget_matches_legacy(self):
        from lfba_comparison.decoupled_policy import build_ranking, choose
        obs = observations()
        legacy = select_all(obs, 4)['results']
        ranking = build_ranking(obs, 4)
        for name in ('mean', 'vote'):
            actual = choose(ranking, 'decoupled_' + name, 4)
            self.assertEqual(actual['anchor_id'], legacy['auto_' + name]['anchor_id'])
            self.assertEqual(actual['candidates'], legacy['auto_' + name]['candidates'])

    def test_permutation_filtering_ties_and_invalid_capacity(self):
        from lfba_comparison.decoupled_policy import build_ranking, choose
        obs = observations()
        obs[0]['gradients'][0].zero_()
        obs[1]['gradients'][1, 0] = float('nan')
        a = build_ranking(obs, 4)
        for row in obs:
            row['ids'] = row['ids'].flip(0)
            row['gradients'] = row['gradients'].flip(0)
        b = build_ranking(obs, 4)
        self.assertEqual(a['rankings'], b['rankings'])
        self.assertEqual(a['excluded_count'], 2)
        self.assertFalse({0, 1} & set(a['rankings']['decoupled_vote']))
        with self.assertRaises(ValueError): choose(a, 'decoupled_vote', 38)
        with self.assertRaises(ValueError): choose(a, 'decoupled_vote', 0)
        with self.assertRaises(ValueError): build_ranking(obs, 19)
        obs[0]['ids'][0] = obs[0]['ids'][1]
        with self.assertRaises(ValueError): build_ranking(obs, 4)
        tied = [dict(epoch=i, ids=torch.arange(12), gradients=torch.ones(12, 3))
                for i in (1, 2, 3)]
        ranking = build_ranking(tied, 3)
        self.assertEqual(ranking['anchor_id'], 0)
        self.assertEqual(choose(ranking, 'decoupled_vote', 2)['candidates'], [1, 2])
        tied[0]['epoch'] = 2
        with self.assertRaises(ValueError): build_ranking(tied, 3)


class AcceptanceTests(unittest.TestCase):
    def test_individual_floors_and_exact_std_ratio_boundary(self):
        from lfba_comparison.decoupled_report import acceptance
        old = [dict(seed=i, asr=a, acc=.8) for i, a in enumerate([.8, .8, .9, .9])]
        new = [dict(seed=i, asr=a, acc=.8) for i, a in enumerate([.85, .85, .925, .925])]
        result = acceptance(new, old, list(range(4)))
        self.assertAlmostEqual(result['std_ratio'], .75)
        self.assertTrue(result['passed'])
        one = acceptance([dict(seed=0, asr=.8, acc=.8)], [dict(seed=0, asr=.9, acc=.8)], [0])
        self.assertTrue(one['checks']['min_asr'])
        self.assertTrue(one['checks']['min_asr_delta'])
        one = acceptance([dict(seed=0, asr=.799, acc=.789)], [dict(seed=0, asr=.9, acc=.8)], [0])
        self.assertFalse(one['checks']['min_asr'])
        self.assertFalse(one['checks']['min_asr_delta'])
        self.assertFalse(one['checks']['mean_acc'])

    def test_threshold_boundaries_are_in_percentage_points(self):
        from lfba_comparison.decoupled_report import acceptance
        old = [dict(seed=i, asr=a, acc=.8) for i, a in enumerate([.9, .9, 1., 1.])]
        new = [dict(seed=i, asr=a, acc=.79) for i, a in enumerate([.85, .85, .95, .95])]
        result = acceptance(new, old, list(range(4)))
        self.assertAlmostEqual(result['mean_asr_delta_pp'], -5.)
        self.assertTrue(result['checks']['mean_asr'])
        self.assertTrue(result['checks']['mean_acc'])
        self.assertFalse(result['checks']['asr_std'])
        self.assertFalse(result['passed'])
        for row in new: row['asr'] = .95
        result = acceptance(new, old, list(range(4)))
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['baseline_asr_std_pp'], 5.773502691896258)

    def test_missing_duplicate_and_zero_variance_are_not_success(self):
        from lfba_comparison.decoupled_report import acceptance
        rows = [dict(seed=i, asr=.9, acc=.8) for i in range(4)]
        result = acceptance(rows[:-1], rows, list(range(4)))
        self.assertFalse(result['passed'])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['checks']['asr_std'])
        with self.assertRaises(ValueError): acceptance(rows + rows[:1], rows, list(range(4)))
        result = acceptance(rows, rows, list(range(4)))
        self.assertIsNone(result['std_ratio'])
        self.assertIsNone(result['checks']['asr_std'])
        self.assertFalse(result['passed'])
        result = acceptance(rows[:1], rows[:1], [0])
        self.assertIsNone(result['new_asr_std_pp'])
        changed = copy.deepcopy(rows); changed[0]['asr'] = None
        self.assertFalse(acceptance(changed, rows, list(range(4)))['passed'])


if __name__ == '__main__':
    unittest.main()
