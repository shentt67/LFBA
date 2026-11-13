"""Unit tests for stabilized scale-decoupled, leverage-aware, core-gated label-free backdoor selection."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from attack.neighbor_vote import reference_radius, select_all
from lfba_comparison.runner import configuration
from lfba_comparison.trainer import _recipient_order
from scripts.compare_lfba_auto_vote import parse_options


def _clustered_observations(n_samples=600, dim=16, n_classes=10, seed=42, weak_class0=False, boundary_outlier=False):
    """Build 3 rounds of synthetic clustered gradients without exposing labels to select_all."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    ids = torch.arange(n_samples, dtype=torch.long)
    # Orthogonal class prototypes
    q, _ = np.linalg.qr(rng.standard_normal((dim, n_classes)))
    prototypes = q[:, :n_classes].T.astype(np.float32)

    observations = []
    for ep in range(1, 4):
        grads = np.zeros((n_samples, dim), dtype=np.float32)
        for i in range(n_samples):
            c = labels[i]
            noise_scale = 0.08 if c == 1 else (0.06 if (weak_class0 and c == 0) else 0.22)
            vec = prototypes[c] + rng.standard_normal(dim).astype(np.float32) * noise_scale
            vec = vec / max(np.linalg.norm(vec), 1e-8)
            # Class 1 has strong attacker gradient leverage; class 0 has tiny norm if weak_class0=True
            if weak_class0 and c == 0:
                mag = 0.01
            elif c == 1:
                mag = 3.5
            else:
                mag = 1.0
            grads[i] = vec * mag
        if boundary_outlier:
            # Make sample 0 a shifting boundary outlier that looks tight in single rounds but drifts across rounds
            grads[0] = prototypes[(ep % 3) + 2] * 4.0
        observations.append(dict(epoch=ep, ids=ids.clone(), gradients=torch.from_numpy(grads)))
    return observations, labels


class StabilizedSelectionTests(unittest.TestCase):
    def test_scale_decoupled_anchor_invariance_and_small_p_stability(self):
        obs, labels = _clustered_observations(n_samples=600, seed=7)
        anchors = []
        for p in (0.01, 0.05, 0.10, 0.20):
            count = int(600 * p)
            choice = select_all(obs, count, stabilized=True)['results']['auto_vote']
            anchors.append(choice['anchor_id'])
            target_label = labels[choice['anchor_id']]
            cand_purity = float(np.mean(labels[choice['candidates']] == target_label))
            self.assertGreaterEqual(cand_purity, 0.95, f'Low candidate purity at p={p}')
            # Negative-similarity donor filtering must exclude same-class samples
            donor_same_class = float(np.mean(labels[choice['donor_pool']] == target_label))
            self.assertLessEqual(donor_same_class, 0.02, f'Same-class donor leakage at p={p}')
        # Anchor must be invariant across all poison rates p
        self.assertEqual(len(set(anchors)), 1)

    def test_purity_preserving_core_gating_at_large_poison_rate(self):
        obs, labels = _clustered_observations(n_samples=600, seed=19)
        # Request p=0.20 -> count=120, which is 2x larger than a single class (60 samples)
        choice = select_all(obs, 120, stabilized=True)['results']['auto_vote']
        target_label = labels[choice['anchor_id']]
        # Core gating caps the candidate pool near k_ref=48 instead of blindly taking 120 impure samples
        self.assertLessEqual(len(choice['candidates']), reference_radius(600))
        self.assertGreaterEqual(len(choice['candidates']), 30)
        cand_purity = float(np.mean(labels[choice['candidates']] == target_label))
        self.assertGreaterEqual(cand_purity, 0.95)

    def test_leverage_weighting_avoids_zero_leverage_class(self):
        obs, labels = _clustered_observations(n_samples=600, seed=23, weak_class0=True)
        choice = select_all(obs, 30, stabilized=True)['results']['auto_vote']
        # Class 0 has slightly tighter direction noise (0.06 vs 0.08) but near-zero gradient norm (0.01 vs 3.5).
        # Leverage weighting must select the high-leverage class 1 rather than the dead-leverage class 0.
        self.assertEqual(int(labels[choice['anchor_id']]), 1)

    def test_medoid_self_consistency_rejects_drifting_outlier_anchor(self):
        obs, labels = _clustered_observations(n_samples=600, seed=31, boundary_outlier=True)
        choice = select_all(obs, 6, stabilized=True)['results']['auto_vote']
        self.assertNotEqual(choice['anchor_id'], 0)
        self.assertEqual(int(labels[choice['anchor_id']]), 1)

    def test_direction_filtered_and_exposure_balanced_recipient_scheduling(self):
        ids = torch.tensor([10, 11, 12, 99], dtype=torch.long)
        # Samples 10, 11, 12 align along +e0 with moderate norms; sample 99 is a wrong-class outlier
        # pointing along -e0 with a larger norm.
        grads = torch.tensor([
            [2.0, 0.1, 0.0],
            [1.9, -0.1, 0.0],
            [1.8, 0.0, 0.1],
            [-3.0, 0.0, 0.0],
        ], dtype=torch.float32)
        order_ep1 = _recipient_order(ids, grads, plans=[], stabilized=True)
        # Wrong-direction outlier (index 3, id 99) must be pushed to the back despite highest norm 3.0
        self.assertEqual(int(ids[order_ep1[-1]]), 99)
        self.assertEqual(int(ids[order_ep1[0]]), 10)
        # After sample 10 is used twice, sample 11 or 12 rotates to the front
        plans = [dict(epoch=4, recipients=[10], donors=[200]), dict(epoch=5, recipients=[10], donors=[201])]
        order_ep3 = _recipient_order(ids, grads, plans=plans, stabilized=True)
        self.assertIn(int(ids[order_ep3[0]]), (11, 12))

    def test_ucihar_calibrated_trigger_defaults_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            opts = parse_options(['--dataset', 'UCIHAR', '--output', str(Path(tmp) / 'out')])
            cfg = configuration(opts, 2004, 'auto_vote', Path(tmp) / 'out')
            self.assertEqual(cfg.poison_dimensions, 20)
            self.assertAlmostEqual(cfg.trigger_feature_clip, 2.5)
            opts_custom = parse_options([
                '--dataset', 'UCIHAR', '--poison-dimensions', '28', '--trigger-feature-clip', '3.0',
                '--output', str(Path(tmp) / 'out2'),
            ])
            cfg_custom = configuration(opts_custom, 2004, 'auto_vote', Path(tmp) / 'out2')
            self.assertEqual(cfg_custom.poison_dimensions, 28)
            self.assertAlmostEqual(cfg_custom.trigger_feature_clip, 3.0)


if __name__ == '__main__':
    unittest.main()
