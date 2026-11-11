import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.dataset import NUSWIDE_VFL
from dataset.utils import get_labeled_data


class NusValidationTests(unittest.TestCase):
    def prepare(self, root):
        for folder in ('Groundtruth/TrainTestLabels', 'Low_Level_Features', 'NUS_WID_Tags'):
            (root / folder).mkdir(parents=True)
        for split in ('Train', 'Test'):
            for name, values in [('a', [1, 0, 1]), ('b', [0, 1, 1])]:
                np.savetxt(root / 'Groundtruth/TrainTestLabels' / f'Labels_{name}_{split}.txt', values, fmt='%d')
            np.savetxt(root / 'Low_Level_Features' / f'{split}_Normalized_a.dat', np.ones((3, 634)))
            np.savetxt(root / 'NUS_WID_Tags' / f'{split}_Tags1k.dat', np.ones((3, 1000)), delimiter='\t')

    def test_missing_root_reports_all_splits_and_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                NUSWIDE_VFL(tmp, ['a', 'b'], True, None)
            message = str(caught.exception)
            for item in ('NUS_WIDE', 'Labels_a_Train.txt', 'Labels_b_Test.txt',
                         'Train_Normalized', 'Test_Tags1k.dat', '--data_dir', 'README'):
                self.assertIn(item, message)
            self.assertIn(str(Path(tmp).resolve()), message)

    def test_each_missing_component_is_reported_before_reading(self):
        for relative in ('Groundtruth/TrainTestLabels/Labels_b_Test.txt',
                         'Low_Level_Features/Test_Normalized_a.dat', 'NUS_WID_Tags/Test_Tags1k.dat'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'NUS_WIDE'
                self.prepare(root)
                (root / relative).unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'NUS-WIDE'):
                    NUSWIDE_VFL(tmp, ['a', 'b'], True, None)

    def test_dimensions_and_row_counts_are_validated(self):
        for relative, shape, expected in (
            ('Low_Level_Features/Train_Normalized_a.dat', (3, 633), '634'),
            ('NUS_WID_Tags/Train_Tags1k.dat', (3, 999), '1000'),
            ('Low_Level_Features/Train_Normalized_a.dat', (4, 634), 'rows'),
            ('NUS_WID_Tags/Train_Tags1k.dat', (4, 1000), 'rows'),
            ('Groundtruth/TrainTestLabels/Labels_b_Train.txt', (4, 1), 'rows'),
        ):
            with self.subTest(relative=relative, shape=shape), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.prepare(root)
                np.savetxt(root / relative, np.ones(shape), delimiter='\t' if 'Tags' in relative else ' ')
                with self.assertRaisesRegex(ValueError, expected):
                    get_labeled_data(root, ['a', 'b'], None)

    def test_filtering_and_sample_limit_keep_labels_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            image, text, labels = get_labeled_data(root, ['a', 'b'], 2)
            self.assertEqual(image.shape, (2, 634))
            self.assertEqual(text.shape, (2, 1000))
            np.testing.assert_array_equal(labels, [0, 1])
