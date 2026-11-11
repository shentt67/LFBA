"""Label-only NUS-WIDE preflight; feature tables are read by the normal loader."""
import math
from pathlib import Path

import numpy as np

LABELS = ['buildings', 'grass', 'animal', 'water', 'person']
POISON_DIMENSIONS = 10


def dataset_profile(data_dir, smoke=False):
    from dataset.utils import validate_nuswide_files

    root = Path(data_dir) / 'NUS_WIDE'
    validate_nuswide_files(root, LABELS)
    counts = {}
    class_counts = {}
    for split in ('Train', 'Test'):
        columns = []
        for label in LABELS:
            path = root / 'Groundtruth' / 'TrainTestLabels' / f'Labels_{label}_{split}.txt'
            values = np.loadtxt(path, ndmin=2)
            if values.shape[1] != 1 or not np.isin(values, [0, 1]).all():
                raise ValueError(f'NUS-WIDE {path}: expected one binary label column')
            if columns and len(values) != len(columns[0]):
                raise ValueError(f'NUS-WIDE {split}: label rows differ')
            columns.append(values[:, 0])
        labels = np.stack(columns, axis=1)
        selected = labels[labels.sum(axis=1) == 1]
        counts[split] = len(selected)
        class_counts[split] = selected.sum(axis=0).astype(int).tolist()
        if not len(selected):
            raise ValueError(f'NUS-WIDE {split}: no single-label samples')
    source_size = min(500, counts['Train']) if smoke else counts['Train']
    validation_size = math.ceil(source_size * .1)
    return dict(selected_labels=list(LABELS), source_train_size=counts['Train'],
                source_test_size=counts['Test'], source_class_counts=class_counts,
                train_size=source_size-validation_size, validation_size=validation_size,
                test_size=min(100, counts['Test']) if smoke else counts['Test'],
                feature_split=[634, 1000], poison_dimensions=POISON_DIMENSIONS,
                trigger_value=1.0)
