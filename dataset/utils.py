import os
from pathlib import Path

import pandas as pd
import numpy as np

from utils.utils import raise_dataset_exception, raise_split_exception


def validate_nuswide_files(data_dir, selected_labels, splits=('Train', 'Test')):
    """Check the manifest before reading any potentially large feature tables."""
    root = Path(data_dir).resolve()
    missing = []
    for split in splits:
        for label in selected_labels:
            path = root / 'Groundtruth' / 'TrainTestLabels' / f'Labels_{label}_{split}.txt'
            if not path.is_file():
                missing.append(str(path))
        features = root / 'Low_Level_Features'
        if not features.is_dir() or not any(
            p.is_file() and p.name.startswith(f'{split}_Normalized') for p in features.iterdir()
        ):
            missing.append(str(features / f'{split}_Normalized*.dat'))
        tags = root / 'NUS_WID_Tags' / f'{split}_Tags1k.dat'
        if not tags.is_file():
            missing.append(str(tags))
    if missing:
        raise FileNotFoundError(
            f'NUS-WIDE 数据不完整，实际目录: {root}\n缺失文件:\n  '
            + '\n  '.join(missing)
            + '\n请下载并解压标签、低层图像特征和文本特征。'
            '\n--data_dir 应指向 NUS_WIDE 的父目录；参见 README.md 的 NUS-WIDE 数据准备。'
        )


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    validate_nuswide_files(data_dir, selected_label, (dtype,))
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        print("Loading {}.".format(file))
        df = pd.read_csv(file, header=None, engine="c")
        if dfs and len(df) != len(dfs[0]):
            raise ValueError(f'NUS-WIDE {file}: label rows differ from other labels')
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # get XA, which are image low level features
    features_path = "Low_Level_Features"
    dfs = []
    for file in sorted(os.listdir(os.path.join(data_dir, features_path))):
        if file.startswith("_".join([dtype, "Normalized"])):
            print("Loading {}.".format(os.path.join(data_dir, features_path, file)))
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ", engine="c")
            df.dropna(axis=1, inplace=True)
            if len(df) != len(data_labels):
                raise ValueError(f'NUS-WIDE {file}: feature rows differ from label rows')
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    if data_XA.shape[1] != 634:
        raise ValueError(f'NUS-WIDE {dtype}: expected 634 image features, got {data_XA.shape[1]}')
    data_X_image_selected = data_XA.loc[selected.index]
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    print("Loading {}.".format(file))
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t", engine="c")
    tagsdf.dropna(axis=1, inplace=True)
    if len(tagsdf) != len(data_labels):
        raise ValueError(f'NUS-WIDE {dtype}: tag rows differ from label rows')
    if tagsdf.shape[1] != 1000:
        raise ValueError(f'NUS-WIDE {dtype}: expected 1000 text features, got {tagsdf.shape[1]}')
    data_X_text_selected = tagsdf.loc[selected.index]
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
        selected.values[:n_samples], axis=1)


def split_vfl(data, args):
    if args.dataset == 'CIFAR10':
        # 32*16*3/32*16*3
        x_a = data[:, :, :, :16]
        x_b = data[:, :, :, 16:]
        return x_a, x_b
    elif args.dataset == 'UCIHAR':
        # 281/280
        x_a = data[:, :281]
        x_b = data[:, 281:]
        return x_a, x_b
    elif args.dataset == 'PHISHING':
        # 281/280
        x_a = data[:, :15]
        x_b = data[:, 15:]
        return x_a, x_b
    elif args.dataset in ('NUSWIDE', 'NUSWIDET'):
        # 634/1000
        x_a = data[:, :634]
        x_b = data[:, 634:]
        return x_a, x_b
    elif args.dataset == 'NUSWIDEI':
        raise ValueError('NUSWIDEI is unsupported; use NUSWIDE (image/text order)')
    else:
        raise_dataset_exception()
