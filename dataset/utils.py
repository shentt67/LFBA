import os

import pandas as pd
import numpy as np

from utils.utils import raise_dataset_exception, raise_split_exception


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        print("Loading {}.".format(file))
        df = pd.read_csv(file, header=None, engine="c")
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
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    print("Loading {}.".format(file))
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t", engine="c")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
        selected.values[:n_samples])


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
