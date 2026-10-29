import copy

import numpy as np
import torch
from attack.add_trigger import add_trigger_to_data
from attack.add_trigger_replace import add_trigger_to_data_replace

### LRA 攻击（Label Replacing Attack）
### 输入：原始数据、触发维度、目标标签、中毒比例、攻击模式
### 输出：中毒后的数据、中毒后的标签
### 先随机选择一些样本，将这些样本的标签替换为触发标签
### 最后返回中毒后的数据、中毒后的标签
def attack_lra(args, logger, data, trigger_dimensions, targets, rate, mode):
    ### 复制原始数据和标签
    new_data = copy.deepcopy(data)
    new_targets = copy.deepcopy(targets)
    ### 随机选择一些样本（相当于不放回随机抽样）
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    ### 将这些样本的标签替换为触发标签
    new_data, new_targets = add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, new_targets,
                                                rate, mode,
                                                replace_label=True)
    return new_data, new_targets

### RSA 攻击（Random Sample Attack）
### 输入：原始数据、触发维度、中毒比例、攻击模式
### 输出：中毒后的数据
### 先随机选择一些样本，将这些样本的标签替换为触发标签
### 最后返回中毒后的数据
def attack_rsa(args, logger, data, trigger_dimensions, rate, mode):
    new_data = copy.deepcopy(data)
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    ### 只返回中毒后的数据，所以函数的第二个返回值poison_targets用_接受但不返回
    new_data, _ = add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, [], rate, mode,
                                      replace_label=False)
    return new_data

#
def attack_LFBA(args, logger, replace_indexes_others, replace_indexes_target, train_indexes, poison_indexes, data,
               target, trigger_dimensions, rate,
               mode):

    if args.poison_all:
    ### 只加触发器不切换特征
        new_data, _ = add_trigger_to_data(args, logger, poison_indexes, data, trigger_dimensions, target, rate, mode,
                                          replace_label=False)
    else:
    ### 又加触发器又切换特征
        new_data, _ = add_trigger_to_data_replace(args, logger, replace_indexes_others, replace_indexes_target,
                                                  train_indexes, poison_indexes, data, trigger_dimensions, target, rate,
                                                  mode,
                                                  replace_label=False)
    return new_data

# 选择LFBA攻击的毒化样本
def select_LFBA(train_features, num_poisons):
    anchor_idx = get_anchor_LFBA(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index, anchor_idx

# 获取LFBA攻击的锚点样本索引
# 输入：训练特征、毒化样本数量
# 输出：锚点样本索引
def get_anchor_LFBA(train_features, num_poisons):
    consistency = train_features @ train_features.T
    w = torch.cat((torch.ones((num_poisons)),
                   -torch.ones((num_poisons))), dim=0)
    top_con = torch.topk(consistency, 2 * num_poisons, dim=1)[0]
    mean_top_con = torch.matmul(top_con, w)
    idx = torch.argmax(mean_top_con)
    return idx


def get_near_index(anchor_feature, train_features, num_poisons, mode='legacy_l1', sample_ids=None):
    """Return candidate positions; resolve equal scores using original sample IDs."""
    if mode not in ('legacy_l1', 'cosine'):
        raise ValueError('Unknown similarity mode')
    p = 1 if mode == 'legacy_l1' else 2
    anchor = anchor_feature.reshape(-1)
    anchor_norm = torch.linalg.vector_norm(anchor, ord=p)
    if not torch.isfinite(anchor).all() or not torch.isfinite(anchor_norm) or anchor_norm <= 0:
        raise ValueError('Invalid or zero anchor gradient')
    norms = torch.linalg.vector_norm(train_features, ord=p, dim=1)
    valid = torch.isfinite(train_features).all(dim=1) & torch.isfinite(norms) & (norms > 0)
    positions = torch.nonzero(valid).flatten()
    scores = (train_features[positions] @ anchor) / (norms[positions] * anchor_norm)
    finite = torch.isfinite(scores)
    positions, scores = positions[finite], scores[finite]
    if num_poisons < 1 or len(positions) < num_poisons:
        raise ValueError('Not enough valid gradients for the requested poison set')
    ids = torch.arange(len(train_features), device=positions.device) if sample_ids is None else torch.as_tensor(sample_ids, device=positions.device)
    by_id = torch.argsort(ids[positions], stable=True)
    order = by_id[torch.argsort(scores[by_id], descending=True, stable=True)]
    return positions[order[:num_poisons]]
