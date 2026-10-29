"""Command-line configuration and supported experiment boundaries."""
import argparse
from datetime import datetime
import math
import os
import sys
import warnings


def attack_start_epoch(args):
    return max(1, args.pretrain_stage)


def validate_args(args):
    if args.dataset == 'NUSWIDET':
        args.dataset = 'NUSWIDE'
    if args.dataset not in ('CIFAR10','PHISHING','UCIHAR','NUSWIDE'):
        raise ValueError('Unsupported dataset: ' + args.dataset)
    if args.client_num != 2 or args.attack_client_num != 1:
        raise ValueError('Only two clients with attacker index 1 are supported')
    if args.select_replace and args.poison_all:
        raise ValueError('select_replace conflicts with poison_all')
    if args.select_replace:
        warnings.warn('--select_replace is deprecated: replacement is already the default', FutureWarning)
    for name in ('poison_rate','select_rate'):
        value = getattr(args,name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + ' must be finite and within [0,1]')
    if not math.isfinite(args.validation_rate) or not 0 < args.validation_rate < 1:
        raise ValueError('validation_rate must be within (0,1)')
    if args.epoch < 1 or args.batch_size < 1 or args.pretrain_stage < 0 or args.early_stop < 0 or args.print_steps < 1:
        raise ValueError('Invalid epoch, batch size, pretrain, early stop or print interval')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.trigger_feature_clip):
        raise ValueError('Learning rate and trigger value must be finite; lr must be positive')
    if args.poison_dimensions < 1:
        raise ValueError('poison_dimensions must be positive')
    if args.pretrained_checkpoint and args.test_checkpoint:
        raise ValueError('Choose either resume or evaluation checkpoint')
    if args.feature_extractor or args.trigger_visualization:
        raise ValueError('Feature extractor / trigger visualization is not implemented')
    return args


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description='Two-party LFBA experiments with isolated validation')
    defaults=dict(data_dir='./data/', dataset='CIFAR10', results_dir=None,
                  pretrained_checkpoint=None, test_checkpoint=None, feature_extractor='')
    for key,value in defaults.items():
        parser.add_argument('--'+key,default=value)
    for key,value in dict(device=0,seed=100,epoch=100,batch_size=256,client_num=2,start_epoch=0,
                          print_steps=10,early_stop=20,pretrain_stage=0,target_label=3,
                          poison_dimensions=5,attack_client_num=1,num_clusters=10,anchor_idx=33930,
                          train_limit=None,eval_limit=None).items():
        parser.add_argument('--'+key,default=value,type=int)
    for key,value in dict(lr=.001,poison_rate=.1,select_rate=1.,trigger_feature_clip=1.,validation_rate=.1).items():
        parser.add_argument('--'+key,default=value,type=float)
    for key in ('random_select','select_replace','poison_all','trigger_visualization'):
        parser.add_argument('--'+key,action='store_true')
    parser.add_argument('--attack',choices=('rsa','lra','LFBA'),default=None)
    parser.add_argument('--similarity',choices=('legacy_l1','cosine'),default='legacy_l1')
    parser.add_argument('--trigger_dimensions',nargs='+',type=int,default=None)
    tokens=list(sys.argv[1:] if argv is None else argv)
    args=parser.parse_args(tokens)
    args._provided=sorted({token[2:].split('=')[0] for token in tokens if token.startswith('--')})
    # Checkpoint configuration is merged before validation by the experiment entry point.
    if not args.pretrained_checkpoint and not args.test_checkpoint:
        validate_args(args)
    if args.results_dir is None:
        args.results_dir=os.path.join('results',args.dataset,datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return args
