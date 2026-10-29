"""Shared sample identities, held-out validation and deferred attack application."""
import copy
import hashlib

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.dataset import CIFAR10_VFL, PHISHING_VFL, UCIHAR_VFL, NUSWIDE_VFL


def split_training_ids(labels, validation_rate, seed, anchor_idx=None, candidates=None):
    labels=np.asarray(labels)
    candidates=np.arange(len(labels)) if candidates is None else np.asarray(candidates)
    train,val=train_test_split(candidates,test_size=validation_rate,random_state=seed,stratify=labels[candidates])
    if anchor_idx is not None:
        if anchor_idx not in candidates:
            raise ValueError('anchor_idx is not in the training candidate set')
        if anchor_idx in val:
            replacements=sorted(int(i) for i in train if labels[i]==labels[anchor_idx])
            if not replacements:
                raise ValueError('Cannot keep anchor in training while preserving stratification')
            replacement=replacements[0]
            train[train==replacement]=anchor_idx
            val[val==anchor_idx]=replacement
    return sorted(train.tolist()),sorted(val.tolist())


def _bounded_ids(labels, limit, seed, anchor=None):
    ids=np.arange(len(labels))
    if limit is None or limit >= len(ids):
        return ids
    if limit < 2:
        raise ValueError('Sample limit is too small')
    selected,_=train_test_split(ids,train_size=limit,random_state=seed,stratify=np.asarray(labels))
    if anchor is not None and anchor not in selected:
        matches=sorted(int(i) for i in selected if labels[i]==labels[anchor])
        if not matches:
            raise ValueError('Bounded candidate set cannot retain anchor class')
        selected[selected==matches[0]]=anchor
    return np.sort(selected)


class DataView(Dataset):
    def __init__(self, base, ids, offset=0):
        self.base=base
        self.ids=[int(i) for i in ids]
        self.offset=offset
        self.attack_plan={}
        self.label_overrides={}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        original=self.ids[index]
        x,clean,y,_=self.base[original-self.offset]
        return x,clean,y,original

    @property
    def data(self): return self.base.data

    @property
    def data_p(self): return self.base.data_p

    @property
    def targets(self): return self.base.targets

    def set_attack(self, plan, args, dimensions, labels=None):
        self.attack_plan={int(i):None if j is None else int(j) for i,j in plan.items()}
        self.label_overrides={int(i):int(y) for i,y in (labels or {}).items()}
        if not set(self.attack_plan).issubset(self.ids):
            raise ValueError('Attack recipients must belong to this split')
        if any(j is not None and j not in self.ids for j in self.attack_plan.values()):
            raise ValueError('Attack donors must belong to the same split')
        local={i-self.offset:None if j is None else j-self.offset for i,j in self.attack_plan.items()}
        local_labels={i-self.offset:y for i,y in self.label_overrides.items()}
        if isinstance(self.base,CIFAR10_VFL):
            self.base.set_attack_plan(local,local_labels)
            return
        self.base.data=copy.deepcopy(self.base.data_p)
        clean_labels=getattr(self.base,'clean_targets',self.base.targets)
        self.base.clean_targets=copy.deepcopy(clean_labels)
        self.base.targets=copy.deepcopy(clean_labels)
        boundary={'PHISHING':15,'UCIHAR':281,'NUSWIDE':634}[args.dataset]
        for i,j in local.items():
            if j is not None:
                self.base.data[i,boundary:]=self.base.data_p[j,boundary:]
            value=args.trigger_feature_clip if args.dataset=='UCIHAR' else 1.
            self.base.data[i,dimensions]=value
        for i,y in local_labels.items():
            self.base.targets[i]=y


def fingerprint(base):
    digest=hashlib.sha256()
    data=np.asarray(base.data_p)
    for start in range(0,len(data),1024):
        digest.update(data[start:start+1024].tobytes())
    digest.update(np.asarray(base.targets).tobytes())
    return digest.hexdigest()


def build_datasets(args,state=None):
    anchor=args.anchor_idx if args.attack=='LFBA' else None
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
        (.4914,.4822,.4465),(.2023,.1994,.2010))])
    phishing_state=None
    if args.dataset=='CIFAR10':
        base=CIFAR10_VFL(args.data_dir,True,transform,download=False)
        official=CIFAR10_VFL(args.data_dir,False,transform,download=False)
    elif args.dataset=='PHISHING':
        if args.eval_limit is not None:
            raise ValueError('PHISHING eval_limit is unsupported; evaluate the fixed held-out split')
        prepared=PHISHING_VFL.prepare(args.data_dir,args.seed,args.validation_rate,
                                    state.get('phishing') if state else None,anchor_idx=anchor)
        base=PHISHING_VFL(args.data_dir,True,prepared=prepared)
        base.ids=list(range(len(base.data)))
        official=base
        phishing_state=prepared['state']
    elif args.dataset=='UCIHAR':
        base=UCIHAR_VFL(args.data_dir,True,None)
        official=UCIHAR_VFL(args.data_dir,False,None)
    else:
        labels=['buildings','grass','animal','water','person']
        base=NUSWIDE_VFL(args.data_dir,labels,True,None)
        official=NUSWIDE_VFL(args.data_dir,labels,False,None)
    labels=np.asarray(base.targets)
    if anchor is not None and not 0 <= anchor < len(labels):
        raise ValueError('anchor_idx is outside the dataset')
    hashes={'source':fingerprint(base),'official_test':fingerprint(official)}
    if state:
        if state['fingerprints']!=hashes:
            raise ValueError('Dataset fingerprint differs from saved experiment')
        train_ids,val_ids,test_ids=(state[k] for k in ('train_ids','val_ids','test_ids'))
    elif phishing_state:
        candidates=sorted(phishing_state['train_ids']+phishing_state['val_ids'])
        candidates=_bounded_ids(labels,args.train_limit,args.seed,anchor) if args.train_limit else candidates
        # Keep PHISHING preprocessing and split identities identical.
        if args.train_limit or (anchor is not None and anchor not in phishing_state['train_ids']):
            raise ValueError('PHISHING limits / anchor outside prepared training split are unsupported')
        train_ids,val_ids,test_ids=(phishing_state[k] for k in ('train_ids','val_ids','test_ids'))
    else:
        candidates=_bounded_ids(labels,args.train_limit,args.seed,anchor)
        train_ids,val_ids=split_training_ids(labels,args.validation_rate,args.seed,anchor,candidates)
        raw_test=_bounded_ids(np.asarray(official.targets),args.eval_limit,args.seed)
        test_ids=(raw_test+len(base.data)).tolist()
    if set(train_ids)&set(val_ids) or set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids):
        raise ValueError('Overlapping experiment splits')
    val_base=copy.copy(base)
    if args.dataset=='CIFAR10':
        val_base.train=False
    test_base=copy.copy(official)
    offset=0 if phishing_state else len(base.data)
    views=(DataView(base,train_ids),DataView(val_base,val_ids),DataView(test_base,test_ids,offset))
    state=dict(train_ids=list(train_ids),val_ids=list(val_ids),test_ids=list(test_ids),
               fingerprints=hashes,phishing=phishing_state)
    return (*views,state)
