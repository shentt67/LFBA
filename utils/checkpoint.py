"""Versioned, weights-only-loadable experiment snapshots."""
import copy
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

VERSION=2


def cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def capture_rng(loaders):
    state=np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loaders={name:loader.generator.get_state() for name,loader in loaders.items()
                         if loader is not None and getattr(loader,'generator',None) is not None})


def restore_rng(state,loaders):
    random.setstate(state['python'])
    name,keys,pos,has_gauss,cached=state['numpy']
    np.random.set_state((name,np.asarray(keys,dtype=np.uint32),pos,has_gauss,cached))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    for name,gen_state in state['loaders'].items():
        if name in loaders and loaders[name] is not None and loaders[name].generator is not None:
            loaders[name].generator.set_state(gen_state)


def atomic_save(state,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary)
    # Windows scanners can briefly hold a newly written large checkpoint open.
    # Preserve the previous file and retry the atomic rename, never truncate it.
    for attempt in range(8):
        try:
            os.replace(temporary,path)
            break
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==7:
                raise
            time.sleep(min(.1 * 2**attempt,1.))


def load_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError('Checkpoint not found: '+str(path))
    return torch.load(path,map_location='cpu',weights_only=True)
