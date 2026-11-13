"""Explicit, locked and recoverable seed-list reduction; never deletes trial data."""
from contextlib import ExitStack
import copy
from lfba_comparison.common import read_json,write_json,batch_lock
from lfba_comparison.stability_config import case_options
from lfba_comparison.runner import frozen_protocol


def reduce_seeds(root,options,target):
    """Caller must hold the parent batch lock. Journal precedes every protocol edit."""
    journal_path=root/'seed_reduction.json'
    journal=read_json(journal_path)
    current=read_json(root/'protocol.json')
    if journal:
        original=journal['original_protocol']
        if journal['target_protocol']!=target:
            raise ValueError('Existing seed reduction targets a different protocol')
    else:
        original=current
    if not original or dict(original,seeds=target['seeds'])!=target:
        raise ValueError('Seed reduction may only change seeds; all other settings must match')
    old_seeds=original['seeds'];new_seeds=target['seeds']
    if not new_seeds or len(new_seeds)>=len(old_seeds) or [s for s in old_seeds if s in new_seeds]!=new_seeds:
        raise ValueError('Seed reduction requires a nonempty proper ordered subset')
    if current not in (original,target):
        raise ValueError('Parent protocol conflicts with seed reduction')
    changes=[]
    with ExitStack() as locks:
        for case in target['cases']:
            child=case_options(options,case)
            if child.output.resolve().parent!=root.resolve():
                raise ValueError('Case must remain directly inside the batch')
            if not child.output.exists():continue
            locks.enter_context(batch_lock(child.output))
            path=child.output/'protocol.json';saved=read_json(path)
            if saved is None:
                if any(child.output.iterdir()):raise ValueError('Occupied case has no protocol')
                continue
            before=copy.copy(child);before.seeds=old_seeds
            old,new=frozen_protocol(before),frozen_protocol(child)
            if saved not in (old,new):raise ValueError('Child protocol conflicts with seed reduction')
            changes.append((path,old,new))
        journal=dict(original_protocol=original,target_protocol=target,
            removed_seeds=[s for s in old_seeds if s not in new_seeds],
            child_backups={str(path.relative_to(root)):old for path,old,new in changes},
            complete=False,note='Removed seeds remain on disk and are excluded from active summaries')
        write_json(journal_path,journal)
        for path,old,new in changes:write_json(path,new)
        write_json(root/'protocol.json',target)
        if read_json(root/'protocol.json')!=target or any(read_json(path)!=new for path,old,new in changes):
            raise ValueError('Reduced protocol read-back failed; retry with --reduce-seeds')
        journal['complete']=True;write_json(journal_path,journal)
    print('SEEDS_REDUCED',old_seeds,'->',new_seeds,'(all trial files preserved)',flush=True)
