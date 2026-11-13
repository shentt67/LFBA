"""Conservatively migrate experiment artifacts; defaults to a read-only inventory."""
import argparse
from contextlib import contextmanager
import csv
import json
import os
import stat
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lfba_comparison.common import batch_lock, read_json, write_json

OUTPUT_ROOTS = ('results', 'scripts/results', '.worktrees/lfba-vs-auto-vote/results',
                '.worktrees/auto-neighbor-vote-feasibility/results')
MERGED = ('history.json', 'choice.json', 'replacement_ids.json', 'epoch_metrics.csv',
          'independent_evaluation.json')
OBSOLETE = ('source_hashes.json', 'source_snapshot.zip', 'environment.json', 'selection.pt')


def safe_path(path, workspace):
    workspace = Path(workspace).resolve()
    path = Path(path).absolute()
    if not path.is_relative_to(workspace):
        raise ValueError('Path outside workspace: ' + str(path))
    for part in (path, *path.parents):
        if part == workspace:
            break
        attributes = getattr(part.lstat(), 'st_file_attributes', 0) if part.exists() else 0
        if (part.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)
                or (hasattr(part, 'is_junction') and part.is_junction())):
            raise ValueError('Linked path refused: ' + str(part))
    if not path.resolve().is_relative_to(workspace):
        raise ValueError('Resolved path outside workspace: ' + str(path))
    return path


def files_under(path, workspace):
    safe_path(path, workspace)
    for current, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            safe_path(Path(current) / name, workspace)
        for name in files:
            yield Path(current) / name


@contextmanager
def old_lock(batch):
    path = batch / 'batch.lock'
    if not path.exists():
        yield
        return
    with path.open('r+b') as stream:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def same_file_contents(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            chunk = a.read(1024 * 1024)
            if chunk != b.read(1024 * 1024):
                return False
            if not chunk:
                return True


def persist(log):
    """Atomic write_json fsyncs the journal before replacing its prior version."""
    if log.get('log_path'):
        write_json(Path(log['log_path']), log)


def delete_file(path, workspace, log, reason):
    safe_path(path, workspace)
    if not path.exists():
        return
    record = dict(path=str(path.relative_to(workspace)), bytes=path.stat().st_size, reason=reason)
    log['pending_deletion'] = record
    persist(log)  # No unlink is allowed before its intent has reached disk.
    path.unlink()
    log['deleted'].append(record)
    log['freed_bytes'] = sum(row['bytes'] for row in log['deleted'])
    log['pending_deletion'] = None
    persist(log)


def migrate_write(path, value, workspace, log):
    log['pending_write'] = str(path.relative_to(workspace))
    persist(log)
    write_json(path, value)
    if read_json(path) != value:
        raise ValueError('Migration readback failed: ' + str(path))
    log.setdefault('migrated', []).append(log['pending_write'])
    log['pending_write'] = None
    persist(log)


def success(folder, protocol):
    metrics = read_json(folder / 'metrics.json')
    if not metrics:
        return False
    data = read_json(folder / 'experiment_data.json') or {}
    history = data.get('history') or read_json(folder / 'history.json') or []
    proof = metrics.get('independent_evaluation') or read_json(folder / 'independent_evaluation.json')
    epoch = metrics.get('epoch', metrics.get('checkpoint_epoch'))
    if not history or epoch is None:
        return False
    if protocol and 'methods' in protocol:
        expected = protocol['epochs']
        audit = data.get('audit') or read_csv(folder / 'epoch_metrics.csv')
        return (proof is not None and metrics.get('test') == proof.get('test')
                and metrics.get('final_model_digest') == proof.get('final_model_digest')
                and epoch == expected and len(history) == expected and len(audit) == expected
                and [int(row['epoch']) for row in audit] == list(range(1, expected + 1)))
    return (any(row.get('epoch') == epoch for row in history)
            and metrics.get('validation') == next(
                (row.get('validation') for row in history if row.get('epoch') == epoch), None))


def compact_batch(batch, workspace, log):
    """Caller owns both lock formats. All files are checked before any mutation."""
    original_files = list(files_under(batch, workspace))
    protocol = read_json(batch / 'protocol.json') or {}
    folders = sorted({p.parent for p in original_files if p.name == 'metrics.json'})
    successful = set()

    def remove(path, reason):
        delete_file(path, workspace, log, reason)

    for folder in folders:
        # Conflicts are errors even when a conflicting copy would otherwise make
        # the completion check fail. Never silently prefer one source of truth.
        existing_data = read_json(folder / 'experiment_data.json') or {}
        for key, filename in (('history', 'history.json'), ('choice', 'choice.json'),
                              ('plans', 'replacement_ids.json')):
            if (folder / filename).exists() and key in existing_data and read_json(folder / filename) != existing_data[key]:
                raise ValueError('Conflicting old/new data: ' + str(folder / filename))
        if ((folder / 'epoch_metrics.csv').exists() and 'audit' in existing_data
                and read_csv(folder / 'epoch_metrics.csv') != existing_data['audit']):
            raise ValueError('Conflicting old/new audit: ' + str(folder))
        existing_metrics = read_json(folder / 'metrics.json') or {}
        stored_proof = existing_metrics.get('independent_evaluation')
        separate_proof = read_json(folder / 'independent_evaluation.json')
        if stored_proof is not None and separate_proof is not None and stored_proof != separate_proof:
            raise ValueError('Conflicting old/new evaluation proof: ' + str(folder))
        if not success(folder, protocol):
            log['preserved'].append(dict(path=str(folder.relative_to(workspace)), reason='completion not proven'))
            continue
        metrics = read_json(folder / 'metrics.json')
        data = read_json(folder / 'experiment_data.json') or dict(version=1)
        for key, filename, default in (('history', 'history.json', []), ('choice', 'choice.json', None),
                                       ('plans', 'replacement_ids.json', [])):
            legacy_path = folder / filename
            if legacy_path.exists() and key in data and read_json(legacy_path) != data[key]:
                raise ValueError('Conflicting old/new data: ' + str(legacy_path))
            if key not in data:
                data[key] = read_json(folder / filename)
                if data[key] is None:
                    data[key] = default
        if (folder / 'epoch_metrics.csv').exists() and 'audit' in data and read_csv(folder / 'epoch_metrics.csv') != data['audit']:
            raise ValueError('Conflicting old/new audit: ' + str(folder))
        if 'audit' not in data:
            data['audit'] = read_csv(folder / 'epoch_metrics.csv')
        config = read_json(folder / 'config.json')
        if config is None:
            log['preserved'].append(dict(path=str(folder), reason='missing configuration'))
            continue
        checkpoint_path = next((p for p in (folder / '.runtime/last_checkpoint.pth.tar',
                                             folder / 'last_checkpoint.pth.tar') if p.exists()), None)
        if checkpoint_path and ('data_state' not in config or data.get('choice') is None or not data.get('plans')):
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            config.setdefault('data_state', checkpoint.get('data_state', {}))
            config.setdefault('trigger_dimensions', checkpoint.get('trigger_dimensions', []))
            state = checkpoint.get('comparison', checkpoint.get('neighbor_vote', {}))
            if data.get('choice') is None:
                data['choice'] = state.get('choice')
                attack = checkpoint.get('attack_state', {})
                if data['choice'] is None and attack.get('poison_indexes') is not None:
                    data['choice'] = dict(anchor_id=checkpoint.get('config', {}).get('anchor_idx'),
                                          candidates=attack['poison_indexes'])
            if not data.get('plans'):
                restored_plans = (checkpoint.get('plans') or state.get('plans'))
                if restored_plans:
                    data['plans'] = restored_plans
                else:
                    final_plan = checkpoint.get('attack_state', {}).get('current_plan', {})
                    if final_plan:
                        data['plans'] = [dict(epoch=checkpoint['epoch'],
                                              recipients=list(final_plan), donors=list(final_plan.values()),
                                              record_scope='final_epoch_only')]
            del checkpoint
        embedded_proof = metrics.get('independent_evaluation')
        legacy_proof = read_json(folder / 'independent_evaluation.json')
        if embedded_proof is not None and legacy_proof is not None and embedded_proof != legacy_proof:
            raise ValueError('Conflicting old/new evaluation proof: ' + str(folder))
        proof = embedded_proof or legacy_proof
        if proof is not None:
            metrics['independent_evaluation'] = proof
        metrics['status'] = 'complete'
        for name, value in (('config.json', config), ('experiment_data.json', data), ('metrics.json', metrics)):
            migrate_write(folder / name, value, workspace, log)
        if not success(folder, protocol):
            raise ValueError('Migration completion verification failed: ' + str(folder))
        successful.add(folder)
        for name in MERGED:
            remove(folder / name, 'merged and verified')
        for name in ('best_checkpoint.pth.tar', 'last_checkpoint.pth.tar'):
            remove(folder / name, 'completed model')
            remove(folder / '.runtime' / name, 'completed temporary model')

    # Batch evidence is retained inside summary, not silently discarded.
    summary = read_json(batch / 'summary.json')
    if isinstance(summary, dict):
        metadata = summary.setdefault('artifact_metadata', {})
        for name in ('budget_check.csv', 'timing_resources.json'):
            path = batch / name
            if path.exists():
                metadata[name] = read_csv(path) if name.endswith('.csv') else read_json(path)
        migrate_write(batch / 'summary.json', summary, workspace, log)
        for name in metadata:
            if name in ('budget_check.csv', 'timing_resources.json'):
                remove(batch / name, 'merged into summary')

    expected = {batch / f'seed_{seed}' / method for seed in protocol.get('seeds', [])
                for method in protocol.get('methods', [])}
    batch_done = bool(expected) and expected <= successful and (batch / 'results.md').is_file() and any(batch.glob('*.png'))
    if batch_done:
        for clean in sorted({p.parent for p in original_files if p.parent.name in ('clean_first', 'clean')}):
            data = read_json(clean / 'experiment_data.json') or dict(version=1)
            for key, filename, default in (('history','history.json',[]), ('choice','choice.json',None),
                                           ('plans','replacement_ids.json',[])):
                path = clean / filename
                if path.exists():
                    legacy = read_json(path)
                    if key in data and data[key] != legacy:
                        raise ValueError('Conflicting clean-prefix data: ' + str(path))
                    data[key] = legacy
                else:
                    data.setdefault(key, default)
            audit_path = clean / 'epoch_metrics.csv'
            if audit_path.exists():
                audit = read_csv(audit_path)
                if 'audit' in data and data['audit'] != audit:
                    raise ValueError('Conflicting clean-prefix audit: ' + str(clean))
                data['audit'] = audit
            else:
                data.setdefault('audit', [])
            migrate_write(clean / 'experiment_data.json', data, workspace, log)
            for name in ('history.json','choice.json','replacement_ids.json','epoch_metrics.csv'):
                remove(clean / name, 'merged shared-prefix records')
    for path in original_files:
        if path.name in OBSOLETE:
            remove(path, 'obsolete intermediate or source metadata')
        if batch_done and path.parent.name in ('clean_first', 'clean') and path.name in ('best_checkpoint.pth.tar', 'last_checkpoint.pth.tar'):
            remove(path, 'completed shared prefix')


def _run(workspace=ROOT, apply=False, log_path=None, log=None):
    workspace = Path(workspace).resolve()
    if log is None:
        log = dict(apply=apply, deleted=[], preserved=[], errors=[], batches=[], freed_bytes=0)
    if apply:
        destination = safe_path(log_path or workspace / 'note' / ('artifact-cleanup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json'), workspace)
        log['log_path'] = str(destination)
        persist(log)
    roots = [safe_path(workspace / root, workspace) for root in OUTPUT_ROOTS]
    batches = []
    for root in roots:
        if root.exists():
            for child in sorted(root.iterdir()):
                safe_path(child, workspace)
                if child.is_dir():
                    batches.append(child)
    for batch in batches:
        try:
            files = list(files_under(batch, workspace))
            log['batches'].append(dict(path=str(batch.relative_to(workspace)), files=len(files),
                                      bytes=sum(p.stat().st_size for p in files)))
            if apply:
                with batch_lock(batch), old_lock(batch):
                    compact_batch(batch, workspace, log)
        except Exception as error:
            log['errors'].append(dict(path=str(batch.relative_to(workspace)), error=str(error)))
        finally:
            persist(log)
    # Remove only a proven byte-identical copy, including reports and every image.
    primary = workspace / 'results/lfba_auto_vote_formal'
    duplicate = workspace / '.worktrees/lfba-vs-auto-vote/results/lfba_auto_vote_formal'
    try:
        if apply and not log['errors'] and primary.exists() and duplicate.exists():
            with batch_lock(primary), batch_lock(duplicate):
                with old_lock(primary), old_lock(duplicate):
                    left = {str(p.relative_to(primary)): p for p in files_under(primary, workspace) if p != primary / 'batch.lock'}
                    right = {str(p.relative_to(duplicate)): p for p in files_under(duplicate, workspace) if p != duplicate / 'batch.lock'}
                    identical = left.keys() == right.keys() and all(same_file_contents(left[k], right[k]) for k in left)
                    if identical:
                        for path in right.values():
                            if path == duplicate / 'batch.lock':
                                continue
                            delete_file(path, workspace, log, 'identical duplicate')
                # Windows requires closing the legacy lock handle before deleting it.
                # Both system batch locks remain held until every directory is removed.
                if identical:
                    legacy_lock = duplicate / 'batch.lock'
                    if legacy_lock.exists():
                        delete_file(legacy_lock, workspace, log, 'identical duplicate')
                    for current, dirs, files in os.walk(duplicate, topdown=False):
                        Path(current).rmdir()
    except Exception as error:
        log['errors'].append(dict(path=str(duplicate.relative_to(workspace)), error=str(error)))
    finally:
        persist(log)
    log['freed_bytes'] = sum(row['bytes'] for row in log['deleted'])
    for batch in batches:
        if batch.exists():
            try:
                remaining = list(files_under(batch, workspace))
                log['preserved'].extend(dict(path=str(path.relative_to(workspace)), reason='retained experiment record or unrecognized file')
                                        for path in remaining)
                entry = next(row for row in log['batches'] if row['path'] == str(batch.relative_to(workspace)))
                entry.update(after_files=len(remaining), after_bytes=sum(p.stat().st_size for p in remaining))
            except (OSError, ValueError) as error:
                log['errors'].append(dict(path=str(batch.relative_to(workspace)), error=str(error)))
    persist(log)
    return log


def run(workspace=ROOT, apply=False, log_path=None):
    log = dict(apply=apply, deleted=[], preserved=[], errors=[], batches=[], freed_bytes=0)
    try:
        return _run(workspace, apply, log_path, log)
    except Exception as error:
        log['errors'].append(dict(path='run', error=str(error)))
        return log
    finally:
        persist(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--workspace', type=Path, default=ROOT)
    flags = parser.parse_args()
    result = run(flags.workspace, flags.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['errors'] else 0)
