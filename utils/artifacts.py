"""Compact durable experiment records and temporary recovery checkpoints."""
import csv
import json
from pathlib import Path

from lfba_comparison.common import read_json, write_json


def runtime_checkpoint(output):
    return Path(output) / '.runtime' / 'last_checkpoint.pth.tar'


def experiment_record(trainer):
    choice = getattr(trainer, 'choice', None)
    if choice is None and trainer.poison_indexes is not None:
        choice = dict(anchor_id=trainer.args.anchor_idx, candidates=trainer.poison_indexes)
    data = dict(version=1, history=trainer.history, choice=choice,
                plans=getattr(trainer, 'plans', []), audit=getattr(trainer, 'audit', []))
    return data


def save_experiment_data(output, trainer):
    data = experiment_record(trainer)
    write_json(Path(output) / 'experiment_data.json', data)
    return data


def load_experiment_data(output):
    output = Path(output)
    data = read_json(output / 'experiment_data.json')
    if data is not None:
        return data
    audit = []
    if (output / 'epoch_metrics.csv').exists():
        with (output / 'epoch_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            audit = list(csv.DictReader(stream))
    return dict(version=1, history=read_json(output / 'history.json') or [],
                choice=read_json(output / 'choice.json'),
                plans=read_json(output / 'replacement_ids.json') or [], audit=audit)


CHECKPOINT_NAMES = ('last_checkpoint.pth.tar', 'best_checkpoint.pth.tar', 'check_point.tar')


def runtime_cleanup_paths(output):
    """Validate the complete allowlist before deleting any file; never follow links."""
    output = Path(output).resolve()
    directories = (output, output / '.runtime', output / 'runtime')
    paths = []
    for directory in directories:
        if directory.is_symlink() or directory.resolve() != directory:
            raise ValueError('Runtime directory must stay inside the experiment output')
        if directory.exists() and not directory.is_dir():
            raise ValueError('Runtime path must be a directory')
        for name in CHECKPOINT_NAMES:
            for filename in (name, name+'.tmp', name+'.copying'):
                path = directory / filename
                if path.is_symlink() or path.resolve() != path:
                    raise ValueError('Runtime checkpoint must not be a link')
                if path.exists():
                    if not path.is_file():
                        raise ValueError('Runtime checkpoint must be a file')
                    paths.append(path)
    return paths


def cleanup_runtime(output):
    output = Path(output).resolve()
    for path in runtime_cleanup_paths(output):
        path.unlink()
    for directory in (output / '.runtime', output / 'runtime'):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def save_configuration(output, trainer):
    config = dict(vars(trainer.args), data_state=trainer.data_state,
                  trigger_dimensions=trainer.trigger_dimensions)
    write_json(Path(output) / 'config.json', config)


def verify_saved_result(output, expected, trainer=None):
    if read_json(Path(output) / 'metrics.json') != expected:
        raise ValueError('Saved metrics differ from evaluated result')
    data = read_json(Path(output) / 'experiment_data.json')
    if not isinstance(data, dict) or not isinstance(data.get('history'), list):
        raise ValueError('Missing durable experiment data')
    if trainer is not None:
        normalize=lambda value:json.loads(json.dumps(value,allow_nan=False))
        if data!=normalize(experiment_record(trainer)):
            raise ValueError('Saved experiment data differs from training records')
        config=dict(vars(trainer.args),data_state=trainer.data_state,
                    trigger_dimensions=trainer.trigger_dimensions)
        if read_json(Path(output)/'config.json')!=normalize(config):
            raise ValueError('Saved configuration differs from training configuration')
