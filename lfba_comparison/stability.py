"""Explicit grid execution; every trial runs the automatic strategy before LFBA."""
from pathlib import Path
from types import SimpleNamespace

from lfba_comparison.common import batch_lock,read_json,write_json
from lfba_comparison.runner import run_experiment,cleanup_completed_batch
from lfba_comparison.stability_config import stability_protocol,case_options,validate_budget,train_size
from lfba_comparison.stability_report import write_stability_report


def _read_protocol(root):
    protocol=read_json(root/'protocol.json')
    if not protocol or protocol.get('kind')!='stability':
        raise ValueError('Missing stability protocol; report-only never starts trials')
    try:
        options=SimpleNamespace(**{key:protocol[key] for key in
            ('dataset','seeds','smoke','poison_rates','select_rates','data_dir','device','block_size')})
        if (options.dataset!='CIFAR10' or not options.seeds or
                len(set(options.seeds))!=len(options.seeds) or
                any(type(seed) is not int or not 0<=seed<=2**32-1 for seed in options.seeds)):
            raise ValueError('Invalid stability dataset or seeds')
        options.anchor_source=protocol.get('anchor_source','auto_vote')
        expected=stability_protocol(options)
        for case in expected['cases']:
            validate_budget(train_size(options),case['poison_rate'],case['select_rate'],expected['epochs'])
        if protocol!=expected:raise ValueError('Saved stability matrix or protocol was changed')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('Invalid saved stability protocol') from error
    return protocol


def _finish(root,blocking_errors=()):
    _read_protocol(root)
    summary=write_stability_report(root)
    if blocking_errors:
        summary['all_complete']=False
        summary['blocking_case_errors']=list(blocking_errors)
        write_json(root/'summary.json',summary)
        with (root/'results.md').open('a',encoding='utf-8') as stream:
            stream.write('\n本轮存在配置级异常，不能视为完整成功；恢复状态予以保留。\n')
            for failure in blocking_errors:
                stream.write(f"- {failure['case_id']}: {failure['error']}\n")
    if summary['all_complete']:
        if read_json(root/'summary.json')!=summary:
            raise ValueError('Stability summary read-back differs')
        protocol=read_json(root/'protocol.json')
        folders=[root]+[root/case['id'] for case in protocol['cases']]
        for folder in folders:
            names=('results.md','all_epoch_metrics.csv',
                   'stability_parameters.png' if folder==root else 'validation_curves_lfba_vs_vote.png')
            if any(not (folder/name).is_file() or (folder/name).stat().st_size==0 for name in names):
                raise ValueError('Missing or empty stability report artifact')
        # No temporary model is removed until all case and global plots succeed.
        for case in protocol['cases']:
            cleanup_completed_batch(root/case['id'])
    return summary


def run_stability(options):
    root=Path(options.output);protocol=stability_protocol(options)
    occupied=root.exists() and any(root.iterdir())
    if occupied and not options.resume:
        raise FileExistsError('Stability output occupied; use a new output or --resume')
    if getattr(options,'reduce_seeds',False) and not occupied:
        raise ValueError('--reduce-seeds requires an existing stability batch')
    case_errors=[]
    with batch_lock(root):
        if occupied:
            if getattr(options,'reduce_seeds',False):
                from lfba_comparison.seed_reduction import reduce_seeds
                reduce_seeds(root,options,protocol)
            if read_json(root/'protocol.json')!=protocol:
                raise ValueError('Resume requires the identical full stability matrix and protocol')
        else:
            write_json(root/'protocol.json',protocol)
        for case in protocol['cases']:
            child=case_options(options,case)
            try:
                result=run_experiment(child)
                case_errors.extend(dict(case_id=case['id'],**failure) for failure in result['failures'])
            except Exception as error:
                case_errors.append(dict(case_id=case['id'],seed=None,stage='case',
                                        error=f'{type(error).__name__}: {error}'))
        summary=_finish(root,[error for error in case_errors if error['stage']=='case'])
        summary['attempt_errors']=case_errors
        write_json(root/'summary.json',summary)
    return 0 if summary['all_complete'] else 1


def report_stability(output):
    root=Path(output)
    if not (root/'protocol.json').is_file():
        raise ValueError('Missing stability protocol; report-only never starts trials')
    with batch_lock(root):
        _read_protocol(root)
        summary=_finish(root)
    return 0 if summary['all_complete'] else 1
