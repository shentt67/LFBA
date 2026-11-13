"""Independent four-arm orchestration, checkpoints and strict audits."""
import hashlib
import logging
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.experiment import build_datasets
from main import build_models, trigger_dimensions, triggered_view, validate_data
from utils.config import parse_args
from utils.utils import set_seed
from utils.checkpoint import load_checkpoint, atomic_save, cpu_tree
from utils.artifacts import runtime_checkpoint, save_experiment_data, save_configuration
from lfba_comparison.common import digest, write_json, read_json, batch_lock
from lfba_comparison.core_policy import METHODS, schedule, select_core, diagnostics
from lfba_comparison.core_trainer import CoreTrainer
from lfba_comparison.policy import purity
from lfba_comparison.completion import load_completion, finish_ablation

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = [ROOT/'main.py', ROOT/'scripts/run_core_ablation.py']
    for directory in ('attack', 'dataset', 'models', 'utils', 'lfba_comparison'):
        paths.extend(sorted((ROOT/directory).glob('*.py')))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def protocol(options):
    return dict(version=1, kind='core_ablation', dataset='CIFAR10', seeds=options.seeds,
                methods=list(METHODS), smoke=options.smoke, epochs=5 if options.smoke else 40,
                batch_size=128, optimizer='Adam', lr=.001, lr_decay=False, early_stop=False,
                poison_rate=.1, select_rate=.5, clean_rounds=3, attack_start=4,
                retained_split_anchor=23470, final_model='last', similarity='cosine',
                train_source_limit=500 if options.smoke else None,
                eval_limit=100 if options.smoke else None, device=options.device,
                block_size=options.block_size, data_dir=str(options.data_dir),
                source_hashes=source_hashes(), keep_checkpoints=getattr(options, 'keep_checkpoints', False))


def assemble(options, seed, method, output, checkpoint=None, choice=None):
    flags = ['--dataset', 'CIFAR10', '--data_dir', str(options.data_dir),
             '--results_dir', str(output), '--seed', str(seed), '--device', str(options.device),
             '--epoch', str(3 if method == 'capture' else (5 if options.smoke else 40)),
             '--batch_size', '128', '--lr', '.001', '--early_stop', '0', '--anchor_idx', '23470',
             '--attack', 'LFBA', '--poison_rate', '.1', '--select_rate', '.5',
             '--pretrain_stage', '3', '--print_steps', '200', '--target_label', '0',
             '--similarity', 'cosine']
    if options.smoke: flags += ['--train_limit', '500', '--eval_limit', '100']
    args = parse_args(flags)
    args.comparison_method = method
    args.block_size = options.block_size
    args.retain_anchor = True
    set_seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cpu' if options.device < 0 else f'cuda:{options.device}')
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(device)
    train, val, test, state = build_datasets(args, checkpoint['data_state'] if checkpoint else None)
    validate_data(args, train, val, test)
    expected = [450, 50, 100] if options.smoke else [45000, 5000, 10000]
    if [len(train), len(val), len(test)] != expected: raise ValueError('Unexpected dataset sizes')
    dims = trigger_dimensions(args, checkpoint)
    views = dict(train=train, val=val, val_asr=triggered_view(val, args, dims),
                 test=test, test_asr=triggered_view(test, args, dims))
    loaders = {name: DataLoader(view, batch_size=128, shuffle=name == 'train', num_workers=0,
                               generator=torch.Generator().manual_seed(seed+i+1))
               for i, (name, view) in enumerate(views.items())}
    models = [m.to(device) for m in build_models(args)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in models]
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('core.'+str(output)); logger.setLevel(logging.INFO); logger.propagate=False
    handler = logging.FileHandler(output/'experiment.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); logger.addHandler(handler)
    trainer = CoreTrainer(device, models, [], None, optimizers, nn.CrossEntropyLoss(),
                          loaders['train'], loaders['test'], loaders['test_asr'], dims, logger,
                          args, checkpoint, loaders['val'], loaders['val_asr'], state, choice=choice)
    return trainer, handler


def close(trainer, handler):
    trainer.logger.removeHandler(handler); handler.close()


def verify_prefix(prefix):
    state = prefix['comparison']
    if (prefix['epoch'] != 3 or state['method'] != 'capture' or state['choice'] is not None
            or state['plans'] or state['cumulative'] or prefix['attack_state']['current_plan']
            or prefix['attack_state']['label_overrides']
            or [o['epoch'] for o in state['observations']] != [1, 2, 3]
            or len(state['prefix']) != 3 or len(state['audit']) != 3):
        raise ValueError('Invalid three-round clean prefix')


def verify_final(final, prefix, choice, epochs):
    verify_prefix(prefix)
    state = final['comparison']
    count = int(len(prefix['data_state']['train_ids'])*.1)
    expected = schedule(epochs, count)
    expected_start = digest(dict(model=prefix['state_dict'], optimizer=prefix['optimizer'], rng=prefix['rng_state']))
    if (final['epoch'] != epochs or state['method'] != choice['method'] or state['choice'] != choice
            or state['start_state'] != expected_start or final['data_state'] != prefix['data_state']
            or state['prefix'] != prefix['comparison']['prefix']
            or digest(state['observations']) != digest(prefix['comparison']['observations'])
            or final['trigger_dimensions'] != prefix['trigger_dimensions']):
        raise ValueError('Final identity, prefix, observations or choice differ')
    candidates = choice['candidates']; pool = set(candidates); train = set(prefix['data_state']['train_ids'])
    if len(pool) != count or not pool <= train or choice['anchor_id'] in pool:
        raise ValueError('Invalid frozen candidates')
    if (state['cumulative'] != sum(expected) or len(state['audit']) != epochs
            or len(final['history']) != epochs or final['attack_state']['poison_indexes'] != candidates
            or final['attack_state']['label_overrides']): raise ValueError('Final budget differs')
    for i, row in enumerate(state['audit']):
        if (row['epoch'] != i+1 or row['recipient_count'] != expected[i]
                or row['cumulative_replacements'] != sum(expected[:i+1])
                or row['candidate_count'] != (0 if i < 3 else count)
                or (i >= 3 and row['candidate_digest'] != digest(candidates))):
            raise ValueError('Per-round audit differs')
    if [p['epoch'] for p in state['plans']] != list(range(4, epochs+1)):
        raise ValueError('Attack epochs differ')
    for plan in state['plans']:
        recipients, donors = plan['recipients'], plan['donors']
        q = expected[plan['epoch']-1]
        if (len(recipients) != q or len(set(recipients)) != q or not set(recipients) <= pool
                or len(donors) != q or len(set(donors)) != q or not set(donors) <= train-pool):
            raise ValueError('Invalid replacement plan')


def run_branch(options, seed, method, folder, prefix, choice):
    output = folder/method
    path = runtime_checkpoint(output)
    checkpoint = load_checkpoint(path) if path.exists() else prefix
    trainer, handler = assemble(options, seed, method, output, checkpoint, choice)
    try:
        trainer.train()
        final = load_checkpoint(path)
        verify_final(final, prefix, choice, trainer.args.epoch)
        metrics = trainer.test(trainer.args.epoch-1, 'test')
        save_configuration(output, trainer); save_experiment_data(output, trainer)
        labels = np.asarray(trainer.train_loader.dataset.targets)
        recipients = [i for p in trainer.plans for i in p['recipients']]
        quality = purity(labels, choice['candidates'], choice['anchor_id'])
        result = dict(status='complete', seed=seed, method=method, epoch=trainer.args.epoch,
                      anchor=choice['anchor_id'], target_class=trainer.args.target_label,
                      test=metrics, candidate_purity=quality,
                      wrong_candidates=int(sum(labels[i] != trainer.args.target_label for i in choice['candidates'])),
                      recipient_purity=float(np.mean(labels[recipients] == trainer.args.target_label)),
                      cumulative_replacements=trainer.cumulative, start_state=trainer.start_state,
                      data_digest=digest(trainer.data_state), candidate_digest=digest(choice['candidates']),
                      final_model_digest=digest(final['state_dict']),
                      branch_seconds=final['comparison']['elapsed_seconds'],
                      choice_digest=digest(choice), checkpoint_digest=digest(final))
    finally:
        close(trainer, handler)
    del trainer
    if options.device >= 0: torch.cuda.empty_cache()
    evaluator, handler = assemble(options, seed, method, output, final, choice)
    try:
        evaluator._restore()
        actual_digest = digest([m.state_dict() for m in evaluator.model_list])
        independent = evaluator.test(evaluator.args.epoch-1, 'test')
        if actual_digest != result['final_model_digest'] or independent != metrics:
            raise ValueError('Independent evaluation differs')
        result['independent_evaluation'] = dict(test=independent, final_model_digest=actual_digest)
    finally: close(evaluator, handler)
    write_json(output/'metrics.json', result)
    return result


def verify_result_metadata(result, final, choice):
    """Bind reported labels and purities to the audited training checkpoint."""
    audit = final['comparison']['audit']
    last = audit[-1]
    count = len(choice['candidates'])
    quality = dict(full=last['candidate_purity'], full_denominator=count,
                   without_anchor=last['candidate_purity_without_anchor'],
                   without_anchor_denominator=count, anchor_in_pool=False)
    total = sum(row['recipient_count'] for row in audit)
    weighted = sum(row['recipient_count']*row['recipient_purity']
                   for row in audit if row['recipient_count'])/total
    if (result['anchor'] != choice['anchor_id'] or result['anchor'] != final['config']['anchor_idx']
            or result['target_class'] != final['config']['target_label']
            or result['data_digest'] != digest(final['data_state'])
            or result['candidate_digest'] != digest(choice['candidates'])
            or result['candidate_purity'] != quality
            or abs(result['recipient_purity']-weighted) > 1e-12
            or result['wrong_candidates'] != round(count*(1-quality['full']))):
        raise ValueError('Saved result metadata differs from checkpoint audit')


def saved_result(folder, method, seed, epochs, prefix, choice):
    """A completed marker alone is insufficient: verify durable plans and checkpoint."""
    output = folder/method
    result = read_json(output/'metrics.json')
    if not result or result.get('status') != 'complete': return None
    if not runtime_checkpoint(output).exists():
        protocol = read_json(folder.parent/'protocol.json')
        proof = load_completion(folder, protocol, seed) if protocol else None
        if proof is None:
            raise ValueError('Missing checkpoint and completion proof')
        if (method not in proof['records'] or result != proof['records'][method]
                or result['epoch'] != epochs or result['choice_digest'] != digest(choice)
                or (prefix is not None and proof['prefix']['digest'] != digest(prefix))):
            raise ValueError('Saved result differs from completion proof')
        return result
    final = load_checkpoint(runtime_checkpoint(output))
    verify_final(final, prefix, choice, epochs)
    verify_result_metadata(result, final, choice)
    if (result['seed'] != seed or result['method'] != method or result['epoch'] != epochs
            or result['choice_digest'] != digest(choice) or result['checkpoint_digest'] != digest(final)
            or result['test'] != result['independent_evaluation']['test']
            or result['final_model_digest'] != result['independent_evaluation']['final_model_digest']
            or result['final_model_digest'] != digest(final['state_dict'])
            or result['start_state'] != final['comparison']['start_state']
            or result['cumulative_replacements'] != final['comparison']['cumulative']):
        raise ValueError('Saved result identity differs')
    data = read_json(output/'experiment_data.json')
    if (not data or data['choice'] != choice or data['plans'] != final['comparison']['plans']
            or data['history'] != final['history'] or data['audit'] != final['comparison']['audit']):
        raise ValueError('Durable experiment records differ')
    import json
    configuration = dict(final['config'], data_state=final['data_state'],
                         trigger_dimensions=final['trigger_dimensions'])
    if read_json(output/'config.json') != json.loads(json.dumps(configuration, allow_nan=False)):
        raise ValueError('Durable configuration differs from checkpoint')
    return result


def run(options):
    from lfba_comparison.core_report import write_report
    root = options.output
    occupied = root.exists() and any(root.iterdir())
    if occupied and not options.resume: raise FileExistsError('Occupied output; use --resume')
    expected = protocol(options)
    failures = []
    with batch_lock(root):
        if occupied:
            if read_json(root/'protocol.json') != expected:
                raise ValueError('Resume requires identical protocol and source hashes')
        else:
            write_json(root/'protocol.json', expected)
        for seed in options.seeds:
            folder = root/f'seed_{seed}'
            if load_completion(folder, expected, seed) is not None:
                print(f'SKIP seed={seed} completed and audited', flush=True)
                continue
            prefix_path = runtime_checkpoint(folder/'clean_prefix')
            try:
                prefix = load_checkpoint(prefix_path) if prefix_path.exists() else None
                if prefix is None or prefix['epoch'] < 3:
                    print(f'CLEAN_PREFIX seed={seed}', flush=True)
                    trainer, handler = assemble(options, seed, 'capture', folder/'clean_prefix', prefix)
                    try: trainer.train()
                    finally: close(trainer, handler)
                    del trainer
                    prefix = load_checkpoint(prefix_path)
                verify_prefix(prefix)
                selection_path = folder/'selection.pth.tar'
                if selection_path.exists():
                    selection = load_checkpoint(selection_path)
                    if selection['prefix_digest'] != digest(prefix): raise ValueError('Selection prefix differs')
                else:
                    selection = select_core(prefix['comparison']['observations'],
                                            int(len(prefix['data_state']['train_ids'])*.1),
                                            options.block_size, 'cpu' if options.device < 0 else f'cuda:{options.device}')
                    selection['prefix_digest'] = digest(prefix)
                    atomic_save(cpu_tree(selection), selection_path)
                # Persist the numeric evidence, choices and timing once per clean trajectory.
                write_json(folder/'selection.json', dict(choices=selection['results'],
                           diagnostics=diagnostics(selection), prefix_digest=digest(prefix)))
            except Exception as error:
                failures.append(dict(seed=seed, method='clean_prefix', error=f'{type(error).__name__}: {error}'))
                print(f'FAILED seed={seed} prefix: {error}', flush=True)
                continue
            for method in METHODS:
                try:
                    choice = selection['results'][method]
                    if saved_result(folder, method, seed, expected['epochs'], prefix, choice):
                        print(f'SKIP seed={seed} method={method}', flush=True); continue
                    print(f'RUN seed={seed} method={method}', flush=True)
                    result = run_branch(options, seed, method, folder, prefix, choice)
                    print(f"COMPLETE seed={seed} method={method} ACC={result['test']['acc']:.5f} ASR={result['test']['asr']:.5f}", flush=True)
                except Exception as error:
                    failures.append(dict(seed=seed, method=method, error=f'{type(error).__name__}: {error}'))
                    import traceback
                    traceback.print_exc()
                finally:
                    if options.device >= 0: torch.cuda.empty_cache()
                write_json(root/'failures.json', failures)
                write_report(root)
        write_json(root/'failures.json', failures)
        report = write_report(root)
        if not failures:
            finish_ablation(root, report, getattr(options, 'keep_checkpoints', False))
    print('ALL_TASKS_COMPLETE' if report['all_complete'] and not failures else 'BATCH_INCOMPLETE', root, flush=True)
    return 0 if report['all_complete'] and not failures else 1
