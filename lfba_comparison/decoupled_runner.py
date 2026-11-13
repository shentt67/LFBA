"""Explicit manual-run orchestration; never imported by dry-run/report-only."""
from pathlib import Path
import traceback

import torch

from lfba_comparison.common import batch_lock, digest, read_json, write_json
from lfba_comparison.decoupled_policy import METHODS, build_ranking, choose
from lfba_comparison.decoupled_protocol import RATES, case_name, input_files, protocol
from lfba_comparison.decoupled_records import (
    completed_record, exposure_diagnostics, file_hash, load_references, required_json,
    snapshot, validate_metrics, verify_final, verify_bound_manifest, verify_prefix, verify_reference_prefix,
)
from utils.artifacts import runtime_checkpoint, save_configuration, save_experiment_data
from utils.checkpoint import atomic_save, capture_rng, cpu_tree, load_checkpoint


def assemble(options, seed, method, output, checkpoint=None, choice=None, rate=.1):
    # Assembly is deterministic and p does not affect model/data construction.
    # CoreTrainer already accepts frozen choices and restores all shared state.
    from lfba_comparison.core_runner import assemble as core_assemble
    trainer, handler = core_assemble(options, seed, method, output, checkpoint, choice)
    trainer.args.poison_rate = rate
    return trainer, handler


def close(trainer, handler):
    trainer.logger.removeHandler(handler)
    handler.close()


def prepare_seed(options, seed, root, expected):
    folder = root / f'seed_{seed}'
    refs = [] if options.smoke else load_references(options.reference_root, seed)
    path = runtime_checkpoint(folder / 'clean_prefix')
    prefix = load_checkpoint(path) if path.exists() else None
    if prefix is None or prefix['epoch'] < 3:
        print(f'CLEAN_PREFIX seed={seed}', flush=True)
        trainer, handler = assemble(options, seed, 'capture', folder / 'clean_prefix', prefix)
        try:
            trainer.train()
        finally:
            close(trainer, handler)
        del trainer
        prefix = load_checkpoint(path)
    verify_prefix(prefix)
    expected_size = [450, 50, 100] if options.smoke else [45000, 5000, 10000]
    if [len(prefix['data_state'][k]) for k in ('train_ids', 'val_ids', 'test_ids')] != expected_size:
        raise ValueError('Shared prefix split sizes differ')
    prefix_digest = digest(prefix)
    reference = (dict(verified=False, reason='Smoke has no formal historical pairing')
                 if options.smoke else verify_reference_prefix(prefix, refs))
    reference['prefix_digest'] = prefix_digest
    check_path = folder / 'reference_check.json'
    if check_path.exists() and required_json(check_path) != reference:
        raise ValueError('Saved reference check differs')
    if not check_path.exists():
        write_json(check_path, reference)
    n = len(prefix['data_state']['train_ids'])
    reference_count = int(n * expected['reference_rate'])
    selection_path = folder / 'selection.pth.tar'
    if selection_path.exists():
        ranking = load_checkpoint(selection_path)
        if ranking['prefix_digest'] != prefix_digest or ranking['reference_count'] != reference_count:
            raise ValueError('Saved ranking prefix/reference budget differs')
    else:
        rng = digest(capture_rng({}))
        ranking = build_ranking(prefix['comparison']['observations'], reference_count,
                                options.block_size, 'cpu' if options.device < 0 else f'cuda:{options.device}')
        if digest(capture_rng({})) != rng:
            raise ValueError('Selection consumed training random state')
        ranking['prefix_digest'] = prefix_digest
        atomic_save(cpu_tree(ranking), selection_path)
    choices = {case_name(p): {m: choose(ranking, m, int(n * p)) for m in METHODS} for p in RATES}
    metadata = dict(prefix_digest=prefix_digest, reference_count=reference_count,
                    anchor_id=ranking['anchor_id'], choices=choices,
                    ranking_digests={m: digest(ranking['rankings'][m]) for m in METHODS},
                    evidence_sha256=file_hash(selection_path),
                    shared_selection_seconds=ranking['elapsed_seconds'])
    selected_path = folder / 'selection.json'
    if selected_path.exists() and required_json(selected_path) != metadata:
        raise ValueError('Saved selection evidence differs')
    if not selected_path.exists():
        write_json(selected_path, metadata)
    for method in METHODS:
        small, large = (choices[case_name(p)][method] for p in RATES)
        if not set(small['candidates']) < set(large['candidates']):
            raise ValueError('Candidate pools must be nested')
    return prefix, choices


def run_branch(options, root, seed, rate, method, prefix, choice, expected):
    output = root / f'seed_{seed}' / case_name(rate) / method
    checkpoint_path = runtime_checkpoint(output)
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else prefix
    if checkpoint is not prefix:
        if (checkpoint['comparison']['method'] != method or checkpoint['config']['poison_rate'] != rate
                or checkpoint['comparison']['choice'] != choice or checkpoint['config']['epoch'] != expected['epochs']):
            raise ValueError('Resume checkpoint belongs to another branch')
    trainer, handler = assemble(options, seed, method, output, checkpoint, choice, rate)
    try:
        trainer.train()
        final = load_checkpoint(checkpoint_path)
        verify_final(final, prefix, choice, expected['epochs'], rate)
        metrics = trainer.test(expected['epochs'] - 1, 'test')
        validate_metrics(metrics, formal=not options.smoke)
        save_configuration(output, trainer)
        save_experiment_data(output, trainer)
        evidence = exposure_diagnostics(trainer.train_loader.dataset.targets, choice, trainer.plans)
        result = dict(status='complete', seed=seed, p=rate, s=.5, method=method,
                      epoch=expected['epochs'], test=metrics,
                      anchor=choice['anchor_id'], target_class=trainer.args.target_label,
                      candidate_count=len(choice['candidates']), reference_count=choice['reference_count'],
                      candidate_purity=evidence['candidate_purity'], recipient_purity=evidence['recipient_purity'],
                      cumulative_replacements=trainer.cumulative, diagnostics=evidence,
                      start_state=trainer.start_state, initial_digest=trainer.initial_digest,
                      first_prefix=trainer.prefix[0], data_digest=digest(final['data_state']),
                      prefix_digest=digest(prefix), candidate_digest=digest(choice['candidates']),
                      choice_digest=digest(choice), final_model_digest=digest(final['state_dict']),
                      protocol_digest=digest(expected))
    finally:
        close(trainer, handler)
    del trainer
    if options.device >= 0:
        torch.cuda.empty_cache()
    evaluator, handler = assemble(options, seed, method, output, final, choice, rate)
    try:
        evaluator._restore()
        actual = digest([m.state_dict() for m in evaluator.model_list])
        independent = evaluator.test(expected['epochs'] - 1, 'test')
        if actual != result['final_model_digest'] or independent != metrics:
            raise ValueError('Independent evaluation differs')
        result['independent_evaluation'] = dict(epoch=expected['epochs'], test=independent,
                                               final_model_digest=actual)
    finally:
        close(evaluator, handler)
    del evaluator
    write_json(output / 'metrics.json', result)
    folder = root / f'seed_{seed}'
    files = [output / name for name in ('metrics.json', 'config.json', 'experiment_data.json')]
    files += [checkpoint_path, runtime_checkpoint(folder / 'clean_prefix'), folder / 'selection.json',
              folder / 'selection.pth.tar', folder / 'reference_check.json']
    write_json(output / 'completion.json', dict(protocol_digest=digest(expected),
               files={str(p.relative_to(root)): file_hash(p) for p in files}))
    if completed_record(root, output, expected) != result:
        raise ValueError('Completion read-back differs')
    return result


def run(options):
    from lfba_comparison.decoupled_report import write_report
    root = options.output
    occupied = root.exists() and any(root.iterdir())
    if occupied and not options.resume:
        raise FileExistsError('Occupied output; use --run --resume for this exact protocol')
    expected = protocol(options)
    with batch_lock(root):
        if occupied:
            saved_protocol = required_json(root / 'protocol.json')
            if {k: v for k, v in saved_protocol.items() if k not in ('input_paths', 'input_manifest_digest')} != expected:
                raise ValueError('Resume requires identical protocol, device and source hashes')
            expected = saved_protocol
            manifest = required_json(root / 'input_manifest.json')
        else:
            manifest = snapshot(input_files(options))
            expected.update(input_paths=sorted(manifest), input_manifest_digest=digest(manifest))
            write_json(root / 'protocol.json', expected)
            write_json(root / 'input_manifest.json', manifest)
        check = verify_bound_manifest(manifest, expected)
        if not check['unchanged']:
            raise ValueError('Original inputs changed: ' + repr(check['changed']))
        failures = read_json(root / 'failures.json') or []
        write_json(root / 'verification.json', dict(status='in_progress', **check))
        try:
            for seed in expected['seeds']:
                try:
                    prefix, choices = prepare_seed(options, seed, root, expected)
                    failures = [f for f in failures if not (f['seed'] == seed and f['stage'] == 'prefix')]
                except Exception as error:
                    failures = [f for f in failures if not (f['seed'] == seed and f['stage'] == 'prefix')]
                    failures.append(dict(seed=seed, stage='prefix', error=f'{type(error).__name__}: {error}'))
                    write_json(root / 'failures.json', failures)
                    print(f'FAILED seed={seed} prefix: {error}', flush=True)
                    continue
                for rate in RATES:
                    for method in METHODS:
                        output = root / f'seed_{seed}' / case_name(rate) / method
                        identity = dict(seed=seed, stage='branch', p=rate, method=method)
                        failures = [f for f in failures if any(f.get(k) != v for k, v in identity.items())]
                        try:
                            saved = completed_record(root, output, expected)
                            if saved is None:
                                print(f'RUN seed={seed} p={rate} method={method}', flush=True)
                                saved = run_branch(options, root, seed, rate, method, prefix,
                                                   choices[case_name(rate)][method], expected)
                            print(f"COMPLETE seed={seed} p={rate} {method} ACC={saved['test']['acc']:.5f} ASR={saved['test']['asr']:.5f}", flush=True)
                        except Exception as error:
                            failures.append(dict(**identity, error=f'{type(error).__name__}: {error}'))
                            traceback.print_exc()
                        finally:
                            if options.device >= 0:
                                torch.cuda.empty_cache()
                            write_json(root / 'failures.json', failures)
                del prefix
                write_report(root)
        finally:
            write_json(root / 'failures.json', failures)
            report = write_report(root)
        print('BATCH_COMPLETE' if report['all_complete'] else 'BATCH_INCOMPLETE', root, flush=True)
        return 0 if report['all_complete'] else 1
