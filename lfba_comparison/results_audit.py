"""Saved-result inventory. No trainer, model or checkpoint module is imported."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def candidate_hash(candidates):
    """Match the training digest for a plain ordered list of integer sample IDs."""
    h=hashlib.sha256()
    for sample_id in candidates:
        h.update(repr(sample_id).encode())
    return h.hexdigest()


def snapshot(paths):
    return {str(Path(p).resolve()): dict(size=Path(p).stat().st_size, sha256=file_hash(p))
            for p in sorted(set(paths))}


def verify_snapshot(before):
    for name, expected in before.items():
        p = Path(name)
        if not p.is_file() or p.stat().st_size != expected['size'] or file_hash(p) != expected['sha256']:
            raise ValueError('Input changed during audit: '+name)


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def durable_data(folder):
    folder = Path(folder)
    if (folder/'experiment_data.json').exists():
        return read(folder/'experiment_data.json')
    audit = []
    if (folder/'epoch_metrics.csv').exists():
        with (folder/'epoch_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            audit = list(csv.DictReader(stream))
    return dict(history=read(folder/'history.json', []), choice=read(folder/'choice.json'),
                plans=read(folder/'replacement_ids.json', []), audit=audit)


def locate(root, name):
    p = Path(name)
    if p.exists():
        return p
    portable = str(name).replace('\\', '/')
    # Historical worktree paths may no longer exist; use a local suffix only.
    for marker in ('scripts/results/', 'results/'):
        if marker in portable:
            candidate = root/portable[portable.rfind(marker):]
            if candidate.exists():
                return candidate
    return p


def protocol_identity(protocol):
    # Paths, seed lists and code provenance do not define a training protocol.
    omitted = {'seeds', 'data_dir', 'device', 'block_size', 'source_hashes', 'source_manifest',
               'source_results', 'keep_checkpoints', 'thresholds', 'primary', 'supplement'}
    return stable_hash({k:v for k,v in protocol.items() if k not in omitted})


def normalize(folder, batch, protocol, category, root, result=None, config=None, data=None):
    result = read(folder/'metrics.json') if result is None else result
    config = read(folder/'config.json', {}) if config is None else config
    data = durable_data(folder) if data is None else data
    if not isinstance(result, dict) or not isinstance(config, dict) or not isinstance(data, dict):
        raise ValueError('Result, configuration and experiment data must be objects')
    test = result.get('test', {})
    if not isinstance(test, dict) or not test:
        raise ValueError('Missing test metrics')
    choice = data.get('choice') or {}
    candidates = choice.get('candidates', [])
    plans = data.get('plans') or []
    history = data.get('history') or []
    proof = result.get('independent_evaluation') or read(folder/'independent_evaluation.json', {})
    independent = 'unavailable'
    if proof:
        independent = 'match' if proof.get('test') == test else 'mismatch'
        if proof.get('final_model_digest') and result.get('final_model_digest'):
            if proof['final_model_digest'] != result['final_model_digest']:
                independent = 'mismatch'
    evaluated = result.get('epoch', result.get('checkpoint_epoch'))
    final = protocol.get('final_model') or ('checkpoint' if config.get('test_checkpoint') else 'validation_best')
    quality = result.get('candidate_purity')
    quality = quality if isinstance(quality, dict) else dict(full=quality)
    observed = [p['epoch'] for p in plans if p.get('recipients')]
    trained = len(history) or result.get('epoch')
    method = result.get('method', config.get('comparison_method', config.get('attack') or 'clean'))
    if config.get('test_checkpoint'):
        category = 'historical_evaluation'
    elif not protocol and category!='formal':
        category = 'smoke' if config.get('train_limit') or config.get('eval_limit') else 'standalone_training'
    state = config.get('data_state') or {}
    events = sum(len(p.get('recipients', [])) for p in plans)
    declared = result.get('cumulative_replacements')
    status = 'complete'
    problems = []
    if category=='formal':
        if (not config.get('dataset') or config.get('seed') is None or not state.get('train_ids')
                or not choice or not candidates or not plans or independent!='match'):
            problems.append('Missing formal configuration, candidates, plans or independent evaluation evidence')
        if choice.get('anchor_id') != result.get('anchor',config.get('anchor_idx')):
            problems.append('Choice anchor differs from saved configuration')
        for key,config_key in [('anchor','anchor_idx'),('target_class','target_label'),('seed','seed')]:
            if result.get(key) is not None and config.get(config_key)!=result[key]:
                problems.append('Result '+key+' differs from configuration')
        if result.get('candidate_digest') and candidate_hash(candidates)!=result['candidate_digest']:
            problems.append('Saved candidate digest differs from actual candidate IDs')
        train=set(state.get('train_ids',[]));pool=set(candidates)
        if not pool<=train:
            problems.append('Candidate IDs outside saved training split')
        for plan in plans:
            recipients=plan.get('recipients',[]);donors=plan.get('donors',[])
            if (len(recipients)!=len(donors) or len(set(recipients))!=len(recipients)
                    or len(set(donors))!=len(donors) or not set(recipients)<=pool
                    or not set(donors)<=train-pool):
                problems.append('Saved replacement plan is invalid')
                break
    if independent == 'mismatch':
        problems.append('Independent evaluation differs')
    if declared is not None and plans and declared != events:
        problems.append('Replacement event total differs from saved plans')
    if result.get('method') and evaluated and len(history) != evaluated:
        problems.append('Training history does not reach evaluation epoch')
    if category=='formal' and evaluated and [h.get('epoch') for h in history]!=list(range(1,evaluated+1)):
        problems.append('Training history epoch sequence differs')
    if proof.get('epoch') is not None and proof['epoch'] != evaluated:
        problems.append('Independent evaluation epoch differs')
    if len(set(candidates)) != len(candidates):
        problems.append('Duplicate candidate IDs')
    if result.get('status') not in (None, 'complete'):
        problems.append('Result status is not complete')
    if protocol.get('epochs') and result.get('method') and evaluated != protocol['epochs']:
        problems.append('Final evaluation epoch differs from saved protocol')
    for k in ('acc','asr'):
        value = test.get(k)
        if value is not None and (not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 <= value <= 1):
            problems.append('Invalid '+k)
    if problems:
        status = 'corrupt'
    return dict(source=str(folder.resolve()), batch=str(batch.relative_to(root)),
        dataset=config.get('dataset', protocol.get('dataset', 'CIFAR10' if protocol else None)),
        protocol=protocol.get('kind', 'comparison' if protocol else 'standalone'),
        protocol_version=protocol.get('version'), protocol_group=protocol_identity(protocol or {
            k:config.get(k) for k in ('dataset','attack','epoch','lr','batch_size','early_stop','similarity')}),
        seed=result.get('seed', config.get('seed')), method=method, category=category,
        status=status, problems=problems, training_epochs=trained, configured_epochs=config.get('epoch'),
        evaluation_epoch=evaluated, endpoint=final, anchor=result.get('anchor', config.get('anchor_idx')),
        target_class=result.get('target_class',config.get('target_label')),
        similarity=config.get('similarity'), poison_rate=config.get('poison_rate',protocol.get('poison_rate',protocol.get('candidate_rate'))),
        select_rate=config.get('select_rate',protocol.get('select_rate')),
        candidate_count=len(candidates) if choice else None,
        attack_start=min(observed) if observed else None,
        cumulative_replacements=declared if declared is not None else (events if plans else None),
        acc=test.get('acc'), asr=test.get('asr'), poison_acc=test.get('poison_acc'),
        acc_denominator=test.get('counts',{}).get('acc'), asr_denominator=test.get('counts',{}).get('asr'),
        candidate_purity=quality.get('full'), candidate_purity_without_anchor=quality.get('without_anchor'),
        recipient_purity=result.get('recipient_purity', result.get('replacement_purity')),
        independent_evaluation=independent, data_digest=result.get('data_digest') or (stable_hash(state) if state else None),
        candidate_digest=result.get('candidate_digest') or (candidate_hash(candidates) if choice else None),
        final_model_digest=result.get('final_model_digest') or proof.get('final_model_digest'),
        start_state=result.get('start_state'), initial_digest=result.get('initial_digest'),
        anchor_source=protocol.get('anchor_source'), origin='saved', reference_to=None)


def scan(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocols = {}
    issues, rows, batches = [], [], []
    def note(path, error):
        issues.append(dict(source=str(path), error=str(error)))
    for base in (root/'results', root/'scripts/results'):
        if not base.exists():
            continue
        for p in sorted(base.rglob('protocol.json')):
            if p.is_relative_to(output):
                continue
            try:
                value = read(p)
                if not isinstance(value, dict):
                    raise ValueError('Protocol must be an object')
                protocols[p.parent] = value
            except (ValueError,OSError) as error:
                note(p,error)
    # Materialize absent matrix children in memory only. Do not create protocols.
    for batch, p in list(protocols.items()):
        if p.get('kind') != 'stability':
            continue
        for case in p.get('cases',[]):
            child=batch/case['id']
            if child not in protocols:
                protocols[child]=dict(p,kind='missing_child_protocol',cases=None,
                    poison_rate=case['poison_rate'],select_rate=case['select_rate'])
    def category(batch, p):
        if p.get('smoke') or 'smoke' in batch.name.lower():
            return 'smoke'
        if p:
            return 'formal'
        return 'historical_evaluation'
    def add(folder, batch, p):
        if not (folder/'metrics.json').exists():
            try:
                cfg=read(folder/'config.json',{})
                progress=len(durable_data(folder).get('history',[]))
            except (ValueError,TypeError,OSError) as error:
                note(folder,error)
                return dict(source=str(folder.resolve()),batch=str(batch.relative_to(root)),
                    status='corrupt',category=category(batch,p),problems=[str(error)])
            return dict(source=str(folder.resolve()),batch=str(batch.relative_to(root)),
                        dataset=cfg.get('dataset',p.get('dataset','CIFAR10' if p else None)),
                        protocol=p.get('kind','comparison' if p else 'standalone'),
                        poison_rate=cfg.get('poison_rate',p.get('poison_rate',p.get('candidate_rate'))),
                        select_rate=cfg.get('select_rate',p.get('select_rate')),
                        configured_epochs=cfg.get('epoch',p.get('epochs')),training_epochs=progress or None,
                        status='missing',category=category(batch,p),problems=['No final metrics'])
        try:
            row = normalize(folder,batch,p,category(batch,p),root)
            if row['problems']:
                note(folder,'; '.join(row['problems']))
            return row
        except (ValueError,KeyError,TypeError,OSError) as error:
            note(folder,error)
            return dict(source=str(folder.resolve()),batch=str(batch.relative_to(root)),
                        status='corrupt', category=category(batch,p),problems=[str(error)])
    handled = set()
    for batch, p in sorted(protocols.items()):
        if p.get('kind') == 'stability':
            # Parent matrix is an inventory roll-up, not another set of trainings.
            continue
        seeds, methods = p.get('seeds',[]), p.get('methods',[])
        if not seeds or not methods:
            note(batch, 'No declared seed/method matrix')
            continue
        local=[]
        for seed in seeds:
            for method in methods:
                folder=batch/f'seed_{seed}'/method
                row=add(folder,batch,p)
                row.setdefault('seed',seed); row.setdefault('method',method)
                if row.get('seed') != seed or row.get('method') != method:
                    row['status']='corrupt'; row['problems'].append('Seed/method differs from folder')
                    note(folder,row['problems'][-1])
                rows.append(row); local.append(row); handled.add(folder.resolve())
        batches.append(dict(batch=str(batch.relative_to(root)),category=category(batch,p),
            expected=len(seeds)*len(methods),complete=sum(r['status']=='complete' for r in local),
            missing=sum(r['status']=='missing' for r in local),corrupt=sum(r['status']=='corrupt' for r in local),
            expected_references=0, complete_references=0))
        # References point to actual old runs. Never call the training/report runners.
        manifest=p.get('source_manifest') or {}
        for seed, info in manifest.get('seeds',{}).items():
            archive=read(batch/f'seed_{seed}'/'references.json',{})
            for method, ref in info.get('references',{}).items():
                batches[-1]['expected_references']+=1
                folder=locate(root,ref['directory'])
                try:
                    row=normalize(folder,batch,p,category(batch,p),root)
                    # Validate the small source artifacts against the saved manifest;
                    # checkpoint files are deliberately never opened.
                    for name, expected_hash in manifest.get('files',{}).items():
                        f=locate(root,name)
                        if f.parent.resolve()==folder.resolve() and f.name in ('metrics.json','config.json','experiment_data.json'):
                            if file_hash(f)!=expected_hash:
                                row['status']='corrupt'
                                row['problems'].append('Referenced source artifact differs from saved manifest')
                                note(f,row['problems'][-1])
                except (ValueError,KeyError,TypeError,OSError):
                    saved=next((r for r in archive.get('records',[]) if r.get('result',{}).get('method')==method),None)
                    if not saved:
                        note(folder,'Referenced result unavailable')
                        continue
                    row=normalize(folder,batch,p,category(batch,p),root,
                                  saved['result'],saved['config'],saved['data'])
                row['origin']='reference';row['reference_to']=str(folder.resolve())
                row['source']=str((batch/f'seed_{seed}'/('reference_'+method)).resolve())
                rows.append(row)
                batches[-1]['complete_references']+=row['status']=='complete'
    for base in (root/'results',root/'scripts/results'):
        if not base.exists():continue
        extras={f.parent for f in base.rglob('metrics.json')} | {f.parent for f in base.rglob('config.json')}
        for directory in sorted(extras):
            f=directory/'metrics.json'
            if f.is_relative_to(output) or directory.resolve() in handled:continue
            if directory.name in ('clean_first','clean_prefix'):continue
            parents=[p for p in protocols if f.is_relative_to(p)]
            batch=max(parents,key=lambda p:len(p.parts)) if parents else base/f.relative_to(base).parts[0]
            if batch==f:batch=f.parent
            p=protocols.get(batch,{})
            row=add(f.parent,batch,p)
            if not parents:
                row['batch']=str(f.parent.relative_to(root))
            if p and f.parent.parent.name.startswith('seed_'):
                row['category']='extra_seed'
            rows.append(row)
            if not parents:
                batches.append(dict(batch=str(f.parent.relative_to(root)),category=row['category'],
                    expected=1,complete=int(row['status']=='complete'),missing=0,
                    corrupt=int(row['status']=='corrupt'),expected_references=0,complete_references=0))
    for batch,p in protocols.items():
        if p.get('kind')!='stability':continue
        children=[b for b in batches if (root/b['batch']).parent==batch]
        batches.append(dict(batch=str(batch.relative_to(root)), category=category(batch,p),
            expected=len(p.get('cases',[]))*len(p.get('seeds',[]))*len(p.get('methods',[])),
            complete=sum(b['complete'] for b in children),missing=sum(b['missing'] for b in children),
            corrupt=sum(b['corrupt'] for b in children),expected_references=0,complete_references=0,rollup=True))
    return rows,batches,issues


def identity(row):
    keys=('dataset','seed','evaluation_epoch','target_class','data_digest','candidate_digest','final_model_digest')
    if any(row.get(k) is None for k in keys):return None
    return stable_hash([row[k] for k in keys])


def deduplicate(rows):
    seen={}
    identities={}
    for row in rows:
        key=identity(row) if row.get('status')=='complete' else None
        if key:identities.setdefault(key,[]).append(row)
    for values in identities.values():
        conflict=any(len({r[k] for r in values if r.get(k) is not None})>1
                     for k in ('acc','asr','poison_acc','acc_denominator','asr_denominator'))
        if conflict:
            for row in values:
                row['status']='corrupt'
                row.setdefault('problems',[]).append('Conflicting metrics for the same saved model identity')
    for row in sorted(rows,key=lambda r:(r.get('origin')=='reference',r['source'])):
        key=identity(row) if row.get('status')=='complete' else None
        row['experiment_id']=key
        row['duplicate_of']=seen.get(key) if key else None
        if key:seen.setdefault(key,row['source'])
    return rows


def eligible(row):
    return (row.get('status')=='complete' and row.get('category')=='formal'
            and row.get('acc') is not None and row.get('asr') is not None)


def update_batch_counts(rows,batches):
    """Deduplication conflicts must also invalidate the inventory completion count."""
    for batch in batches:
        if batch.get('rollup'):continue
        members=[r for r in rows if r['batch']==batch['batch']
                 and r.get('origin')!='reference' and r.get('category')!='extra_seed']
        for state in ('complete','missing','corrupt'):
            batch[state]=sum(r['status']==state for r in members)
        batch['complete_references']=sum(r['status']=='complete' for r in rows
            if r['batch']==batch['batch'] and r.get('origin')=='reference')
    for batch in batches:
        if not batch.get('rollup'):continue
        parent=Path(batch['batch'])
        children=[b for b in batches if not b.get('rollup') and Path(b['batch']).parent==parent]
        for state in ('complete','missing','corrupt'):
            batch[state]=sum(b[state] for b in children)


def group_key(row):
    return tuple(row.get(k) for k in ('dataset','protocol_group','poison_rate','select_rate',
                                     'configured_epochs','evaluation_epoch','endpoint','method'))


def unique_seeds(rows):
    by_seed={}
    for r in rows:by_seed.setdefault(r['seed'],[]).append(r)
    selected=[];ambiguous=[]
    for seed,values in by_seed.items():
        ids={r.get('experiment_id') for r in values}
        if len(values)>1 and (len(ids)!=1 or None in ids):
            ambiguous.append(seed)
        else:selected.append(values[0])
    return selected,sorted(ambiguous)


def aggregate(rows):
    groups={}
    for r in rows:
        if eligible(r):groups.setdefault(group_key(r),[]).append(r)
    output=[]
    for _,members in groups.items():
        selected,ambiguous=unique_seeds(members)
        first=members[0]
        group={k:first.get(k) for k in ('dataset','protocol_group','poison_rate','select_rate',
                                      'configured_epochs','evaluation_epoch','endpoint','method')}
        group.update(n=len(selected),seeds=[r['seed'] for r in selected],ambiguous_seeds=ambiguous,
                     batches=sorted({r['batch'] for r in members}))
        for metric in ('acc','asr'):
            values=[r[metric] for r in selected]
            group[metric+'_mean']=statistics.mean(values) if values else None
            group[metric+'_std']=statistics.stdev(values) if len(values)>1 else None
        output.append(group)
    return output


def paired(rows):
    comparisons=[('lfba_original','auto_vote'),('lfba_matched','auto_vote'),
                 ('auto_single','auto_rank_last'),('auto_rank_last','auto_mean'),
                 ('auto_mean','auto_vote'),('fixed_single','fixed_vote'),
                 ('fixed_single','auto_rank_last'),('fixed_vote','auto_vote'),('auto_rank_last','auto_vote')]
    batches={}
    candidates={}
    for row in rows:
        if eligible(row):candidates.setdefault((row['batch'],row['seed'],row['method']),[]).append(row)
    for (batch,seed,method),values in candidates.items():
        values,_=unique_seeds(values)
        if values:batches.setdefault((batch,seed),{})[method]=values[0]
    output=[]
    for (batch,seed),methods in batches.items():
        for a,b in comparisons:
            if a not in methods or b not in methods:continue
            x,y=methods[a],methods[b]
            if any(not x.get(k) or x[k]!=y.get(k) for k in ('data_digest','start_state')):continue
            if any(x.get(k)!=y.get(k) for k in ('dataset','poison_rate','select_rate','evaluation_epoch','endpoint')):continue
            if x.get('anchor_source')=='auto_vote' and (x.get('anchor')!=y.get('anchor') or x['target_class']!=y['target_class']):continue
            if (a,b) in [('auto_rank_last','auto_mean'),('auto_mean','auto_vote'),
                         ('fixed_single','fixed_vote'),('auto_rank_last','auto_vote')]:
                if x.get('anchor')!=y.get('anchor') or x['target_class']!=y['target_class']:continue
            output.append(dict(batch=batch,seed=seed,comparison=b+' - '+a,
                delta_acc_pp=100*(y['acc']-x['acc']),delta_asr_pp=100*(y['asr']-x['asr']),
                same_target=x['target_class']==y['target_class'],
                base_replacements=x.get('cumulative_replacements'),new_replacements=y.get('cumulative_replacements'),
                base_asr_denominator=x.get('asr_denominator'),new_asr_denominator=y.get('asr_denominator')))
    return output


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
