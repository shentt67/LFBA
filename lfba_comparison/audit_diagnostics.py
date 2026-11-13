"""Label joins and actual replacement-event accounting, without inference."""
from collections import Counter
import hashlib
import pickle
from pathlib import Path

import numpy as np

from lfba_comparison.results_audit import durable_data, read


def histogram(values):
    return {str(k):int(v) for k,v in sorted(Counter(int(i) for i in values).items())}


def exposure_statistics(labels, candidates, target, plans):
    labels=np.asarray(labels,dtype=int)
    if not candidates or len(set(candidates))!=len(candidates):
        raise ValueError('Missing or duplicate candidates')
    if any(not 0<=i<len(labels) for i in candidates):
        raise ValueError('Candidate outside raw label array')
    uses=Counter();donors=[]
    for p in plans:
        recipients=p.get('recipients',[]);suppliers=p.get('donors',[])
        if (len(recipients)!=len(suppliers) or len(set(recipients))!=len(recipients)
                or len(set(suppliers))!=len(suppliers) or not set(recipients)<=set(candidates)
                or set(suppliers)&set(candidates)
                or any(i is None or not 0<=i<len(labels) for i in suppliers)):
            raise ValueError('Invalid recipient/donor plan')
        uses.update(recipients);donors.extend(suppliers)
    events=sum(uses.values())
    wrong=sum(n for i,n in uses.items() if labels[i]!=target)
    classes=histogram(labels[candidates])
    majority=max(classes,key=classes.get)
    return dict(candidate_class_counts=classes,candidate_majority_class=int(majority),
        majority_matches_target=int(majority)==target,
        candidate_purity_recomputed=float(np.mean(labels[candidates]==target)),
        wrong_candidates=int(np.sum(labels[candidates]!=target)),
        replacement_events=events,wrong_recipient_events=int(wrong),
        recipient_purity_recomputed=1-wrong/events if events else None,
        recipient_class_events={str(c):int(sum(n for i,n in uses.items() if labels[i]==c))
                                for c in sorted(set(labels[candidates]))},
        unique_recipients=len(uses),recipient_coverage=len(uses)/len(candidates),
        repeated_events=events-len(uses),max_recipient_uses=max(uses.values(),default=0),
        recipient_use_histogram=histogram([uses[i] for i in candidates]),
        unique_donors=len(set(donors)),donor_class_events=histogram(labels[donors]),
        donor_class_unique=histogram(labels[sorted(set(donors))]))


def load_labels(data_dir, dataset):
    """Return original training-source labels and the exact input file manifest."""
    data_dir=Path(data_dir)
    if dataset=='CIFAR10':
        files=[data_dir/'cifar-10-batches-py'/f'data_batch_{i}' for i in range(1,6)]
        labels=[]
        for p in files:
            with p.open('rb') as stream:
                entry=pickle.load(stream,encoding='latin1')
            labels.extend(entry['labels'])
        return np.asarray(labels,dtype=int),files,None
    if dataset=='UCIHAR':
        directory=data_dir/'UCI-HAR'/'UCI HAR Dataset'/'train'
        files=[directory/'y_train.txt',directory/'X_train.txt']
        return np.loadtxt(files[0],dtype=int)-1,files,np.loadtxt(files[1])
    if dataset=='NUSWIDE':
        directory=data_dir/'NUS_WIDE'/'Groundtruth'/'TrainTestLabels'
        files=[directory/f'Labels_{name}_Train.txt' for name in ('buildings','grass','animal','water','person')]
        columns=np.column_stack([np.loadtxt(p,dtype=int) for p in files])
        selected=columns[columns.sum(axis=1)==1]
        return selected.argmax(axis=1),files,None
    raise ValueError('Unsupported label source: '+dataset)


def selected_batch(row):
    name=row['batch'].replace('\\','/')
    return (name.startswith('results/lfba_stability_full/p_') or
            (name.startswith('results/nuswide_lfba_vs_vote_') and 'p005_s05' not in name) or
            name=='results/ucihar_lfba_vs_vote_p01_s05_v1')


def diagnose(rows, data_dir, issues):
    selected=[r for r in rows if r.get('status')=='complete' and selected_batch(r)]
    datasets={};inputs=[]
    for dataset in sorted({r['dataset'] for r in selected}):
        try:
            datasets[dataset]=load_labels(data_dir,dataset)
            inputs.extend(datasets[dataset][1])
        except (ValueError,OSError) as error:
            issues.append(dict(source=str(data_dir),error=f'{dataset}: {error}'))
    diagnostics=[];curves=[];trigger_rows=[]
    for row in selected:
        folder=Path(row['source'])
        try:
            branch_curves=[];branch_triggers=[]
            if row['dataset'] not in datasets:raise ValueError('Raw labels unavailable')
            labels,_,features=datasets[row['dataset']]
            config=read(folder/'config.json',{})
            data=durable_data(folder);choice=data.get('choice') or {}
            candidates=choice.get('candidates',[]);plans=data.get('plans',[])
            anchor=row['anchor'];target=row['target_class']
            if not 0<=anchor<len(labels) or int(labels[anchor])!=target:
                raise ValueError('Raw anchor label differs from saved target')
            train=config.get('data_state',{}).get('train_ids',[])
            if not set(candidates)<=set(train):raise ValueError('Candidate outside training split')
            if any(not set(p['donors'])<=set(train) for p in plans):
                raise ValueError('Donor outside training split')
            stats=exposure_statistics(labels,candidates,target,plans)
            if row.get('cumulative_replacements')!=stats['replacement_events']:
                raise ValueError('Recomputed event count differs')
            for key,recomputed in [('candidate_purity','candidate_purity_recomputed'),
                                   ('recipient_purity','recipient_purity_recomputed')]:
                if row.get(key) is not None and abs(row[key]-stats[recomputed])>1e-10:
                    raise ValueError('Recomputed '+key+' differs')
            target_count=int(np.sum(labels[train]==target))
            stats['target_class_train_count']=target_count
            stats['target_class_candidate_coverage']=int(np.sum(labels[candidates]==target))/target_count
            available=target_count-int(anchor not in candidates)
            stats['candidate_purity_upper_bound']=min(len(candidates),available)/len(candidates)
            stats['wrong_event_fraction']=stats['wrong_recipient_events']/stats['replacement_events']
            q=choice.get('diagnostics',{})
            stats.update({k:q.get(k) for k in ('cutoff_votes','boundary_tied_count','selected_by_tiebreak','vote_histogram')})
            values=[]
            for h in data.get('history',[]):
                v=h.get('validation') or {}
                branch_curves.append(dict(batch=row['batch'],dataset=row['dataset'],seed=row['seed'],
                    method=row['method'],epoch=h['epoch'],acc=v.get('acc'),asr=v.get('asr'),
                    poison_acc=v.get('poison_acc'),target_acc=v.get('target_acc')))
                if v.get('asr') is not None:values.append((h['epoch'],v['asr']))
            stats.update(validation_asr_last=values[-1][1] if values else None,
                validation_asr_last5_mean=float(np.mean([v for _,v in values[-5:]])) if values else None,
                validation_asr_last5_std=float(np.std([v for _,v in values[-5:]])) if values else None,
                validation_asr_max=max((v for _,v in values),default=None),
                validation_asr_max_epoch=max(values,key=lambda p:p[1])[0] if values else None)
            diagnosis=dict(source=row['source'],batch=row['batch'],dataset=row['dataset'],
                seed=row['seed'],method=row['method'],poison_rate=row['poison_rate'],select_rate=row['select_rate'],
                anchor=anchor,target_class=target,acc=row['acc'],asr=row['asr'],
                candidate_count=len(candidates),**stats)
            if row['dataset']=='UCIHAR':
                dims=config.get('trigger_dimensions',[])
                if not dims or len(set(dims))!=len(dims) or any(not 281<=d<561 for d in dims):
                    raise ValueError('Invalid attacker trigger dimensions')
                value=config.get('trigger_feature_clip',1.)
                # This check binds the current raw source to the saved training fingerprint.
                fingerprint=hashlib.sha256()
                fingerprint.update(features.tobytes());fingerprint.update(labels.astype(float).tobytes())
                saved=config.get('data_state',{}).get('fingerprints',{}).get('source')
                matches=fingerprint.hexdigest()==saved if saved else None
                for dim in dims:
                    for population,ids in [('train',train),('target',[i for i in train if labels[i]==target]),
                                           ('non_target',[i for i in train if labels[i]!=target])]:
                        x=features[ids,dim];delta=np.abs(x-value)
                        branch_triggers.append(dict(seed=row['seed'],method=row['method'],target_class=target,
                            dimension=dim,population=population,n=len(x),trigger_value=value,
                            min=float(x.min()),q25=float(np.quantile(x,.25)),median=float(np.median(x)),
                            q75=float(np.quantile(x,.75)),max=float(x.max()),mean=float(x.mean()),
                            std=float(x.std()),mean_absolute_change=float(delta.mean()),
                            unchanged_fraction=float(np.mean(x==value)),source_fingerprint_matches=matches))
                if matches is False:raise ValueError('UCIHAR raw source fingerprint differs')
            diagnostics.append(diagnosis)
            curves.extend(branch_curves);trigger_rows.extend(branch_triggers)
        except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
            issues.append(dict(source=row['source'],error='diagnosis: '+str(error)))
    return diagnostics,curves,trigger_rows,inputs
