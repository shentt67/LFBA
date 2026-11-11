"""Fixed-budget clean-trajectory experiment. Noise below is selection-only diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import main as entry
from utils.config import parse_args
from attack.multiround import similarity_scores, select_positions


class CaptureTrainer(entry.Trainer):
    def test(self,ep=0,split='test'):
        if split=='val':
            if self.current_plan or self.poison_indexes is not None:
                raise RuntimeError('Capture must remain unpoisoned')
            order=torch.argsort(self.indexes_epoch)
            torch.save(dict(epoch=ep+1,ids=self.indexes_epoch[order],
                            gradients=self.grad_vec_epoch[order],labels=self.target_epoch[order]),
                       Path(self.args.results_dir)/f'gradient_epoch_{ep+1}.pt')
        return super().test(ep,split)


def analyze(root,seeds,epochs):
    rows=[]
    for seed in seeds:
        traces=[torch.load(root/f'seed_{seed}'/f'gradient_epoch_{t}.pt',weights_only=True)
                for t in range(1,epochs+1)]
        ids=traces[0]['ids']
        labels=traces[0]['labels'].numpy()
        assert all(torch.equal(t['ids'],ids) and np.array_equal(t['labels'].numpy(),labels) for t in traces)
        rng=np.random.default_rng(seed+4000)
        # Stratification is evaluation-only. Selection receives neither these labels nor purity.
        anchors=sorted(int(x) for label in range(10)
                       for x in rng.choice(np.flatnonzero(labels==label),5,replace=False))
        historical=int(torch.nonzero(ids==23470).item())
        if historical not in anchors:
            anchors.append(historical)
        (root/f'seed_{seed}'/'evaluation_anchors.json').write_text(
            json.dumps([int(ids[p]) for p in anchors]),encoding='utf-8')
        for noise in (0.,1.,3.):
            observations=[]
            for t,trace in enumerate(traces):
                g=trace['gradients'].clone()
                if noise:
                    gen=torch.Generator().manual_seed(seed*1000+t+7000)
                    g+=torch.randn(g.shape,generator=gen)*g.square().mean().sqrt()*noise
                observations.append(g)
            for mode in ('legacy_l1','cosine'):
                score=[similarity_scores(g,anchors,mode) for g in observations]
                for window in range(2,epochs+1):
                    methods=dict(single_first=score[0],single_last=score[window-1],
                                 multi_mean=torch.stack(score[:window]).mean(0))
                    for rate in (.01,.05,.1):
                        count=int(len(ids)*rate)
                        for column,anchor in enumerate(anchors):
                            sets={name:select_positions(matrix[:,column],ids,count)
                                  for name,matrix in methods.items()}
                            for name,selected in sets.items():
                                chosen=selected.numpy()
                                reference=sets['single_last'].numpy()
                                overlap=len(np.intersect1d(chosen,reference))
                                # Exclude anchor from purity because anchor membership is tautological.
                                eligible=chosen[chosen!=anchor]
                                rows.append(dict(seed=seed,window=window,noise_rms_ratio=noise,
                                    similarity=mode,rate=rate,method=name,anchor_id=int(ids[anchor]),
                                    anchor_class=int(labels[anchor]),historical_anchor=int(anchor==historical),
                                    selected_count=count,purity=float((labels[eligible]==labels[anchor]).mean()),
                                    jaccard_last=overlap/(2*count-overlap)))
    with (root/'selection_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for seed in seeds:
        for noise in (0.,1.,3.):
            for mode in ('legacy_l1','cosine'):
                for window in range(2,epochs+1):
                    for rate in (.01,.05,.1):
                        part=[r for r in rows if r['seed']==seed and r['noise_rms_ratio']==noise
                              and r['similarity']==mode and r['window']==window and r['rate']==rate]
                        means={m:float(np.mean([r['purity'] for r in part if r['method']==m
                                               and not r['historical_anchor']])) for m in methods}
                        # 50 randomly chosen anchors are aggregated separately from the historical anchor.
                        summary.append(dict(seed=seed,noise_rms_ratio=noise,similarity=mode,window=window,
                                            rate=rate,**means,delta_last=means['multi_mean']-means['single_last'],
                                            delta_first=means['multi_mean']-means['single_first']))
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('ANALYSIS_COMPLETE',root,flush=True)


def run(root,seeds,epochs):
    root.mkdir(parents=True,exist_ok=True)
    tracked=[p for folder in ('attack','dataset','models','utils','scripts') for p in (ROOT/folder).glob('*.py')]
    tracked.append(ROOT/'main.py')
    (root/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in tracked},indent=2),encoding='utf-8')
    (root/'protocol.json').write_text(json.dumps(dict(seeds=seeds,epochs=epochs,primary_window=3,
        primary_rate=.1,primary_similarity='legacy_l1',noise='offline selection diagnostic only',
        label_use='evaluation only; 5 stratified random anchors per class; historical anchor separate',
        attack_training=False),indent=2),encoding='utf-8')
    entry.Trainer=CaptureTrainer
    for seed in seeds:
        output=root/f'seed_{seed}'
        if (output/f'gradient_epoch_{epochs}.pt').exists():
            continue
        flags=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0','--attack','LFBA',
               '--epoch',str(epochs),'--pretrain_stage',str(epochs+1),'--batch_size','128',
               '--seed',str(seed),'--early_stop','0','--lr','.001','--anchor_idx','23470',
               '--poison_rate','.1','--select_rate','.5','--print_steps','200','--results_dir',str(output)]
        print('CAPTURE_START',seed,time.strftime('%H:%M:%S'),flush=True)
        entry.main(parse_args(flags))
        print('CAPTURE_DONE',seed,time.strftime('%H:%M:%S'),flush=True)
    analyze(root,seeds,epochs)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'results'/'multiround_feasibility_20261030'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[100,101,102])
    parser.add_argument('--epochs',type=int,default=5)
    args=parser.parse_args()
    run(Path(args.output),args.seeds,args.epochs)
