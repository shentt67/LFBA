"""Training, validation and resumable two-party attack state."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from attack.attack import get_near_index
from dataset.utils import split_vfl
from utils.config import attack_start_epoch
from utils.checkpoint import VERSION, cpu_tree, capture_rng, restore_rng, atomic_save


class Trainer:
    def __init__(self, device, model_list, extractor_list, extractor, optimizer_list, criterion,
                 train_loader, test_loader, test_asr_loader, trigger_dimensions, logger,
                 args=None, checkpoint=None, val_loader=None, val_asr_loader=None, data_state=None):
        self.device=device
        self.model_list=model_list
        self.optimizer_list=optimizer_list
        self.criterion=criterion
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.test_asr_loader=test_asr_loader
        self.val_loader=val_loader
        self.val_asr_loader=val_asr_loader
        self.trigger_dimensions=list(trigger_dimensions)
        self.logger=logger
        self.args=args
        self.checkpoint=checkpoint
        self.data_state=data_state or {}
        self.poison_indexes=None
        self.grad_vec_epoch=None
        self.indexes_epoch=None
        self.target_epoch=None
        self.current_plan={}
        self.label_overrides={}
        self.best_score=None
        self.best_epoch=0
        self.best_metrics=None
        self.best_state_dict=None
        self.no_change=0
        self.history=[]
        self.completed_epoch=0
        self._static_ready=False

    def _loaders(self):
        return dict(train=self.train_loader,val=self.val_loader,val_asr=self.val_asr_loader,
                    test=self.test_loader,test_asr=self.test_asr_loader)

    def adjust_learning_rate(self,epoch):
        lr=self.args.lr * .1 ** (epoch // 20)
        for opt in self.optimizer_list:
            for group in opt.param_groups:
                group['lr']=lr

    def _apply_attack(self):
        self.train_loader.dataset.set_attack(self.current_plan,self.args,self.trigger_dimensions,self.label_overrides)

    def _configure_attack(self,ep):
        args=self.args
        ids=list(self.train_loader.dataset.ids)
        count=int(args.poison_rate*len(ids))
        if args.attack is None:
            return
        if args.attack in ('rsa','lra'):
            if not self._static_ready:
                recipients=np.random.permutation(ids)[:count].tolist()
                self.current_plan={int(i):None for i in recipients}
                self.label_overrides={int(i):args.target_label for i in recipients} if args.attack=='lra' else {}
                self._apply_attack()
                self._static_ready=True
            return
        if ep < attack_start_epoch(args):
            return
        if self.grad_vec_epoch is None:
            raise ValueError('LFBA requires previous-epoch gradients; checkpoint is incomplete')
        features=self.grad_vec_epoch
        sample_ids=self.indexes_epoch
        if self.poison_indexes is None:
            anchor_positions=torch.nonzero(sample_ids==args.anchor_idx).flatten()
            if len(anchor_positions)!=1:
                raise ValueError('anchor_idx must occur exactly once in the training split')
            anchor_pos=int(anchor_positions.item())
            near=get_near_index(features[anchor_pos],features,count,
                                getattr(args,'similarity','legacy_l1'),sample_ids)
            self.poison_indexes=sorted(sample_ids[near].tolist())
            # Ground-truth label is used for evaluation, never as an input to ranking.
            args.target_label=int(self.target_epoch[anchor_pos])
            self.logger.info('Constructed poison set: %d; target label: %d',len(self.poison_indexes),args.target_label)
        positions=torch.nonzero(torch.isin(sample_ids,torch.tensor(self.poison_indexes))).flatten()
        norms=torch.linalg.vector_norm(features[positions],ord=2,dim=1)
        if not torch.isfinite(norms).all():
            raise ValueError('Non-finite hard-sample gradients')
        selected_count=int(len(self.poison_indexes)*args.select_rate)
        if args.random_select:
            selected=random.sample(self.poison_indexes,selected_count)
        else:
            by_id=torch.argsort(sample_ids[positions],stable=True)
            order=by_id[torch.argsort(norms[by_id],descending=True,stable=True)]
            selected=sample_ids[positions[order[:selected_count]]].tolist()
        if args.poison_all:
            recipients=selected if args.random_select else self.poison_indexes
            self.current_plan={int(i):None for i in recipients}
        else:
            available=sorted(set(ids)-set(self.poison_indexes))
            if selected_count>len(available):
                raise ValueError('Not enough clean donor samples for replacement')
            donors=random.sample(available,selected_count)
            self.current_plan={int(i):int(j) for i,j in zip(selected,donors)}
        self.label_overrides={}
        self._apply_attack()
        self.logger.info('Epoch %d attack recipients: %d',ep+1,len(self.current_plan))

    def _restore(self):
        ck=self.checkpoint
        if not ck:
            return
        if ck.get('format_version')!=VERSION:
            raise ValueError('Legacy checkpoint lacks full state and cannot be strictly resumed')
        for model,state in zip(self.model_list,ck['state_dict']):
            model.load_state_dict(state)
        for opt,state in zip(self.optimizer_list,ck['optimizer']):
            opt.load_state_dict(state)
        self.completed_epoch=ck['epoch']
        for key in ('best_score','best_epoch','best_metrics','best_state_dict','no_change','history'):
            setattr(self,key,cpu_tree(ck[key]))
        attack=ck['attack_state']
        for key in ('poison_indexes','grad_vec_epoch','indexes_epoch','target_epoch','current_plan','label_overrides'):
            setattr(self,key,cpu_tree(attack[key]))
        self._static_ready=attack['static_ready']
        self.args.target_label=ck['config']['target_label']
        self._apply_attack()
        restore_rng(ck['rng_state'],self._loaders())

    def _snapshot(self):
        return dict(format_version=VERSION,epoch=self.completed_epoch,
                    state_dict=cpu_tree([m.state_dict() for m in self.model_list]),
                    optimizer=cpu_tree([o.state_dict() for o in self.optimizer_list]),
                    best_score=self.best_score,best_epoch=self.best_epoch,best_metrics=self.best_metrics,
                    best_state_dict=self.best_state_dict,no_change=self.no_change,history=self.history,
                    config=cpu_tree(vars(self.args)),trigger_dimensions=self.trigger_dimensions,
                    data_state=self.data_state,rng_state=capture_rng(self._loaders()),
                    attack_state=dict(poison_indexes=self.poison_indexes,grad_vec_epoch=self.grad_vec_epoch,
                        indexes_epoch=self.indexes_epoch,target_epoch=self.target_epoch,current_plan=self.current_plan,
                        label_overrides=self.label_overrides,static_ready=self._static_ready))

    def train(self):
        if self.val_loader is None or self.val_asr_loader is None:
            raise ValueError('Independent validation loaders are required')
        directory=Path(self.args.results_dir)
        directory.mkdir(parents=True,exist_ok=True)
        self._restore()
        if not self.checkpoint and self.args.start_epoch:
            raise ValueError('start_epoch requires a complete resume checkpoint')
        started=time.perf_counter()
        for ep in range(self.completed_epoch,self.args.epoch):
            epoch_start=time.perf_counter()
            for model in self.model_list:
                model.train()
            self._configure_attack(ep)
            grads,indices,labels=[],[],[]
            total=correct=0
            loss_sum=0.
            for step,(x,_,y,index) in enumerate(self.train_loader):
                x=x.to(self.device,dtype=torch.float32)
                y=y.to(self.device,dtype=torch.long)
                parts=split_vfl(x,self.args)
                local=[model(part) for model,part in zip(self.model_list[1:],parts)]
                if self.args.attack=='LFBA':
                    local[self.args.attack_client_num].retain_grad()
                logits=self.model_list[0](local)
                if len(y)==0 or bool(((y<0)|(y>=logits.shape[1])).any()):
                    raise ValueError('Training labels are outside the model class range')
                loss=self.criterion(logits,y)
                if not torch.isfinite(loss):
                    raise ValueError('Training loss is not finite')
                for opt in self.optimizer_list:
                    opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.attack=='LFBA':
                    grads.append(local[self.args.attack_client_num].grad.detach().cpu())
                    indices.append(index.detach().cpu())
                    labels.append(y.detach().cpu())
                for opt in self.optimizer_list:
                    opt.step()
                total+=len(y)
                correct+=int((logits.argmax(1)==y).sum().item())
                loss_sum+=float(loss.item())*len(y)
                if step % self.args.print_steps==0:
                    self.logger.info('Epoch %d batch %d/%d loss %.4f train_acc %.4f',
                                     ep+1,step+1,len(self.train_loader),loss_sum/total,correct/total)
            if total==0:
                raise ValueError('Empty training split')
            if grads:
                self.grad_vec_epoch=torch.cat(grads)
                self.indexes_epoch=torch.cat(indices)
                self.target_epoch=torch.cat(labels)
            self.adjust_learning_rate(ep+1)
            metrics=self.test(ep,split='val')
            score=metrics['acc']
            if self.args.attack is not None:
                score=None if score is None or metrics['asr'] is None else (score+metrics['asr'])/2
            improved=score is not None and (self.best_score is None or score>self.best_score)
            if improved:
                self.best_score=score
                self.best_epoch=ep+1
                self.best_metrics=metrics
                self.best_state_dict=cpu_tree([m.state_dict() for m in self.model_list])
                self.no_change=0
            elif ep>=attack_start_epoch(self.args):
                self.no_change+=1
            self.history.append(dict(epoch=ep+1,train_loss=loss_sum/total,train_acc=correct/total,
                                     validation=metrics,score=score))
            self.completed_epoch=ep+1
            snapshot=self._snapshot()
            atomic_save(snapshot,directory/'last_checkpoint.pth.tar')
            if improved:
                atomic_save(snapshot,directory/'best_checkpoint.pth.tar')
            # Restore a best checkpoint in a new output directory even without later improvements.
            if self.best_state_dict is not None and not (directory/'best_checkpoint.pth.tar').exists():
                evaluation_snapshot=dict(snapshot,state_dict=self.best_state_dict,epoch=self.best_epoch,
                                         evaluation_only=True)
                atomic_save(evaluation_snapshot,directory/'best_checkpoint.pth.tar')
            self.logger.info('End epoch %d val=%s best_epoch=%d score=%s seconds=%.2f',
                             ep+1,json.dumps(metrics),self.best_epoch,self.best_score,time.perf_counter()-epoch_start)
            if self.args.early_stop and self.no_change>=self.args.early_stop:
                break
        self.logger.info('Training seconds: %.2f',time.perf_counter()-started)
        return self.history

    @torch.no_grad()
    def test(self,ep=0,split='test'):
        clean_loader=self.val_loader if split=='val' else self.test_loader
        poison_loader=self.val_asr_loader if split=='val' else self.test_asr_loader
        if clean_loader is None or poison_loader is None:
            raise ValueError('Missing '+split+' loaders')
        for model in self.model_list:
            model.eval()
        total=correct=target_total=target_correct=0
        poison_total=poison_correct=asr_total=asr_correct=0
        loss_sum=0.
        target=self.args.target_label
        for poisoned,loader in ((False,clean_loader),(True,poison_loader)):
            for x,_,y,_ in loader:
                x=x.to(self.device,dtype=torch.float32)
                y=y.to(self.device,dtype=torch.long)
                logits=self.model_list[0]([m(p) for m,p in zip(self.model_list[1:],split_vfl(x,self.args))])
                if bool(((y<0)|(y>=logits.shape[1])).any()) or not 0<=target<logits.shape[1]:
                    raise ValueError('Evaluation label or target_label outside model class range')
                predicted=logits.argmax(1)
                if poisoned:
                    poison_total+=len(y)
                    poison_correct+=int((predicted==y).sum().item())
                    eligible=y!=target
                    asr_total+=int(eligible.sum().item())
                    asr_correct+=int(((predicted==target)&eligible).sum().item())
                else:
                    total+=len(y)
                    correct+=int((predicted==y).sum().item())
                    mask=y==target
                    target_total+=int(mask.sum().item())
                    target_correct+=int(((predicted==y)&mask).sum().item())
                    loss_sum+=float(self.criterion(logits,y).item())*len(y)
        counts=dict(acc=total,poison_acc=poison_total,asr=asr_total,target_acc=target_total)
        numerators=dict(acc=correct,poison_acc=poison_correct,asr=asr_correct,target_acc=target_correct)
        metrics={key:numerators[key]/count if count else None for key,count in counts.items()}
        metrics.update(loss=loss_sum/total if total else None,counts=counts,
                       undefined={key:'no eligible samples' for key,count in counts.items() if not count})
        self.logger.info('%s epoch %d metrics: %s',split,ep+1,json.dumps(metrics))
        return metrics
