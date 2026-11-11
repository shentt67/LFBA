"""Pure configuration and feasibility checks for paired stability experiments."""
import copy
import math

DEFAULT_POISON_RATES = [.01,.05,.1,.2]
DEFAULT_SELECT_RATES = [.3,.5,.7]
DEFAULT_SEEDS = [2004,592,3383,100,101]
BASELINE = dict(poison_rate=.1,select_rate=.5)


def validate_budget(train_size,poison_rate,select_rate,epochs=40):
    if (not math.isfinite(poison_rate) or not 0<poison_rate<=1 or
            not math.isfinite(select_rate) or not 0<select_rate<=1):
        raise ValueError('Poison and select rates must be finite and within (0,1]')
    count=math.floor(train_size*poison_rate)
    replacements=math.floor(count*select_rate)
    if count<2:raise ValueError('At least two candidates are required')
    if replacements<1:raise ValueError('At least one replacement per poisoned epoch is required')
    if replacements>train_size-count:raise ValueError('Not enough clean donors')
    if 2*count+1>train_size:raise ValueError('Automatic selection needs at least 2P+1 samples')
    return dict(candidates=count,replacements_per_epoch=replacements,
                budgets=dict(lfba_original=(epochs-1)*replacements,auto_vote=(epochs-3)*replacements))


def train_size(options):
    return 450 if options.smoke else (45000 if options.dataset=='CIFAR10' else 6616)


def build_cases(options):
    pairs=[(.1,.5)]+[(p,.5) for p in options.poison_rates]+[(.1,s) for s in options.select_rates]
    seen=set();cases=[]
    for p,s in pairs:
        if (p,s) in seen:continue
        seen.add((p,s))
        identifier=f'p_{float(p)!r}_s_{float(s)!r}'.replace('.','p')
        cases.append(dict(id=identifier,poison_rate=p,select_rate=s))
    return cases


def case_options(options,case):
    child=copy.copy(options)
    child.poison_rate=case['poison_rate'];child.select_rate=case['select_rate']
    child.output=options.output/case['id']
    child.anchor_source=options.anchor_source;child.stability=True
    child.continue_on_error=True;child.defer_cleanup=True
    return child


def stability_protocol(options):
    from lfba_comparison.policy import ANCHORS
    source=getattr(options,'anchor_source','fixed')
    protocol=dict(version=1 if source=='auto_vote' else 2,kind='stability',dataset=options.dataset,seeds=list(options.seeds),
        smoke=options.smoke,epochs=5 if options.smoke else 40,
        poison_rates=list(options.poison_rates),select_rates=list(options.select_rates),
        baseline=dict(BASELINE),cases=build_cases(options),data_dir=str(options.data_dir),
        device=options.device,block_size=options.block_size,batch_size=128,optimizer='Adam',
        lr=.001,early_stop=False,lr_decay=False,methods=['auto_vote','lfba_original'],
        anchor_source=source,train_source_limit=500 if options.smoke else None,
        eval_limit=100 if options.smoke else None,split_anchor_retention=source=='fixed',
        interpretation=('same-anchor paired strategies; original starts poisoning at 2, vote at 4' if source=='auto_vote' else
                        'full strategies: automatic anchor versus fixed LFBA anchor; original starts poisoning at 2, vote at 4'))
    if source=='fixed':protocol['fixed_lfba_anchor']=ANCHORS[options.dataset]
    return protocol
