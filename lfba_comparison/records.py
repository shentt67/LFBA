"""Validate saved comparison records without loading a model checkpoint."""
from pathlib import Path

from lfba_comparison.common import digest, read_json
from lfba_comparison.policy import schedule
from utils.artifacts import load_experiment_data


def epoch_rows(data):
    rows = []
    for item in data.get('audit', []):
        row = {k: v for k, v in item.items() if k != 'validation'}
        if 'validation' in item:
            row.update({'val_' + k: item['validation'].get(k)
                        for k in ('acc', 'asr', 'poison_acc')})
        rows.append(row)
    return rows


def report_schedule(method, epochs, count, select_rate=.5):
    """Read historical budgets without enabling historical training methods."""
    if method == 'lfba_matched':
        total=(epochs-3)*int(count*select_rate)
        base,remainder=divmod(total,epochs-1)
        return [0]+[base+int(i<remainder) for i in range(epochs-1)]
    if method in ('auto_single','auto_mean'):
        return [0]*3+[int(count*select_rate)]*(epochs-3)
    return schedule(method,epochs,count,select_rate)


def normalize_result(result, choice):
    result=dict(result)
    result.setdefault('candidate_digest',digest(choice['candidates']))
    if not isinstance(result.get('candidate_purity'),dict):
        result['candidate_purity']=dict(full=result.get('candidate_purity'),
            full_denominator=len(choice['candidates']),without_anchor=None,without_anchor_denominator=None)
    result.setdefault('recipient_purity',result.get('replacement_purity'))
    result.setdefault('first_prefix',(result.get('prefix') or [None])[0])
    result.setdefault('timing',dict(shared_first_round=result.get('shared_clean_seconds',0),
                                  training_seconds=result.get('training_seconds',0)))
    return result


def verified_group(output, method, epochs, seed, select_rate=.5, strict=False):
    output = Path(output)
    result = read_json(output / 'metrics.json')
    data = load_experiment_data(output)
    proof = ((result or {}).get('independent_evaluation') or
             read_json(output / 'independent_evaluation.json'))
    choice = data.get('choice')
    if (not result or not proof or not choice or result.get('method') != method
            or result.get('seed') != seed or result.get('epoch') != epochs
            or proof.get('epoch') != epochs or proof.get('test') != result.get('test')
            or proof.get('final_model_digest') != result.get('final_model_digest')):
        raise ValueError(f'Incomplete or unverified saved result: {seed}/{method}')
    result=normalize_result(result,choice)
    if result['candidate_digest'] != digest(choice['candidates']):
        raise ValueError('Saved candidate digest differs')
    rows = epoch_rows(data)
    expected = report_schedule(method, epochs, len(choice['candidates']), select_rate)
    if (len(data.get('history', [])) != epochs or len(rows) != epochs
            or [int(r['epoch']) for r in rows] != list(range(1, epochs + 1))
            or [int(r['recipient_count']) for r in rows] != expected
            or [int(r['cumulative_replacements']) for r in rows] !=
                [sum(expected[:i+1]) for i in range(epochs)]
            or result['cumulative_replacements'] != sum(expected)):
        raise ValueError('Budget audit failed')
    clean = 1 if method in ('lfba_original','lfba_matched') else 3
    if (any(int(r['candidate_count']) != 0 for r in rows[:clean]) or
            any(r['candidate_digest'] != result['candidate_digest'] or
                int(r['candidate_count']) != len(choice['candidates']) for r in rows[clean:])):
        raise ValueError('Frozen selection audit failed')
    if strict:
        config = read_json(output / 'config.json')
        if (not config or result.get('status') != 'complete'
                or not result.get('final_model_digest')
                or config.get('seed') != seed or config.get('comparison_method') != method
                or config.get('anchor_idx') != result.get('anchor')
                or config.get('target_label') != result.get('target_class')
                or choice.get('anchor_id') != result.get('anchor')
                or not isinstance(config.get('trigger_dimensions'), list)):
            raise ValueError('Missing or inconsistent saved configuration')
        train_ids = config.get('data_state', {}).get('train_ids')
        if not isinstance(train_ids, list) or not train_ids:
            raise ValueError('Missing saved training identities')
        train = set(train_ids)
        candidates = set(choice['candidates'])
        if len(candidates) != len(choice['candidates']) or not candidates <= train:
            raise ValueError('Invalid saved candidates')
        plans = data.get('plans', [])
        if [p.get('epoch') for p in plans] != list(range(clean + 1, epochs + 1)):
            raise ValueError('Missing saved replacement plans')
        for plan in plans:
            count = expected[plan['epoch'] - 1]
            recipients, donors = plan.get('recipients', []), plan.get('donors', [])
            if (len(recipients) != count or len(donors) != count
                    or len(set(recipients)) != count or len(set(donors)) != count
                    or not set(recipients) <= candidates or not set(donors) <= train - candidates):
                raise ValueError('Invalid saved replacement plan')
    return result, data, rows


def verified_pair(directory, epochs, seed, poison_rate, select_rate, anchor_source='auto_vote', fixed_anchor=None):
    """Validate shared setup and the protocol-specific anchor rules for two completed arms."""
    directory=Path(directory)
    records={}
    configs={}
    for method in ('auto_vote','lfba_original'):
        record,_,_=verified_group(directory/method,method,epochs,seed,select_rate,strict=True)
        records[method]=record
        configs[method]=read_json(directory/method/'config.json')
        if configs[method].get('poison_rate')!=poison_rate or configs[method].get('select_rate')!=select_rate:
            raise ValueError('Paired poison/select rate differs from protocol')
    auto,base=records['auto_vote'],records['lfba_original']
    shared=('initial_digest','data_digest','first_prefix','start_state')
    if anchor_source=='auto_vote':shared=('anchor','target_class')+shared
    elif anchor_source=='fixed':
        from lfba_comparison.policy import ANCHORS
        fixed_anchor=ANCHORS['CIFAR10'] if fixed_anchor is None else fixed_anchor
        if base['anchor']!=fixed_anchor:raise ValueError('Original LFBA fixed anchor differs')
    else:raise ValueError('Unknown paired anchor protocol')
    for key in shared:
        if key not in auto or key not in base or auto[key]!=base[key]:
            raise ValueError('Paired '+key+' differs')
    for key in ('dataset','seed','epoch','batch_size','lr','client_num','attack_client_num',
                'validation_rate','data_dir','train_limit','eval_limit','data_state','trigger_dimensions'):
        if configs['auto_vote'].get(key)!=configs['lfba_original'].get(key):
            raise ValueError('Paired configuration '+key+' differs')
    if configs['auto_vote'].get('pretrain_stage')!=3 or configs['lfba_original'].get('pretrain_stage')!=1:
        raise ValueError('Paired poisoning start differs from declared schedule')
    if anchor_source=='fixed':
        for key in ('acc','poison_acc'):
            if auto['test'].get('counts',{}).get(key)!=base['test'].get('counts',{}).get(key):
                raise ValueError('Paired full-test evaluation denominators differ')
    if anchor_source=='auto_vote' and auto['test'].get('counts')!=base['test'].get('counts'):
        raise ValueError('Paired evaluation denominators differ')
    return records
