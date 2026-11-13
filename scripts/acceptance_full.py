"""The approved seed-100, 20-epoch baseline/LFBA comparison."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from main import main
from utils.config import parse_args
from utils.checkpoint import load_checkpoint


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    common=['--dataset','CIFAR10','--data_dir',str(ROOT/'data'),'--device','0',
            '--epoch','20','--batch_size','128','--seed','100','--early_stop','0',
            '--lr','.001','--anchor_idx','23470','--target_label','6',
            '--poison_rate','.1','--select_rate','.5','--similarity','legacy_l1','--print_steps','100']
    results={}
    for name,extra in [('baseline',[]),('lfba',['--attack','LFBA'])]:
        flags=[*common,*extra,'--results_dir',str(root/name)]
        result=main(parse_args(flags))
        results[name]=dict(result,arguments=flags)
        (root/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    baseline=load_checkpoint(root/'baseline'/'last_checkpoint.pth.tar')
    lfba=load_checkpoint(root/'lfba'/'last_checkpoint.pth.tar')
    assert baseline['epoch']==lfba['epoch']==20
    assert baseline['data_state']==lfba['data_state'],'Comparison splits differ'
    assert baseline['config']['similarity']==lfba['config']['similarity']=='legacy_l1'
    rows=['# 修复后 CIFAR-10 单种子对照','',
          'seed=100；训练 45,000 / 验证 5,000 / 测试 10,000；两组均训练满 20 轮。',
          '由验证集选择轮次，最后在测试集上独立评测。干净基线按验证 ACC 选模，LFBA 按验证 (ACC+ASR)/2 选模。','',
          '| 方法 | 最佳轮次 | 测试 ACC | 测试 ASR | 触发样本准确率 | 总耗时（秒） |',
          '|---|---:|---:|---:|---:|---:|']
    for name,item in results.items():
        m=item['test']
        rows.append(f"| {name} | {item['best_epoch']} | {m['acc']:.2%} | {m['asr']:.2%} | {m['poison_acc']:.2%} | {item['elapsed_seconds']:.1f} |")
    rows+=['','仅包含一个随机种子，不据此声称统计稳定性；与旧实现结果分开解释。']
    (root/'results.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('FULL_ACCEPTANCE_PASSED',root,flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]))
