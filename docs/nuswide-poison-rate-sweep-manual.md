# NUS-WIDE：固定 s=0.5 的投毒比例对照

本轮只比较 p=0.01、0.05、0.1，保持 40 轮、种子 2004／592／3383／100，以及原双方法协议。复用现有入口，无需修改训练代码，不启用只适用于 CIFAR10 的 `--stability`。

## 固定协议与预算

训练／验证／测试为 62969／6997／46693；batch 128、Adam、lr=0.001；无学习率衰减、无早停；GPU 0 串行。原始 LFBA 固定锚点 33930、首轮 legacy_l1 选样，第 2 轮起投毒；auto_vote 前三轮干净训练、自动选锚点及 cosine 投票，第 4 轮起投毒。两者均使用攻击方文本特征中按种子确定的 10 维触发器，赋值 1.0。最终统一测试第 40 轮并独立复评。

| p | P=floor(Np) | Q=floor(Ps) | LFBA 累计替换 | auto_vote 累计替换 | 状态 |
|---|---:|---:|---:|---:|---|
| 0.01 | 629 | 314 | 12246 | 11618 | 新增 8 分支 |
| 0.05 | 3148 | 1574 | 61386 | 58238 | 新增 8 分支 |
| 0.1 | 6296 | 3148 | 122772 | 116476 | 复用已完成 8 分支 |

合计分析 24 个分支，本次新增训练 16 个。p=0.1 使用原目录，不复制或重新训练。三个协议已核验，除 poison_rate 外其余字段一致；已存在的 p=0.1 协议与当前入口生成的协议完全一致，8 个已完成分支通过严格记录核验。

## PowerShell 初始化

```powershell
Set-Location -LiteralPath 'D:\MyHomework\2026_2\LFBA\second\LFBA'
$experimentPython = 'D:\MyHomework\AIAIAIAI\environment\python.exe'
$comparisonScript = '.\scripts\compare_lfba_auto_vote.py'
$fixedArgs = @(
    '--dataset', 'NUSWIDE', '--anchor-source', 'fixed',
    '--select-rate', '0.5', '--device', '0', '--data-dir', '.\data',
    '--seeds', '2004', '592', '3383', '100'
)
$rateCases = @(
    @{ Rate = '0.01'; Output = '.\results\nuswide_lfba_vs_vote_p001_s05_v1' },
    @{ Rate = '0.05'; Output = '.\results\nuswide_lfba_vs_vote_p005_s05_v1' },
    @{ Rate = '0.1';  Output = '.\results\nuswide_lfba_vs_vote_p01_s05_v1' }
)
```

## 预检

```powershell
foreach ($case in $rateCases) {
    & $experimentPython -B $comparisonScript @fixedArgs --poison-rate $case.Rate --output $case.Output --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed: p=$($case.Rate)" }
}
```

每组应显示 8 个任务、40 轮和表中的预算。首次启动时，前两组目录未占用；p=0.1 目录已占用是正常情况。

## 正式执行与续训

```powershell
foreach ($case in $rateCases) {
    & $experimentPython -B $comparisonScript @fixedArgs --poison-rate $case.Rate --output $case.Output --run --resume
    if ($LASTEXITCODE -ne 0) { throw "Experiment stopped: p=$($case.Rate); rerun this loop to resume." }
}
```

这段循环先运行 p=0.01 的四种子两方法，再运行 p=0.05，最后核验 p=0.1 并重建其报告。现有入口的 `--resume` 同时支持启动空目录和恢复已有目录；完成且通过验证的分支自动跳过。遇到错误停止整个循环，中断后执行同一段即可续训，不要更换输出目录或参数。

本次不自动启动真实训练，不增加额外短训练。先前已完成同入口的真实 NUS-WIDE 训练，新参数已完成预算和文件预检。

## 报告重建与比较口径

全部完成后，可只重建三组报告：

```powershell
foreach ($case in $rateCases) {
    & $experimentPython -B $comparisonScript @fixedArgs --poison-rate $case.Rate --output $case.Output --report-only
    if ($LASTEXITCODE -ne 0) { throw "Report failed: p=$($case.Rate)" }
}
```

此命令分别生成各组的 `results.md`、`summary.json` 和验证曲线，不会自动合并为跨 p 的总报告。三组结果完成后按以下口径统一分析：

- 每个 p：两方法 ACC、ASR、去锚点候选纯度、替换纯度的四种子均值及样本标准差。
- 每个 p、每个种子：auto_vote 减原始 LFBA 的配对差值，同时列出锚点、目标类、ASR 分母和实际预算。
- 跨 p：检查降低投毒强度后是否仍饱和、候选纯度是否变化，以及方法差值是否稳定。
- 保留持平和负向结果；不因结果改变种子、触发器或测试轮次。

auto_vote 的邻域大小与候选数由 p 决定，因此不同 p 下锚点及目标类别可能自然变化。本轮固定的是自动选择规则，不是强制固定其选择结果；应把这类变化记录下来。这是完整方法对 p 的敏感性实验，不能直接视作固定目标、固定候选集上的纯预算实验。

现有检查点清理规则不变：全部分支验证通过且报告成功后清理临时检查点；失败或中断时保留恢复状态。

预检记录：`note/nuswide-poison-rate-sweep-preflight-2026-11-06.json`。准备阶段未启动训练。
