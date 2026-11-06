# 同一代学生的纠偏复习

第一阶段八个检查点均被原门槛拒绝。开发结果显示，学生把部分原先识别正确的直接攻击改判为疑似。第一阶段记录原样保留。正式测试集尚未运行；在测试前安排一次有限的监督复习，之后结束本轮选版。

固定使用第一阶段按开发规则选中的学生，以及从同一发布父模型、原种子和监督批次重建的对应对照。对照重建的晋级混淆矩阵和 Brier 必须与原记录一致。比较父教师与学生在训练锚点上的标签分数变化。每步复习两条容易遗忘的直接攻击、一条普通请求、一条 deepset 训练样本和两条场景样本。直接攻击和审阅样本都使用已知训练标签。两组接收完全相同的复习样本。这个阶段不产生新一代教师。

固定复习学习率为 0.000007 和 0.000018，各跑 64 步，在第 16、32、64 步分别评估完整更新和保留 65% 更新的插值版本。父模型、监督对照、晋级七项检查和最终切换条件沿用第一阶段。沿用相同开发排序冻结一个候选；之后只评测这份候选，不再根据测试成绩尝试替换其他检查点。

执行记录：第一次单独重建监督对照时未通过与原记录的一致性检查，程序在复习前停止。为避免把两次运行的权重配成一对，固定已选学习率、步数、样本和日程，在一个进程内重新成对训练学生与对照。保存双方权重，逐一从磁盘回读并核对各自的混淆矩阵及 Brier 后，再开始上述复习。两份回读数据和指纹写入 matched_replication，原始开发记录完整保留。

## 运行顺序

以下命令从模型仓库根目录执行。teacher-1105 指向标签 v1-1106 的根目录正式适配器，base 是 report.json 指定修订的本地 Qwen 基座，splits 是五份冻结 JSONL 和 manifest.json，deepset-train.json 来自 deepset 的 train 划分。每次执行均使用新的输出目录。数据指纹可从 results/focus-pilot.json 查验。

```shell
python training/recursive/focus_round.py --base base --teacher teacher-1105 --data splits --deepset-train deepset-train.json --extra training/recursive/november-check.json --out pilot
python training/recursive/focus_replay.py --base base --teacher teacher-1105 --data splits --deepset-train deepset-train.json --extra training/recursive/november-check.json --pilot pilot --out replay
```

学生与父模型分别使用 release_eval.py 做单条 GPU 评测。focus_acceptance.py 接收 trial、baseline、candidate 三份 JSON，重新计算七项原条件并核验固定测试文件、权重指纹及两套专项结果。
