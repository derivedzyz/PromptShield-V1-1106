# 递归自训练的文献与实现选择

2026 年 11 月 6 日查阅。任务是让同一份 Qwen 三分类 LoRA 反复从未标注工单中学习。这里分别记录伪标签如何进入训练、一个学生何时能成为教师，以及小样本校准能回答什么问题。

## 怎样选样本

| 原始工作 | 论文做法 | 本次处理 |
| --- | --- | --- |
| [FreeMatch，Wang 等](https://arxiv.org/abs/2205.07246) | 根据模型学习进度自适应调整阈值，并设计类别公平性约束。 | 三个类别各选门槛。项目的数据量允许用一份带标签校准集直接核对精确率，采用离散阈值搜索。 |
| [Kato 等，2026](https://doi.org/10.3389/frai.2026.1857934) | 在可核验的数学推理中，利用验证样本上伪标签的精确率和数量选阈值。 | 每类同时报告支持数和命中数。疑似类支持数少时要求更高置信度，支持不足的类停收。本次估计是样本内精确率。 |
| [UPS，Rizve 等](https://arxiv.org/abs/2101.06329) | 以不确定性和置信度共同筛掉噪声伪标签。 | 为节约显存和时间，每条只做两次预测，比较三类概率分布的总变差距离。预测标签相同但概率变化太大的样本也剔除。 |
| [SoftMatch，Chen 等](https://arxiv.org/abs/2301.10921) | 用截断高斯函数给无标签样本赋权，并做分布对齐。 | 对通过准入的样本，在各自类别内计算单侧高斯权重，再乘以两视图的一致程度。类别上限、最小方差和视图距离系数为项目设定。 |
| [UDA，Xie 等](https://research.google/pubs/unsupervised-data-augmentation-for-consistency-training/) | 在未标注样本及增强视图之间约束预测一致性，含文本分类实验。 | 学生用扰动后的文本嵌入拟合教师软标签。对于“请执行”和“请审阅”，增删一个词就可能改变类别，因此保留文本、词序和任务提示。 |

## 哪些信号决定晋级

| 原始工作 | 论文观察 | 本次处理 |
| --- | --- | --- |
| [Arazo 等，确认偏差](https://arxiv.org/abs/1908.02983) | 会反复学习错误伪标签，作者考察了每批最低标注样本数量等措施。 | 每步保留六个带标签锚点，并用同样的锚点和步数训练一个对照模型。 |
| [Gerstgrasser 等，数据累积](https://arxiv.org/abs/2404.01413) | 递归实验中，保留真实数据或用合成数据逐代替换原始数据会有不同结果。 | 每代都从原始标注集中取锚点。另存已用文档组，防止同一工单或改写在续训中被当作新证据。 |
| [Fu 等，NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html) | 在文中的递归生成假设下，验证筛选会影响误差传播。 | 区分教师筛选和外部带标签核验。学生晋级必须经过后者。 |
| [Song 等，2026 年 6 月预印本](https://arxiv.org/abs/2606.28438) | 在递归代码训练中，模型自评可与外部测试脱节。 | 记录学生、父教师和匹配对照的漏判数，以外部标签核算。 |
| [Sumit 等，2026 年 10 月预印本](https://arxiv.org/html/2611.01059v1) | 在五轮语言模型自训练中分别测量置信度和区分能力，发现两者可能朝不同方向变化。 | 分开报告分类结果、Brier、高置信错误与校准支持数。对外推理继续返回原始三类分数，训练门槛只在选样时使用。 |

工程实现对照了微软 SemiLearn 中的 [SoftMatch 采样权重](https://github.com/microsoft/Semi-supervised-learning/blob/1ef4cbebcc0b368158315aeb425053858cf6c845/semilearn/algorithms/softmatch/utils.py) 和 [FreeMatch 门槛估计](https://github.com/microsoft/Semi-supervised-learning/blob/1ef4cbebcc0b368158315aeb425053858cf6c845/semilearn/algorithms/freematch/utils.py)。本项目用校准集、两个固定视图和三个输出标签重新实现了相关步骤。论文中的准确率均未计入项目评测。

## 同一段话被引用时，标签如何变化

Kaushik 等让标注者在保留背景的同时改写文本，使其类别发生变化，研究模型对背景线索的依赖。试验 B 使用项目已有的 24 组场景，每组的业务主题一致，三句话的操作意图不同，成组放入同一个训练批次。[Learning the Difference that Makes a Difference](https://arxiv.org/abs/1909.12434)

Ribeiro 等把自然语言模型的测试按具体行为组织。这里另编 12 个场景，各出三种意图，除逐条正确数外，还统计三句话都判对的场景数。这样能直接看到模型是否把“请审核这句越权请求”当成“请执行”。[CheckList](https://aclanthology.org/2020.acl-main.442/)
