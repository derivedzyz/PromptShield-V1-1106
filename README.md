---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags:
- prompt-injection
- lora
- semi-supervised-learning
language:
- zh
- en
---
# PromptShield-V1-1106

PromptShield 把归一化后的中英文输入分为正常请求、明确恶意、疑似风险三类。调用方负责解码、上下文隔离及后续处置。模型采用 Qwen2.5-0.5B-Instruct 和 Q/V LoRA，训练参数 540,672 个。输入窗口为 256 个 token，长文本保留首尾，返回三类原始分数。

## 取哪份权重

| 目录 | 用途 | 适配器 SHA256 前 12 位 |
| --- | --- | --- |
| 根目录 | 正式推理。沿用 1105 已通过晋级检查的教师权重 | 6ce4ed9c6c82 |
| experimental/ | 11 月 6 日新训练的候选。按晋级集成绩选出，供复核及试用 | cc3a9a4a926f |

本次发布更新了递归训练器，并保留新权重供独立试验。新候选在 324 条晋级样本中答对 321 条，原教师及匹配对照均为 320 条。候选将恶意漏判从 1 条降至 0 条，但分组平均召回仅由 0.93583 升到 0.93750，温度调整后的 Brier 从 0.02171 降到 0.02098。两项变化均低于预先设定的晋级幅度（分别为 0.002 和 0.003），因此两个版本分目录保存。

## 留出集复测

以下为 RTX 5060 Laptop GPU、BF16、单条推理结果。三个模型分别从磁盘加载。四个历史集合共 1,055 条，另有 36 条在本轮训练开始前编写的中英文业务检查。训练和选版结束后一次性评分。

| 集合 | 原始 Qwen 基座 | 1105 正式权重 | 1106 新候选 |
| --- | ---: | ---: | ---: |
| NeurAlchemy 公开子集，855 条 | 516 | 842 | 843 |
| deepset 公开子集，116 条 | 60 | 112 | 111 |
| 既有业务场景，72 条 | 24 | 68 | 68 |
| 项目三分类样本，12 条 | 4 | 12 | 12 |
| 本轮新增场景，36 条 | 12 | 31 | 31 |

新增场景中，两份适配器都识别出 12/12 条要求实际执行的越权请求，识别出 8/12 条引用攻击进行审阅的请求。两者均有 7/12 个业务场景三种意图全部答对。1105 早先报告使用 CPU，表内各列均采用本次 GPU 复测值。完整混淆矩阵和数据指纹在 [results](results) 目录。

## 使用模型

下载本仓库，在正式权重目录或 experimental 目录中启动 Python。实测依赖为 PyTorch 2.13.0、Transformers 5.5.0 和 safetensors 0.8.0。首次运行按 report.json 内固定的修订号下载基座。

~~~python
from inference import Detector

detector = Detector(device="cpu")  # 支持 BF16 的 NVIDIA 显卡可改为 cuda
print(detector.classify("请审核这封要求导出客户资料的来信。"))
~~~

## 递归训练模块

三类分别用带标签校准样本选择门槛。本次教师对正常、恶意、疑似采用 0.82、0.82、0.94，通过门槛的校准样本分别判对 68/70、66/66、7/7。学生根据两次预测的一致程度和同类样本的置信度学习软标签。续训时先剔除父权重实际学过的文档组。

在两种训练方式下分别运行两个种子，每次看 1,024 条候选，采纳 96 条软标签，每类 32 条，训练 48 步；每次另训一个监督对照。成组方式使用 24 组已标注的“正常请求、要求执行、要求审阅”场景。四次试验均在第一代验证后停止，训练记录同时保留通过与未通过的条件。

- [训练说明与数据格式](training/recursive/TRAINING.md)
- [12 篇原始论文与具体实现选择](training/recursive/RESEARCH-1106.md)
- [基础试验约定](training/recursive/PROTOCOL-1106.md) · [成组取样试验约定](training/recursive/CONTRAST-1106.md)
- [四次晋级记录与对照](results/trials.json) · [来源组和重复检查](results/data-audit.json)

基座及公开数据来源为 [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[deepset](https://huggingface.co/datasets/deepset/prompt-injections) 和 [NeurAlchemy](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)。此前的 1105 发布保存在两个仓库的 v1-1105 标签。
