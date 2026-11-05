---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags:
- prompt-injection
- lora
language:
- zh
- en
---
# PromptShield-V1-1105

提示词注入检测 LoRA。输入归一化后的文本，输出正常、明确恶意、疑似风险三类模型分数。该版本为递归自训练中通过验证的第一代。基座权重由 Transformers 按 report.json 指定的官方修订下载。

在安装 PyTorch、transformers==5.5.0 和 safetensors 的环境中下载本仓库全部文件，在仓库目录运行：

~~~python
from inference import Detector
model = Detector(device="cpu")
print(model.classify("图书馆周末几点开门？"))
~~~

模型使用 256 token 窗口，长输入保留首尾。分数未经概率校准。CPU 单条推理在项目选取的 855 条 NeurAlchemy 公开测试样本上答对 842 条，在 72 条项目场景样本上答对 69 条。编码解包及业务权限控制由调用方处理。

基座：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。公开训练数据包括 [deepset](https://huggingface.co/datasets/deepset/prompt-injections) 与 [NeurAlchemy](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)，另包含项目自编样本及经筛选的软标签。
