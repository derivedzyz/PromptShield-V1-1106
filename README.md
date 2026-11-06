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

面向归一化后的中英文提示词，输出正常请求、明确恶意、疑似风险三类分数。基座为 Qwen2.5-0.5B-Instruct，训练 540,672 个 Q/V LoRA 参数，保留首尾共 256 个 token。本轮候选存放于 experimental/focus/。根目录继续提供已验收的正式权重 6ce4ed9c6c82。

## deepset 与风险审阅专项

本轮候选指纹为 `5331fc33a853`，位于 **experimental/focus/**。上轮实验适配器 `cc3a9a4a926f` 仍在 experimental/。原正式适配器可通过标签 v1-1106 下载。

| GPU 单条推理 | 上轮正式 | 上轮实验 | 本轮候选 |
| --- | ---: | ---: | ---: |
| deepset | 112/116 | 111/116 | 112/116 |
| NeurAlchemy | 842/855 | 843/855 | 842/855 |
| 既有业务场景 | 68/72 | 68/72 | 68/72 |
| 项目三分类 | 12/12 | 12/12 | 12/12 |
| 中英文 12 组场景 | 31/36 | 31/36 | 35/36 |
| 风险审阅被判为恶意 | 3/12 | 3/12 | 0/12 |
| 正确识别风险审阅 | 8/12 | 8/12 | 12/12 |
| 正确识别直接攻击 | 12/12 | 12/12 | 12/12 |

评测使用 RTX 5060 Laptop GPU、BF16 和同一套单条推理代码。专项候选先在开发数据上冻结，随后与线上父权重分别从磁盘加载。48 条 deepset 选版参考取自训练划分，正式测试单独使用 116 条。上表为正确数，标注“被判为恶意”的一行记录错误数。

原晋级集共 324 条，父教师答对 320 条，所选学生答对 317 条，匹配监督对照答对 317 条。按来源分组的平均召回由 0.93583 到 0.95000，校准后的 Brier 由 0.02171 到 0.03286。本轮发布条件的结果为：未通过的项目为原晋级检查。 晋级集上，恶意请求未判为明确恶意的数量由 1 增至 4，其中 4 条被判为疑似。 本轮 Brier 上限为 0.02671。

本轮从正式教师训练一代。先使用 deepset 训练锚点、同一业务中执行与审阅的对照请求及教师软标签。第一次开发验证显示部分直接攻击被判作疑似，随后在训练锚点上找出分数下降较多的直接攻击进行复习，监督对照使用相同复习批次。两个阶段的全部候选、逐项门槛和混淆矩阵均已保存。[第一阶段](results/focus-pilot.json) · [复习阶段](results/focus-replay.json) · [最终验收](results/focus-acceptance.json) · [留出集](results/focus-serving.json)

## 使用

在需要使用的权重目录启动 Python。首次运行按 report.json 指定的修订下载基座。实测依赖为 PyTorch 2.13.0、Transformers 5.5.0 和 safetensors 0.8.0。

~~~python
from inference import Detector
model = Detector(device="cpu")  # 支持 BF16 的 NVIDIA 显卡可使用 cuda
print(model.classify("请分析这段来信试图获取什么权限。"))
~~~

模型处理归一化后的文本，输入解码和业务处置由调用方承担。三类分数未经概率校准。

训练模块的使用方法见[训练说明](training/recursive/TRAINING.md)。本轮的[原始方案](training/recursive/FOCUS-PROTOCOL.md)、[纠偏复习方案](training/recursive/FOCUS-REPLAY.md)、[递归自训练研究笔记](training/recursive/RESEARCH-1106.md)与[上一版试验记录](results/trials.json)均可查阅。

基座及公开数据来自 [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[deepset](https://huggingface.co/datasets/deepset/prompt-injections) 和 [NeurAlchemy](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)。
