# 用自己的数据续训

训练器读取本仓库根目录的教师权重和来源记录。训练要求支持 BF16 的 NVIDIA 显卡。本次环境为 PyTorch 2.13.0+cu130、Transformers 5.5.0 和 safetensors 0.8.0。普通 CPU 可运行仓库根目录的推理示例。

## 数据文件

准备五份 UTF-8 JSONL，每行一个对象。模型用 0 表示正常、1 表示明确恶意、2 表示疑似或审阅攻击。

| 文件 | 每行字段与用途 |
| --- | --- |
| anchors.jsonl | `text`、整数 `label`。训练用人工标注，每类都要覆盖。 |
| calibration.jsonl | `text`、整数 `label`。单独采样，用于温度和分类别选样门槛。 |
| gate.jsonl | `text`、整数 `label`、字符串 `set`。按业务或来源分组进行晋级验证。 |
| pool.jsonl | `text` 和字符串 `group`。同一原始工单及其所有改写共用一个稳定组号。 |
| exclusion-texts.jsonl | `text`。保留测试集的文本，仅用于剔除重复，无需传入其标签。 |

划分时先按源文档分组，再划分训练、校准、晋级和测试数据。运行器核对文件指纹，再检查文本重叠及规范化后的相近文本。生产数据使用方负责提供一致的文档组号。可选的 `--contrast-anchors` 是 JSON 数组，每一项含 `group` 和 `texts`，三句话须按 0、1、2 的顺序逐字出现在锚点文件中。

## 运行

在仓库根目录执行以下命令。把基座下载到 `base` 目录，在此目录的 `REVISION` 文本文件内写入 `report.json` 中的官方提交号。将五份数据放到 `my-splits`。

```shell
python training/recursive/freeze_inputs.py my-splits
python training/recursive/run.py --base base --teacher . --data my-splits --out run-01 --adaptive --rounds 2 --seed 271828
python -m unittest discover -s training/recursive -p "test*.py"
```

输出目录使用新名字。每轮保存父教师、对照组和学生的验证成绩、样本数、伪标签权重和选择依据。只有通过全部条件的学生才会写入下一代教师路径；继续训练时读取 `trial.json` 里的 `active`。演示所用 24 组成组训练样本见 `training/round3/train-scenarios.json`，用于自己的锚点清单时须先核对样本归属。对于适配器根目录附带的 `source-groups.json`，训练器会校验它与权重、元数据之间的指纹。

试验方案、两阶段结果和数据文件指纹见同仓库的 `PROTOCOL-1106.md`、`CONTRAST-1106.md`、`results/trials.json`。公开包提供代码、场景和汇总成绩；第三方全量语料按其原数据集取得。新增检查集 `november-check.json` 用于验收，须与个人训练语料分开。
