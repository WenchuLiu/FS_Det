# FS_Det

基于 MMRotate 的 SAR/RGB 少样本目标检测项目。Base 阶段联合训练 SAR 与 RGB，微调阶段使用单一模态，默认 SAR。

| 方法 | 配置目录 | 说明 |
| --- | --- | --- |
| method1 | `configs/method1/` | 冻结 backbone |
| method2 | `configs/method2/` | 特征对齐 |
| method3 | `configs/method3/` | 文本对齐 |
| method4 | `configs/method4/` | Ours |
| method5 | `configs/method5/` | Ours2，增加路由迁移 |

每种方法提供 ConvNeXt Tiny/Large，以及 base、18-shot、60-shot、100-shot 配置。method 配置直接继承 `_base_` 中的模型、数据、训练计划和运行设置，不继承其他 method 配置。

配置分工：

- `fsdet_configs/schema.py`：数据路径、类别、文本名称、语义映射和样本数，不构建数据集。
- `configs/_base_/datasets/`：读取 schema，定义数据加载与 pipeline；`build_fewshot_config` 位于这里。
- `configs/_base_/models/`：读取 schema，选择 SAR 检测头、类别数和预训练设置。
- `configs/_base_/schedules/`：学习率、迭代数与评估间隔。
- `configs/method*/tiny或large/`：方法自己的模型和对齐参数，无 `if/else` 判断。

例如在 method 配置中写 `optimizer = dict(lr=5e-5)`，即可覆盖继承的学习率；SAR head 的公共参数在对应 `_base_/models` 文件中修改。

`_base_/models` 只保留 **9 个文件**：8 个公共模型配置和 1 个 RSCLIP 路径校验配置。

| 模型规模 | Base：SAR GFL | Base：SAR DOTA/ORCNN | SAR few-shot：GFL | SAR few-shot：DOTA/ORCNN |
| --- | --- | --- | --- | --- |
| Tiny | `fsdet_tiny_base_sar_gfl.py` | `fsdet_tiny_base_sar_dota.py` | `fsdet_tiny_finetune_sar_gfl.py` | `fsdet_tiny_finetune_sar_dota.py` |
| Large | `fsdet_large_base_sar_gfl.py` | `fsdet_large_base_sar_dota.py` | `fsdet_large_finetune_sar_gfl.py` | `fsdet_large_finetune_sar_dota.py` |

method1–5 共用这些文件，不再为每种方法复制模型配置。各方法自己的 `align_cfg`、`routing_transfer_cfg` 和对齐 head 类型在 method 文件中设置。Base 的 SAR head 使用 base 类别，SAR few-shot 使用全部类别。

模型文件根据 schema 中的 SAR 标注格式选择：`hbb_json` 对应 GFL，`dota` 对应 ORCNN。当前实验配置只提供 SAR few-shot；不再提供 RGB few-shot 或水平框两阶段模型方案。

以 method5 Tiny 的 SAR 18-shot 为例，继承关系为：

```python
from fsdet_configs.schema import *

_base_ = [
    '../../_base_/models/fsdet_tiny_finetune_sar_' + sar_model_format + '.py',
    '../../_base_/datasets/fsdet_18shot_sar.py',
    '../../_base_/schedules/fsdet_finetune_18_sar.py',
    '../../_base_/default_runtime.py',
]
```

也可把模型路径直接写成 `fsdet_tiny_finetune_sar_dota.py` 或 `fsdet_tiny_finetune_sar_gfl.py`，与 schema 的标注格式保持一致。60/100-shot 只替换数据和训练计划中的 `18`。

## Requirements

本项目已验证的环境：

- Python 3.9
- PyTorch 1.13.1 / torchvision 0.14.1
- mmcv-full 1.7.1
- mmdet 2.28.2

## Installation

先安装与机器 CUDA 版本匹配的 PyTorch，再在项目根目录执行：

```bash
pip install openmim
mim install mmcv-full==1.7.1
pip install mmdet==2.28.2
pip install -r requirements.txt
pip install -v -e .
```

### Pretrained models

Tiny backbone 权重放在：

```text
pretrained/convnext_tiny_22k_224_mmpretrain.pth
```

Large backbone 通过 `FSDET_CONVNEXT_LARGE_CKPT` 指定权重路径。method1 会冻结 backbone，应先准备预训练权重。

method3 默认使用本地 Hugging Face 格式的 RSCLIP，也可通过开关选择 SAR-CLIP；method4/5 使用 SAR-CLIP：

```bash
# method3
export FSDET_METHOD3_CLIP_MODEL_PATH=/path/to/RSCLIP

# method3 使用本地 SAR-CLIP 时，无需设置 RSCLIP 路径
export FSDET_METHOD3_TEXT_ENCODER=sarclip

# method4/5：默认读取仓库内该目录，也可指定其他目录
export FSDET_SARCLIP_DIR=/path/to/ViT-B-16
```

SAR-CLIP 目录需包含 `vit_b_16_model.pth`、`vocab.json`、`merges.txt`，默认位置为 `mmrotate/models/sar_clip/model_configs/ViT-B-16/`。
method3 的 SAR-CLIP 默认读取此仓库目录；自定义位置在 align_cfg 中设置 sar_clip_path 和 sar_clip_cache。生成扩展权重与训练时须使用同一个编码器开关；扩展工具保留文本投影层和温度参数，仅跳过文本编码器加载。

## Dataset Preparation

### Data layout

本工作目录的 `data` 已软链接到 `../detection/data`，其中包含 `SARDet_100K` 和 `DOTA_800pix` 等数据集。使用这些数据时，在 schema 中设置对应标注和图片路径。

SAR 默认使用 COCO JSON 水平框标注，RGB 使用 DOTA 文本旋转框标注。自定义数据可按以下目录组织：

```text
data/FS_Det/
├── SAR/
│   ├── annotations/
│   │   ├── base_train.json
│   │   ├── base_val.json
│   │   ├── test.json
│   │   ├── fewshot_18.json
│   │   ├── fewshot_60.json
│   │   └── fewshot_100.json
│   └── images/
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── fewshot_18/
│       ├── fewshot_60/
│       └── fewshot_100/
└── RGB/
    ├── train/
    │   ├── images/
    │   ├── annotations_base/
    │   ├── annotations_few_shot/
    │   └── split_manifest.json
    └── val/
        ├── images/
        └── annfiles/
```

数据根目录、标注路径、类别划分和样本数统一在 `fsdet_configs/schema.py` 中设置。RGB 默认用 val 数据进行测试；有独立测试集时，修改 `rgb_test_ann` 和 `rgb_test_img_root`。

### Custom dataset

编辑 `fsdet_configs/schema.py`：

| 配置 | 用途 |
| --- | --- |
| `sar_base_classes` / `sar_novel_classes` | SAR 标注类名与 base/novel 划分 |
| `rgb_classes` | RGB 标注类名 |
| `sar_base_text_classes` / `sar_novel_text_classes` | 手动填写的 SAR CLIP 文本类名 |
| `rgb_text_classes` | 手动填写的 RGB CLIP 文本类名 |
| `semantic_mapping` | SAR/RGB 语义对应关系，编号为各模态从 0 开始的本地标签位置 |

文本类名与标注类名按位置一一对应，可以使用不同名称，无需修改原始标注。检测头类别数由类别列表自动计算。

SAR 若使用 DOTA 旋转框，在训练、初始化和测试前统一设置：

```bash
export FSDET_SAR_ANN_FORMAT=dota
```

此时 SAR 自动使用 ORCNN，method3 使用会返回旋转 RoI 原型的对齐 head，默认目录为 `SAR/train/{images,annotations_base,annotations_few_shot}`、`SAR/val/{images,annfiles}` 和 `SAR/test/{images,annfiles}`。JSON 水平框使用 GFL。

DOTA 的 `difficulty=100` 是标注难度的过滤阈值：只排除难度值大于 100 的目标，因此通常保留难度为 0、1、2 的标注。它不是图像尺寸或百分比；设为 0 时只保留难度为 0 的目标。

### Few-shot annotations

SAR JSON few-shot 标注需提前准备。DOTA 数据可用以下脚本生成 base/support 标注：

```bash
python tools/prepare_dota_fewshot.py \
  --images data/source/images \
  --annotations data/source/annfiles \
  --output data/FS_Det/RGB/train \
  --shots-per-class 18 \
  --seed 0
```

脚本输出 `annotations_base/`、`annotations_few_shot/` 和 `split_manifest.json`，不复制图片。将 schema 中的图片路径指向输入影像目录。60/100-shot 使用对应 shot 数和独立输出目录；SAR DOTA 使用同一脚本。

SAR 的 18/60/100-shot 路径分别在 `configs/_base_/datasets/fsdet_18shot_sar.py`、`fsdet_60shot_sar.py` 和 `fsdet_100shot_sar.py` 设置。每个文件中的 `fewshot_ann_file`、`fewshot_img_prefix`、`fewshot_manifest_file` 都可以单独修改。DOTA 的 `fewshot_ann_file` 指向 `annotations_few_shot/` 目录，`fewshot_img_prefix` 指向影像目录，`fewshot_manifest_file` 指向同一份 split 的 `split_manifest.json`；schema 路径仅作为默认值。没有可用 manifest 时，将该 shot 配置中的 `fewshot_manifest_file` 设为 `''`，图片数会按 shot 数和 novel 类别数估算，也可用 `FSDET_FT18_IMAGES`、`FSDET_FT60_IMAGES` 或 `FSDET_FT100_IMAGES` 覆盖。对应的 `FSDET_SAR_FT18_*`、`FSDET_SAR_FT60_*`、`FSDET_SAR_FT100_*` 环境变量仍可覆盖配置文件路径。

RGB 也使用相同的 per-shot 配置方式：编辑 `configs/_base_/datasets/fsdet_18shot_rgb.py`、`fsdet_60shot_rgb.py` 或 `fsdet_100shot_rgb.py` 中的 `fewshot_ann_file`、`fewshot_img_prefix`、`fewshot_manifest_file`。对应环境变量为 `FSDET_RGB_FT18_ANN` / `FSDET_RGB_FT18_IMG_ROOT` / `FSDET_RGB_FT18_MANIFEST`，60/100-shot 也按此命名；图片数可用 `FSDET_RGB_FT18_IMAGES`、`FSDET_RGB_FT60_IMAGES`、`FSDET_RGB_FT100_IMAGES` 覆盖。RGB 路径默认取自 schema。当前 method5 微调配置显式使用 SAR dataset base；要使用 RGB 数据，所用实验配置也需继承对应的 `fsdet_*shot_rgb.py`。

### 每个 xxshot.py 的自定义评估阈值

method1–5 的 Tiny/Large `18shot.py`、`60shot.py`、`100shot.py` 均提供独立的 `other_score_threshold`、`other_iou_threshold`、`other_nms_threshold` 配置项，并传给 `data.val` 和 `data.test`。修改一个文件只影响该实验。默认保持 dataset 原来的值：score=0.3、匹配 IoU=0.3，HBB 的 NMS=0.6、DOTA 的 NMS=0.3。

这三个参数控制 `detection_rate`、`recognition_rate`、`false_alarm_rate` 的自定义评估，不改变 mAP 的 IoU 设置，也不改变模型推理时的 `score_thr` 和 NMS。模型已过滤或抑制的候选框无法通过降低自定义评估阈值恢复。只在文件顶层定义变量而不传入 dataset，或把它们写进 `evaluation`，都不会设置 dataset 的这三个属性。

命令行直接覆盖时，训练验证使用 `--cfg-options data.val.other_score_threshold=0.4 data.val.other_iou_threshold=0.5 data.val.other_nms_threshold=0.6`；`tools/test.py` 使用对应的 `data.test.*`。命令行只覆盖顶层 `other_*` 变量不会重新计算已生成的 `data` 字典。

### method3：只有一个大类时的对齐配置

method2/3 共用的体积对齐损失为 `loss_align = vol_weight * softplus(mean(logvol))`，即 `vol_weight * log(1 + exp(mean(logvol)))`。该损失非负，并保留体积越小、损失越小的方向。`align_logvol` 是原始统计量，允许为负，不单独计入总 loss；`vol_weight` 必须为有限非负数。相比旧的直接 log-volume 损失，此变换也改变了梯度尺度。

训练稳定性修复：EMA 原型首次初始化时对当前批次的导数为 1，之后为 `ema_beta`；反向传播现与实际 EMA 更新一致。FocusGate 的路由计算使用 FP32，偏好损失直接使用 `log_softmax` / `logsigmoid`，避免先计算概率再取 log 导致下溢、无穷损失和 NaN 梯度。SparseDispatcher 对非有限权重明确报错，并用同一个正权重掩码构造索引与切分大小。

旧版直接 log-volume 出现负值是正常的：归一化原型的体积小于 1 时，其对数为负。接近线性相关时，log-det 的导数可很大；单纯降低 `vol_weight` 或把损失变成非负，不能替代路由和 EMA 修复。配置中的 `grad_clip.max_norm=35` 不意味着日志中的 `grad_norm` 必须小于 35，该值是裁剪前的范数。修复后的短训练只能验证数值行为，性能恢复仍需重新完整训练和验证集评估。体积目标本身也不区分同向与反向共线，三个以上原型体积变小不必然意味着所有原型彼此靠近。

先区分标注中只有一个检测类别，还是一个大类下面有多个细类别。

**只有一个检测类别**，例如所有目标均标注为 `target`，在 schema 中设置：

```python
sar_base_classes = ('target',)
sar_novel_classes = ()
rgb_classes = ('target',)

# 填写目标的实际语义；这里以船为例，不需要修改标注名称。
sar_base_text_classes = ('ship',)
sar_novel_text_classes = ()
rgb_text_classes = ('ship',)
semantic_mapping = {'ship': {'sar': 0, 'rgb': 0}}
```

保留 schema 中拼接 `sar_all_classes`、`sar_all_text_classes` 和计算类别数的代码。RGB 有多个类别时，保留原有 RGB 类别表，只把映射中的 `rgb` 改为对应船类的本地 ID。

method3 Tiny 的 base 配置使用以下完整 `align_cfg`；`clip_model_path`、`clip_processor_path` 使用文件中已有的本地 RSCLIP 路径变量：

```python
align_cfg = dict(
    is_base_training=True,
    init_clip=True,
    modalities=['sar', 'rgb', 'text'],
    in_dims=dict(sar=256, rgb=256, text=512),
    align_dim=256,
    ema_beta=0.05,
    vol_weight=0.5,
    semantic_mapping=semantic_mapping,
    text_encoder_type='huggingface',
    clip_model_name=clip_model_path,
    clip_processor_name=clip_processor_path,
    use_llm_prompts=False,
    base_modal_categories=base_modal_categories,
    all_modal_categories=all_modal_categories,
)
```

同一类别在 SAR 上继续做少样本适配时，微调配置使用：

```python
align_cfg = dict(
    is_base_training=False,
    init_clip=True,
    modalities=['sar', 'text'],
    in_dims=dict(sar=256, text=512),
    align_dim=256,
    semantic_mapping={},
    novel_class_ids=[0],  # 指定本次需要文本监督的类别；默认空列表会跳过微调 ITA。
    text_encoder_type='huggingface',
    clip_model_name=clip_model_path,
    clip_processor_name=clip_processor_path,
    use_llm_prompts=False,
    base_modal_categories={'sar': sar_base_text_classes},
    all_modal_categories={'sar': sar_all_text_classes},
)
```

在对应 method 文件中添加上述 `align_cfg` 声明，并在已有的 `model = dict(...)` 中加入 `align_cfg=align_cfg`，覆盖 `_base_` 中的默认值。

这里是同类别少样本适配，不是新增 novel 类。Large 将视觉 `in_dims` 和 `align_dim` 中的 256 改为 768，文本维度保持 512。单个文本类别的交叉熵恒为 0，代码现在自动使用余弦损失，让文本对齐保留训练梯度。

如果只有 SAR 数据，base 的 `model.train_modalities` 设为 `['sar']`，数据也只加载 SAR；`align_cfg` 的 `modalities` 设为 `['sar', 'text']`、`in_dims` 移除 `rgb`、`semantic_mapping={}`。无 RGB 数据时无法计算 SAR/RGB 原型对齐。

**一个大类包含多个细类**时，保留检测与文本类别表中的各个细类，不要把检测头改成一类。只在 schema 的 `semantic_mapping` 中将同一大类的标签聚合，例如：

```python
semantic_mapping = {'vessel': {'sar': [0, 1], 'rgb': 6}}
```

整数仍是对应模态的本地标签 ID。method3 按大类聚合原型做跨模态对齐，文本监督仍使用每个细类自己的名称。

## Getting Started

以下以 method4 Tiny、SAR 18-shot 为例。训练前核对 schema 中的数据路径和 base 图片数；SAR 微调图片数按 shot 或 manifest 计算，可用 `FSDET_FT18_IMAGES` 覆盖。在所选实验配置中调整学习率、batch 和训练计划。

### Base training

```bash
# Single GPU
FSDET_GPUS=1 python tools/train.py configs/method4/tiny/base.py \
  --work-dir work_dirs/method4/tiny/base

# Multiple GPUs（示例：4 张）
FSDET_GPUS=4 bash tools/dist_train.sh configs/method4/tiny/base.py 4 \
  --work-dir work_dirs/method4/tiny/base
```

### Fine-tuning

先生成与微调模型匹配的初始化权重，再开始训练：

```bash
python tools/misc/prepare_ft_checkpoint.py \
  configs/method4/tiny/18shot.py \
  work_dirs/method4/tiny/base/latest.pth \
  --output work_dirs/method4/tiny/base/expanded_for_ft.pth

FSDET_GPUS=1 python tools/train.py configs/method4/tiny/18shot.py \
  --work-dir work_dirs/method4/tiny/18shot
```

将 `method4`、`tiny`、`18shot` 替换为所需方法、模型规模和 shot 数。多卡微调使用 `tools/dist_train.sh`，并设置相同的 `FSDET_GPUS`。

method5 base 默认收集路由签名，微调直接复用；初始化输出文件名使用 `expanded_for_ft_with_signatures.pth`。旋转框签名现在按实际旋转区域采样，旧版本生成的 DOTA 签名应重新收集。只有一个总类别、没有新增类别时，应关闭 `enable_transfer`，此时不适用 base-to-novel 路由迁移。初始化工具按类别名迁移 SAR 分类层的 base 行，并将 softmax 背景行移到扩展层的最后一行；仅 novel 行保留新模型初始化值。使用旧工具生成的扩展权重需要从 base checkpoint 重新生成，不能通过继续训练恢复原来的 base 分类权重。SAR 单模态微调会自动冻结 RGB 检测头及 RGB 路由投影层，以避免 DDP 未使用参数错误。

当前微调配置固定使用 SAR；使用自定义初始化文件名时，同步修改微调配置的 `load_from`。

#### method5 SAR DOTA 微调

method5 的 `18shot.py`、`60shot.py`、`100shot.py` 会分别继承同 shot 数的 SAR dataset base。因此，先在上面对应的 `configs/_base_/datasets/fsdet_*shot_sar.py` 中填写该 split 的三个路径。例如 18-shot：

```python
# configs/_base_/datasets/fsdet_18shot_sar.py
fewshot_ann_file = '/path/to/sar_18shot/annotations_few_shot'
fewshot_img_prefix = '/path/to/SAR/train/images'
fewshot_manifest_file = '/path/to/sar_18shot/split_manifest.json'
```

60/100-shot 分别修改 `fsdet_60shot_sar.py` 和 `fsdet_100shot_sar.py` 中的相同字段。将 `ann_file` 和 manifest 指向各自的独立输出目录；若几套 split 共用原始影像，`fewshot_img_prefix` 可以相同。DOTA 格式开关必须在生成初始化权重和训练时都设置，这样 config 才会选择旋转框 ORCNN 模型及 DOTA pipeline。以下为 Tiny、18-shot 示例；换成 `large`、`60shot` 或 `100shot` 即使用对应配置：

```bash
export FSDET_SAR_ANN_FORMAT=dota

# Base checkpoint -> 与 few-shot 模型结构匹配的初始化权重
python tools/misc/prepare_ft_checkpoint.py \
  configs/method5/tiny/18shot.py \
  work_dirs/method5/tiny/base/latest.pth \
  --output work_dirs/method5/tiny/base/expanded_for_ft_with_signatures.pth

# Few-shot fine-tuning
FSDET_GPUS=1 python tools/train.py configs/method5/tiny/18shot.py \
  --work-dir work_dirs/method5/tiny/18shot
```

各 method5 微调 config 的 `load_from` 默认指向上面的 `expanded_for_ft_with_signatures.pth`。如果初始化文件换了位置或文件名，需要同步改所选 `configs/method5/{tiny,large}/{18shot,60shot,100shot}.py` 的 `load_from`。

### Testing

```bash
# SAR JSON 水平框
python tools/test.py configs/method4/tiny/18shot.py \
  work_dirs/method4/tiny/18shot/latest.pth --eval bbox

# DOTA 旋转框（SAR 或 RGB）使用 --eval mAP
# 多卡测试：
bash tools/dist_test.sh CONFIG CHECKPOINT 4 --eval bbox
```

测试时保持与训练一致的模态、标注格式和图像尺寸。输入尺寸默认 800，可通过 `FSDET_IMAGE_SIZE` 修改，须为 32 的正整数倍。

method5 base 的 val/test 均使用 base 类别表；包含 novel 类的测试应使用微调配置和权重。DOTA 的 F1 按 IoU 0.50 计算，无正确检测时记为 0。

SAR DOTA 出现训练 loss 正常、测试框数为 0 时，可检查训练图片在评估 pipeline 下的中间结果：

```bash
FSDET_SAR_ANN_FORMAT=dota python tools/diagnose_sar_dota.py \
  configs/method5/tiny/base.py YOUR_CHECKPOINT.pth --split train --samples 10
```

输出包含 FPN 尺寸、RPN 框数及 recall50、阈值前最大前景分数、背景分数和各类最终框数。RPN recall50 低说明候选框未覆盖目标；候选框正常但前景分数低于阈值时，继续检查分类 head。`gt_roi_correct_class_score_mean` 使用真实框直接检查正确类别分数，可帮助区分 RPN 与分类问题。`--split val/test` 检查对应数据集。若权重是在旧版 SAR FPN 层级错误时训练的，应修正层级后重新训练；重新测试无法修复已学到的错误空间尺度。

## Acknowledgement

本项目基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 和 [MMDetection](https://github.com/open-mmlab/mmdetection)。README 结构参考 [SAE-FSDet](https://github.com/YanxingLiu/SAE-FSDet)。
