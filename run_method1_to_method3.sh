#!/usr/bin/env bash
set -e

# 在项目根目录、openmmlab 环境中运行；以下使用 Tiny 和 2 张 GPU。
export FSDET_SAR_ANN_FORMAT=dota
export FSDET_METHOD3_TEXT_ENCODER=sarclip
export FSDET_GPUS=2
export FSDET_IMAGE_SIZE=1024

# method1
bash tools/dist_train.sh configs/method1/tiny/base.py 2 --work-dir work_dirs/method1/tiny/base
python tools/misc/prepare_ft_checkpoint.py configs/method1/tiny/18shot.py work_dirs/method1/tiny/base/latest.pth --output work_dirs/method1/tiny/base/expanded_for_ft.pth

# 同一方法的 18/60/100-shot 模型结构相同，可共用上述扩展权重。
export FSDET_FT18_IMAGES=432
bash tools/dist_train.sh configs/method1/tiny/18shot.py 2 --work-dir work_dirs/method1/tiny/18shot
export FSDET_FT60_IMAGES=1440
bash tools/dist_train.sh configs/method1/tiny/60shot.py 2 --work-dir work_dirs/method1/tiny/60shot
export FSDET_FT100_IMAGES=2400  # 按实际 100-shot 图片数修改
bash tools/dist_train.sh configs/method1/tiny/100shot.py 2 --work-dir work_dirs/method1/tiny/100shot

# method2
bash tools/dist_train.sh configs/method2/tiny/base.py 2 --work-dir work_dirs/method2/tiny/base
python tools/misc/prepare_ft_checkpoint.py configs/method2/tiny/18shot.py work_dirs/method2/tiny/base/latest.pth --output work_dirs/method2/tiny/base/expanded_for_ft.pth

export FSDET_FT18_IMAGES=432
bash tools/dist_train.sh configs/method2/tiny/18shot.py 2 --work-dir work_dirs/method2/tiny/18shot
export FSDET_FT60_IMAGES=1440
bash tools/dist_train.sh configs/method2/tiny/60shot.py 2 --work-dir work_dirs/method2/tiny/60shot
export FSDET_FT100_IMAGES=2400  # 按实际 100-shot 图片数修改
bash tools/dist_train.sh configs/method2/tiny/100shot.py 2 --work-dir work_dirs/method2/tiny/100shot

# method3：使用 SAR-CLIP
bash tools/dist_train.sh configs/method3/tiny/base.py 2 --work-dir work_dirs/method3/tiny/base
python tools/misc/prepare_ft_checkpoint.py configs/method3/tiny/18shot.py work_dirs/method3/tiny/base/latest.pth --output work_dirs/method3/tiny/base/expanded_for_ft.pth

export FSDET_FT18_IMAGES=432
bash tools/dist_train.sh configs/method3/tiny/18shot.py 2 --work-dir work_dirs/method3/tiny/18shot
export FSDET_FT60_IMAGES=1440
bash tools/dist_train.sh configs/method3/tiny/60shot.py 2 --work-dir work_dirs/method3/tiny/60shot
export FSDET_FT100_IMAGES=2400  # 按实际 100-shot 图片数修改
bash tools/dist_train.sh configs/method3/tiny/100shot.py 2 --work-dir work_dirs/method3/tiny/100shot
