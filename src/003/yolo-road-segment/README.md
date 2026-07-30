# YOLO Road Segmentation

基于 Ultralytics YOLO26 的单类别道路实例分割项目。项目按照以下流水线组织：

```text
环境与资源 -> 原始图像 -> 标注 -> 数据划分 -> 数据检查 -> 训练 -> 验证 -> 推理 -> 导出
```

## 1. 项目结构

```text
yolo-road-segment/
├── configs/
│   ├── pipeline.yaml          # 训练、验证和推理参数
│   └── road_seg.yaml          # 数据集定义
├── datasets/
│   └── road/
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── models/                    # 预训练权重，不提交 Git
├── outputs/                   # 实验输出，不提交 Git
├── scripts/
│   ├── bootstrap.py           # 创建 uv 环境并下载资源
│   ├── prepare_assets.py      # 下载模型和 COCO8-Seg
│   ├── smoke_test.py          # 小数据集冒烟测试
│   ├── extract_frames.py      # 从视频抽帧
│   ├── split_dataset.py       # 划分数据集
│   ├── check_dataset.py       # 检查图像和分割标签
│   └── run_pipeline.py        # 串联训练、验证和推理
└── src/
    ├── common.py
    ├── train.py
    ├── validate.py
    ├── predict.py
    └── export.py
```

## 2. 一键搭建环境

需要先安装 [uv](https://docs.astral.sh/uv/)。确认 uv 可用后，在项目根目录运行：

```bash
python scripts/bootstrap.py
```

如果系统没有 Python，可先用 uv 自带的 Python 运行器：

```bash
uv run --python 3.11 scripts/bootstrap.py
```

该脚本执行以下工作：

1. 安装 Python 3.11；
2. 创建 `.venv`；
3. 使用 uv 自动选择 PyTorch GPU/CPU 后端；
4. 安装固定版本的 Ultralytics 和项目依赖；
5. 下载 `yolo26n-seg.pt` 和 COCO8-Seg；
6. 输出 `environment.lock.txt` 记录实际环境。

强制重建环境：

```bash
python scripts/bootstrap.py --recreate
```

指定 CPU：

```bash
python scripts/bootstrap.py --torch-backend cpu
```

## 3. 冒烟测试

```bash
python -m scripts.smoke_test
```

使用 COCO8-Seg 训练 3 轮，用于确认模型、数据加载器、PyTorch 和显卡均可正常工作。

## 4. 数据准备

### 从视频抽帧

```bash
python -m scripts.extract_frames D:\videos \
    --output datasets/road_raw/images \
    --interval 2
```

`--interval 2` 表示每 2 秒保存一帧。抽帧后使用 CVAT 标注 `road` 多边形，并将标签放入：

```text
datasets/road_raw/labels/
```

### 划分数据集

```bash
python -m scripts.split_dataset --clean
```

默认按照 `8:1:1` 划分训练、验证和测试集，随机种子固定为 42。

### 检查数据

```bash
python -m scripts.check_dataset
```

检查内容包括：

- 图像与标签是否同名配对；
- 图片能否读取；
- 类别编号是否合法；
- 多边形点数是否足够；
- 坐标是否归一化到 `[0, 1]`；
- 是否存在空面积多边形或孤立标签。

检查报告保存在 `outputs/dataset_check.json`。

## 5. 完整流水线

```bash
python -m scripts.run_pipeline --stage all --clean-run
```

依次执行：

```text
check -> train -> val -> predict
```

只训练：

```bash
python -m scripts.run_pipeline --stage train --clean-run
```

指定 GPU 0：

```bash
python -m scripts.run_pipeline --stage all --device 0 --clean-run
```

## 6. 单独运行核心阶段

```bash
# 训练
python -m src.train

# 验证
python -m src.validate

# 推理
python -m src.predict --source test_images

# 导出 ONNX
python -m src.export --format onnx
```

临时覆盖配置参数：

```bash
python -m src.train --epochs 100 --batch 4 --device 0
```

长期实验参数应修改 `configs/pipeline.yaml`，保证实验配置能够提交和复现。

## 7. 主要输出

```text
outputs/train/road_yolo26n_seg/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── results.png
└── run_metadata.json
```

验证指标写入 `metrics_summary.json`，推理信息写入 `prediction_summary.json`。
