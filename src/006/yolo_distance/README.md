# 基于 YOLOv10 的单目距离检测

本项目使用 YOLOv10 检测前方车辆和人员，通过针孔相机的相似三角形模型估算目标距离，并在目标距离低于设定阈值时显示警告。

## 环境准备

在仓库根目录执行：

```bash
conda activate hands_on
python -m pip install -r src/006/yolo_distance/requirements.txt
```

如果 `hands_on` 环境已经安装 `ultralytics`、`torch`、`opencv-python`、`PyYAML` 和 `pytest`，可以跳过安装步骤。首次运行推理时，程序会自动将 `yolov10n.pt` 下载到本项目目录。使用 GPU 时，应确保 PyTorch 与本机 CUDA 版本匹配。

## 1. 标定摄像头

固定摄像头的位置、分辨率和变焦倍率。将一个实际尺寸已知的物体放在测量好的距离处，并记录它在检测框中的像素宽度。例如，目标实际宽度为 `1.8 m`，距离为 `3.0 m`，检测框宽度为 `240 px`：

```bash
python src/006/yolo_distance/calibrate.py --pixel-size 240 --known-distance 3.0 --known-size 1.8
```

将输出的焦距填入 `config.yaml` 中的 `camera.fx_px`。使用人员身高和检测框高度标定时，应增加 `--axis y`，并将结果填入 `camera.fy_px`。

推荐采集多组标定数据，以降低单次测量误差。CSV 格式如下：

```csv
axis,pixel_size_px,known_distance_m,known_size_m
x,430,2.0,1.8
x,286,3.0,1.8
x,216,4.0,1.8
```

将文件保存在当前项目目录后执行：

```bash
python src/006/yolo_distance/calibrate.py --csv src/006/yolo_distance/measurements.csv
```

## 2. 运行检测

使用默认摄像头：

```bash
python src/006/yolo_distance/main.py
```

处理已有视频：

```bash
python src/006/yolo_distance/main.py --source path/to/video.mp4
```

保存标注后的视频：

```bash
python src/006/yolo_distance/main.py --save outputs/result.mp4
```

运行窗口中按 `q` 退出。相对输出路径会创建在 `src/006/yolo_distance/` 内。在尚未配置 `fx_px` 或 `fy_px` 时，程序只显示检测结果和 `uncalibrated`，不会输出没有标定依据的距离。

### 现成测试视频

项目的 `data/vtest.avi` 来自
[OpenCV 官方示例数据](https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi)，
画面中包含多名行人，可用于验证视频读取、YOLO 检测、结果绘制和保存流程：

```bash
python src/006/yolo_distance/main.py --source src/006/yolo_distance/data/vtest.avi
```

由于该视频的相机参数未知，上述命令会将距离显示为 `uncalibrated`。如需查看完整测距界面，可以临时假设垂直像素焦距为 `695 px`：

```bash
python src/006/yolo_distance/main.py --source src/006/yolo_distance/data/vtest.avi --fy-px 695
```

`695 px` 是根据视频高度和假设视场角得到的演示值，不是该摄像头的真实标定结果，显示的距离只能用于检查程序流程。该视频也没有真实距离标注，不能用于评价测距误差。正式实验仍需使用固定摄像头采集带有实测距离的数据，并把标定得到的 `fy_px` 写入 `config.yaml`。

## 3. 评价测距结果

准备包含以下字段的 CSV 文件：

```csv
ground_truth_m,predicted_m
2.0,2.15
3.0,2.86
4.0,4.21
```

然后执行：

```bash
python src/006/yolo_distance/evaluate.py src/006/yolo_distance/results.csv
```

程序会输出样本数、平均绝对误差（MAE）、均方根误差（RMSE）、平均绝对百分比误差（MAPE）和系统偏差。

## 测试

测试过程不会加载 YOLO 权重，也不需要连接摄像头：

```bash
python -m pytest src/006/yolo_distance/tests
```

`config.yaml` 中的人员身高和车辆宽度只是初始估计值。正式实验应使用实测尺寸，并在论文中分析车型差异、人员姿态、遮挡、检测框抖动和镜头畸变带来的误差。
