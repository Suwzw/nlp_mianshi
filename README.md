# 基于大小模型协同的小样本低空宽带信号检测

本项目面向低空频谱监测场景，研究在标注样本有限时，如何利用大模型教师与轻量化学生模型协同完成宽带通信信号检测。选题组合为 A2「基于时频图的宽带信号检测与识别」+ B7「大小模型协同」+ C2「小样本学习」。

## 研究背景

低空经济、无人机、物联网设备和无线音频设备大量工作在 2.4GHz ISM 频段，多协议信号共存使频谱监测和干扰识别变得更加重要。实际部署中，射频信号采集和标注成本较高，新设备或新协议出现时往往只有少量标注样本；同时，边缘设备对模型大小和推理速度有严格要求。

本项目将宽带信号时频图视为二维图像，用目标检测方法定位并识别 WiFi、Bluetooth、ZigBee、Lightbridge 和 XPD 五类信号。

## 方法概述

项目采用 Teacher-Student 伪标签蒸馏框架：

1. 使用全量标注数据训练 YOLOv8m 作为 Teacher。
2. 使用少量标注数据训练 YOLOv8n 作为小样本 Student 基线。
3. Teacher 对小样本训练集生成高置信度伪标签。
4. 将人工标注与伪标签合并，训练轻量化 YOLOv8n Student。

数据集按录制场次划分为训练集、验证集和测试集，避免同一采集场景同时出现在训练和测试中造成数据泄露。

## 实验结果

### 整体性能对比

| 模型 | 训练数据 | mAP@50 | mAP@50-95 | 参数量 | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8n 全量 | 1902 张 | 96.4% | 81.5% | 3.0M | 122 |
| YOLOv8m Teacher | 1902 张 | 96.4% | 81.9% | 25.9M | 40 |
| YOLOv8n 小样本 | 232 张 | 95.2% | 78.3% | 3.0M | 153 |
| YOLOv8n 蒸馏 | 232 张 + 伪标签 | 94.9% | 78.0% | 3.0M | 156 |

### 逐类 AP@50 对比

| 模型 | WiFi | Bluetooth | ZigBee | Lightbridge | XPD |
|---|---:|---:|---:|---:|---:|
| YOLOv8n 全量 | 91.9 | 92.8 | 98.4 | 99.5 | 99.5 |
| YOLOv8m Teacher | 91.3 | 93.5 | 98.4 | 99.5 | 99.5 |
| YOLOv8n 小样本 | 89.4 | 90.2 | 97.7 | 99.5 | 99.4 |
| YOLOv8n 蒸馏 | 86.4 | 90.9 | 98.2 | 99.5 | 99.5 |

### 结果分析

小样本训练使 mAP@50 从 96.4% 降至 95.2%，但 mAP@50-95 从 81.5% 降至 78.3%，说明样本减少对定位质量影响更明显。伪标签蒸馏在验证集上有提升，但测试集略低于小样本基线，表明伪标签质量和类别差异会影响知识迁移效果。逐类结果显示，Bluetooth 和 ZigBee 略有收益，而 WiFi 下降更明显，后续可从置信度加权、类别自适应筛选和响应级知识蒸馏方向改进。

## 可视化结果

- `figures/framework_diagram.png`：大小模型协同框架
- `figures/map_comparison.png`：整体 mAP 对比
- `figures/per_class_ap.png`：逐类 AP 对比
- `figures/distillation_gain.png`：蒸馏前后逐类变化
- `figures/model_comparison.png`：参数量、模型大小和推理速度对比

## Demo

本项目提供 Streamlit 演示系统，用于展示时频图输入、检测框输出、类别解释、模型性能和实验对比。

运行方式：

```powershell
& 'C:\Users\28271\Anaconda3\envs\pytorch\python.exe' -m streamlit run demo.py
```

说明：仓库未包含原始数据集和模型权重。若需完整运行 Demo，请在本地保留训练得到的 `runs/*/weights/best.pt` 权重文件和测试样例图像。

## 文件结构

```text
.
├── README.md
├── demo.py                  # Streamlit 演示系统
├── prepare_data.py           # 数据集划分脚本
├── train.py                  # YOLOv8 训练脚本
├── distill.py                # 伪标签生成与蒸馏流程
├── distill_train_simple.py   # 蒸馏训练脚本
├── analyze.py                # 实验分析与图表生成
├── inference.py              # 单图推理脚本
├── figures/                  # 实验图表
├── ppt/ppt_content.md        # 三页汇报内容
└── speech/speech_3min.md     # 三分钟汇报讲稿
```

## 环境依赖

- Python 3.11
- PyTorch + CUDA
- ultralytics
- opencv-python
- streamlit
- pandas
- matplotlib
- pillow
