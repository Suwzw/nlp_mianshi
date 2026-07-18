# 基于大小模型协同的小样本低空宽带信号检测

## 课题信息

- **选题组合**: A2(宽带信号检测) + B7(大小模型协同) + C2(小样本学习)
- **数据集**: 低空五类宽带信号检测与识别数据集 (WiFi, Bluetooth, ZigBee, Lightbridge, XPD)
- **核心问题**: 当标注样本有限时，如何通过Teacher-Student知识迁移保持检测精度？

## 环境要求

```
Python 3.11+
PyTorch 2.5.1+ (CUDA 12.4)
ultralytics
opencv-python
matplotlib
seaborn
pandas
streamlit
```

## 安装

```bash
conda activate pytorch
pip install ultralytics streamlit
```

## 项目结构

```
project/
├── dataset/               # 数据集 (按录制场次划分)
│   ├── images/
│   │   ├── train/         # 6场录制, 1902张
│   │   ├── val/           # 2场录制, 634张
│   │   ├── test/          # 1场录制, 317张
│   │   ├── train_fewshot/ # 小样本子集, 232张
│   │   └── train_distill/ # 蒸馏数据 (few-shot + Teacher伪标签)
│   ├── labels/
│   ├── data.yaml
│   ├── data_fewshot.yaml
│   └── data_distill.yaml
├── prepare_data.py        # 数据集处理
├── train.py               # 训练脚本
├── distill.py             # 伪标签知识蒸馏
├── inference.py           # 推理脚本
├── analyze.py             # 实验分析
├── demo.py                # Streamlit Demo
├── runs/                  # 训练输出
├── figures/               # 实验图表
├── ppt/                   # PPT
└── speech/                # 讲稿
```

## 使用方法

### 1. 数据处理
```bash
python prepare_data.py
```

### 2. 训练 (4组实验)
```bash
# ① YOLOv8n 全量训练 (Student上界)
python train.py v8n_full

# ② YOLOv8m 全量训练 (Teacher)
python train.py v8m_full

# ③ YOLOv8n 小样本训练 (退化)
python train.py v8n_fewshot

# ④ 伪标签蒸馏训练 (恢复)
python distill.py all
```

### 3. 实验分析
```bash
python analyze.py
```

### 4. 推理
```bash
python inference.py runs/v8n_distilled/weights/best.pt test_image.jpg 0.25
```

### 5. Demo
```bash
streamlit run demo.py
```

## 实验设计

| 实验 | 模型 | 训练数据 | 目的 |
|------|------|----------|------|
| ① 上界 | YOLOv8n | 全量(1902张) | Student精度天花板 |
| ② Teacher | YOLOv8m | 全量(1902张) | Teacher模型参照 |
| ③ 退化 | YOLOv8n | 小样本(232张) | 展示小样本性能下降 |
| ④ 恢复 | YOLOv8n | 小样本+伪标签 | 知识迁移恢复效果 |

## 信号类别

| ID | 类别 | 调制方式 | 带宽 | 应用 |
|----|------|----------|------|------|
| 0 | WiFi | OFDM | 20-160 MHz | 无线局域网 |
| 1 | Bluetooth | GFSK | 1 MHz | 短距离通信 |
| 2 | ZigBee | DSSS | 2 MHz | 物联网 |
| 3 | Lightbridge | OFDM | 10 MHz | 无人机图传 |
| 4 | XPD | FM | 6 MHz | 无线麦克风 |
