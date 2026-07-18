# 项目交接说明（给接手 AI）

> **文档作用**：本文档是「基于大小模型协同的小样本低空宽带信号检测」项目的完整交接说明。阅读本文档即可掌握项目全貌、当前状态、所有文件用途、待办事项和关键技术细节。
>
> **最后更新**：2026-07-18（实验全部完成，PPT/讲稿已用真实数据定稿）
>
> **旧文档提示**：`HANDOFF.md` 是训练中途写的，部分内容已过时（当时蒸馏还在训练中）。以本文档为准；如需保留历史可看 HANDOFF.md，但**一切以本文档为准**。

---

## 〇、一句话项目状态

**4 组对比实验全部完成，4 个 best.pt 权重都已生成，实验图表、PPT 内容、3 分钟讲稿均已用真实数据定稿。** 剩余工作主要是：① 修复 `analyze.py` 逐类 AP 提取 bug（当前全为 0.0）；② 测试 Streamlit Demo；③ 用户手动把 PPT 内容做成 3 页幻灯片。

---

## 一、项目概述

- **课题名称**：基于大小模型协同的小样本低空宽带信号检测
- **选题组合**：A2(宽带信号检测) + B7(大小模型协同) + C2(小样本学习)
- **用途**：保研面试交叉创新课题展示（≤3 分钟 PPT，3 页）
- **面试时间**：2026-07-19 周日 19:20，南方科技大学卓工夏令营
- **用户**：万振威，哈尔滨工程大学通信工程专业
- **核心研究问题**：当标注样本从 1902 张降至 232 张（减少 88%）时，如何通过 Teacher-Student 知识迁移保持检测精度？

---

## 二、最终实验结果（测试集，已定稿）

| 实验 | 模型 | 训练数据 | mAP@50 | mAP@50-95 | 参数量 | 模型大小 | FPS |
|------|------|----------|--------|-----------|--------|----------|-----|
| ① 全量 Student | YOLOv8n | 1902 张 | 96.4% | 81.5% | 3.0M | 5.9MB | 99 |
| ② Teacher | YOLOv8m | 1902 张 | 96.4% | 81.9% | 25.9M | 49.6MB | 42 |
| ③ 小样本 Student | YOLOv8n | 232 张 | 95.2% | 78.3% | 3.0M | 5.9MB | 153 |
| ④ 蒸馏 Student | YOLOv8n | 232 张+伪标签 | 94.9% | 78.0% | 3.0M | 5.9MB | 156 |

**三个关键发现（PPT 已采用）**：
1. **小样本退化**：数据减 88%，mAP@50 仅降 1.2%，但 mAP@50-95 降 3.2%——定位精度受影响更大。
2. **蒸馏效果**：伪标签蒸馏在**验证集**上有提升（95.0%→95.8%），但**测试集**上略降（95.2%→94.9%）——伪标签质量是关键制约。这是一个"有深度的发现"，PPT 中诚实讨论。
3. **模型效率**：Student 仅 3.0M 参数、156 FPS，满足边缘部署；Teacher 需 25.9M 参数、42 FPS。

---

## 三、环境与路径（务必使用绝对路径）

- **项目根目录**：`D:/Code/nlp_mianshi/project/`
- **Python 解释器（conda pytorch 环境）**：`/c/Users/28271/Anaconda3/envs/pytorch/python.exe`
- **关键依赖**：PyTorch 2.5.1 + CUDA 12.4、ultralytics、opencv-python、matplotlib、seaborn、pandas、streamlit
- **硬件**：RTX 4060 8GB GPU
- **数据集原始位置**：`D:/Code/nlp_mianshi/低空五类宽带信号数据集/`（YOLO 格式，2853 张 512×512 图片）

> ⚠️ 所有 bash 命令请用上面的 Python 绝对路径，不要直接用 `python`，否则可能调到错误的解释器。

---

## 四、项目文件清单

### 4.1 代码脚本

| 文件 | 用途 | 状态 |
|------|------|------|
| `prepare_data.py` | 数据集按录制场次划分（train/val/test/fewshot） | ✅ 已执行，无需重跑 |
| `train.py` | 训练脚本，支持 `v8n_full`/`v8m_full`/`v8n_fewshot` 三种模式 | ✅ 已完成训练 |
| `distill.py` | 伪标签生成(`generate`)+蒸馏训练(`train`)，TEACHER 指向 v8m_full/best.pt | ✅ 已执行 |
| `distill_train_simple.py` | 独立蒸馏训练脚本，带 `if __name__=='__main__'` 保护，workers=0 | ✅ 已用此脚本完成最终蒸馏 |
| `analyze.py` | 实验分析+图表生成（mAP 对比、模型参数、Loss 曲线、蒸馏增益、检测结果） | ⚠️ 有 bug：逐类 AP 提取为 0.0 |
| `inference.py` | 推理脚本，含 AI 解释模块 | ✅ 可用 |
| `demo.py` | Streamlit Demo | ⏳ 未测试运行 |
| `generate_diagram.py` | 知识蒸馏框架图生成 | ✅ 已生成 |

### 4.2 训练参数（重要，复现用）

```
epochs=100, imgsz=512, batch=8, workers=0, amp=False, patience=30
蒸馏伪标签阈值: conf=0.7（第一轮用 0.5 测试集退化，故提高到 0.7）
```

### 4.3 模型权重（4 个 best.pt，均已生成）

| 路径 | 说明 |
|------|------|
| `runs/v8n_full/weights/best.pt` | YOLOv8n 全量 (6.2MB) |
| `runs/v8m_full/weights/best.pt` | YOLOv8m Teacher (52MB) |
| `runs/v8n_fewshot/weights/best.pt` | YOLOv8n 小样本 (6.2MB) |
| `runs/v8n_distilled/weights/best.pt` | YOLOv8n 蒸馏 (6.2MB) |

### 4.4 交付物（已定稿）

| 文件 | 说明 |
|------|------|
| `ppt/ppt_content.md` | PPT 内容（3 页，含真实数据）—用户据此手动制作幻灯片 |
| `speech/speech_3min.md` | 3 分钟讲稿（180 秒，分段时间已标注） |
| `figures/framework_diagram.png` | 知识蒸馏框架图 |
| `figures/map_comparison.png` | mAP 对比柱状图 |
| `figures/model_comparison.png` | 模型参数/FLOPs/FPS 对比 |
| `figures/loss_curves.png` | Loss 曲线 |
| `figures/distillation_gain.png` | 蒸馏增益图 |
| `figures/per_class_ap.png` | 逐类 AP 图（⚠️ 数据全 0，需修复） |
| `figures/summary.md` | 实验结果汇总（⚠️ 逐类 AP 全 0） |
| `figures/detections_v8n_full/`、`detections_v8n_distilled/` | 检测结果可视化 |
| `README.md` | 项目说明（精简版） |

### 4.5 数据集（已划分，在 `dataset/`）

```
dataset/
├── images/
│   ├── train/         # 6 场录制(rec_154,155,180,181,182,183), 1902 张
│   ├── val/           # 2 场录制(rec_184,185), 634 张
│   ├── test/          # 1 场录制(rec_186), 317 张
│   ├── train_fewshot/ # 小样本子集, 每类 50 张, 共 232 张
│   └── train_distill/ # 蒸馏数据 (few-shot + Teacher 伪标签, conf=0.7, 135 个新增框)
├── labels/
├── data.yaml
├── data_fewshot.yaml
└── data_distill.yaml
```

---

## 五、待办事项（按优先级排序）

### 5.1 修复 analyze.py 逐类 AP 提取（⭐ 高优先级，可选）

**问题**：`figures/summary.md` 和 `per_class_ap.png` 的逐类 AP 全是 0.0，说明 `analyze.py` 在从 ultralytics 验证结果中提取逐类 AP 时解析失败。

**影响**：PPT 第 2 页本想用逐类 AP 图，但因数据全 0 当前未用。若修复可增强 PPT 第 2 页的说服力；不修复也不影响面试（PPT 现有内容已完整）。

**排查方向**：
- ultralytics 的 `model.val()` 返回结果对象，逐类 AP 在 `results.box.ap_class_index` / `results.box.ap50` / `results.box.ap` 中（具体属性名随版本变化，需 print 确认）
- 或从 `runs/detect/val*/` 下的日志/CSV 读取
- 验证结果输出在 `runs/detect/val*`（有 val 到 val-12 多个目录，最新的是 val-12 左右）

**重新运行**：
```bash
cd D:/Code/nlp_mianshi/project
/c/Users/28271/Anaconda3/envs/pytorch/python.exe analyze.py
```

### 5.2 测试 Streamlit Demo（⭐ 中优先级，可选）

```bash
cd D:/Code/nlp_mianshi/project
/c/Users/28271/Anaconda3/envs/pytorch/python.exe -m pip install streamlit   # 若未装
/c/Users/28271/Anaconda3/envs/pytorch/python.exe -m streamlit run demo.py
```
Demo 支持上传图片、选择模型、显示检测结果。面试现场若需演示可用。需确认 `demo.py` 里的模型路径与实际权重路径一致。

### 5.3 制作最终 PPT（低优先级，用户手动）

用户根据 `ppt/ppt_content.md` 手动做 3 页幻灯片：
- **第 1 页**：研究动机（低空经济 + 频谱监测 + 小样本痛点）
- **第 2 页**：方案框图 + 实验结果表 + 逐类/增益图
- **第 3 页**：反思展望（伪标签质量 + 开集识别 + AI 芯片部署）

---

## 六、关键技术说明（接手必读）

### 6.1 知识蒸馏方案（伪标签蒸馏，非经典特征级 KD）

本项目用的是**伪标签蒸馏（Label Distillation）**，不是经典的特征级 KD：
1. Teacher(YOLOv8m) 在全量 1902 张数据上训练
2. Teacher 对 few-shot 数据(232 张)推理，把置信度 >0.7 的检测框作为**伪标签**
3. 伪标签与人工标注合并（新增 135 个框），Student(YOLOv8n) 在合并数据上训练
4. Teacher 的检测知识通过伪标签传递给 Student

**为什么测试集蒸馏反而略降？** 伪标签可能含误检/漏检噪声，低质量伪标签会"放大误差"而非"传递知识"。验证集提升(95.0%→95.8%)但测试集略降(95.2%→94.9%)，提示过拟合风险。这是 PPT「反思」部分的核心论点。

### 6.2 数据集必须按录制场次划分（不能随机划分）

- 同一录制场次的图片来自同一段连续采集，高度相关
- 随机划分会让 train/test 含同场景图片 → 数据泄露 → mAP 虚高
- 按场次划分的测试集是**完全未见的采集场景**，结果更有说服力
- 划分：Train(6 场) / Val(2 场) / Test(1 场)

### 6.3 信号类别信息

| ID | 类别 | 调制 | 带宽 | 应用 |
|----|------|------|------|------|
| 0 | WiFi | OFDM | 20-160 MHz | 无线局域网 |
| 1 | Bluetooth | GFSK | 1 MHz | 短距通信 |
| 2 | ZigBee | DSSS | 2 MHz | 物联网 |
| 3 | Lightbridge | OFDM | 10 MHz | 无人机图传 |
| 4 | XPD | FM | 6 MHz | 无线麦克风 |

### 6.4 Windows 训练的坑（复现必看）

1. **必须 `amp=False`**：AMP 检查要下载参考模型，网络不通会卡住
2. **`workers=0` 或 `if __name__=='__main__'` 保护**：Windows 多进程有 bug，workers>0 必须在 main guard 里调用 train
3. **`batch=8`**：Windows 页面文件太小，batch=16 会 OOM
4. **后台运行用 nohup 重定向**：`nohup python script.py > log.log 2>&1 &`（不要用 head 管道，会被杀）

---

## 七、关键命令速查

```bash
PYTHON=/c/Users/28271/Anaconda3/envs/pytorch/python.exe
cd D:/Code/nlp_mianshi/project

# 训练（已全部完成，一般无需重跑）
$PYTHON train.py v8n_full        # ① YOLOv8n 全量
$PYTHON train.py v8m_full        # ② YOLOv8m Teacher
$PYTHON train.py v8n_fewshot     # ③ YOLOv8n 小样本

# 蒸馏（已完成）
$PYTHON distill.py generate      # 生成伪标签(conf=0.7)
$PYTHON distill_train_simple.py  # 蒸馏训练(workers=0)

# 分析（⚠️ 逐类 AP 有 bug 待修）
$PYTHON analyze.py

# 推理
$PYTHON inference.py runs/v8n_distilled/weights/best.pt test_image.jpg 0.25

# Demo
$PYTHON -m streamlit run demo.py

# 查看某个模型的测试集结果
$PYTHON -c "from ultralytics import YOLO; m=YOLO('runs/v8n_distilled/weights/best.pt'); r=m.val(data='dataset/data.yaml', split='test'); print(r.box.map50, r.box.map)"
```

---

## 八、PPT 叙事框架（3 页，已定稿）

- **第 1 页 研究动机**：低空经济 → 2.4GHz 频段信号密集 → 频谱监测需求 → 标注昂贵小样本痛点 → 大模型重/小模型差的核心矛盾 → 提出"大小模型协同"
- **第 2页 方案与结果**：框架图 + 4 组实验表 + 关键发现（小样本退化、蒸馏测试集略降、模型轻量化）
- **第 3 页 反思展望**：伪标签质量依赖（诚实讨论测试集下降）→ 置信度加权/响应级 KD → 开集识别 + 持续学习 + ONNX 部署

讲稿见 `speech/speech_3min.md`，已按 180 秒分配好段落。

---

## 九、给接手 AI 的工作建议

1. **先读本文档**，再看 `ppt/ppt_content.md` 和 `speech/speech_3min.md`——这两个文件已是最终定稿，**不要改动其中的实验数据**。
2. 用户的核心诉求是"聚焦深度，讲透一个小问题"，不贪大求全。面试在即，**稳定优先，不要为了追求更好结果而重训**（重训有风险且时间紧）。
3. 蒸馏测试集略降(94.9% vs 95.2%)是**已接受的结果**，并已转化为"有深度的发现"写进 PPT 反思部分。**不要试图掩盖或重新编造数据**。
4. 若用户要求进一步优化，可建议（但不主动执行）：① 调高 conf 阈值到 0.8 重新蒸馏；② 引入置信度加权损失；③ 改用响应级 KD（KL 散度对齐概率分布）。
5. `runs/detect/val*` 下有多次验证的输出目录，最新结果在最末编号，历史目录可忽略。

---

## 十、记忆与上下文文件索引（WorkBuddy 记忆系统）

WorkBuddy 有三层记忆系统，接手 AI 可按需查阅。以下是本项目的记忆文件清单和路径。

### 10.1 项目级记忆（⭐ 最重要，优先读）

目录：`D:\Code\nlp_mianshi\.workbuddy\memory\`

| 文件 | 内容摘要 | 说明 |
|------|----------|------|
| `MEMORY.md` | 项目长期记忆：项目概述、关键约定（数据集按场次划分/蒸馏分级策略/训练环境）、用户偏好（深度优先/要 Demo/交付清单） | 跨会话保留的核心约定，限制 3000 字符/会话 |
| `2026-07-11.md` | 7.11 工作日志：课题确定、数据集、4 组实验设计、环境配置、训练进度、关键结果、网络/Windows 问题记录 | 最详细的一天日志，含全部踩坑过程 |
| `2026-07-18.md` | 7.18 工作日志：写本交接文档的过程 | 本日日志 |

**维护规则**：每日日志（YYYY-MM-DD.md）只追加不覆盖；只记有跨会话价值的内容；30 天以上的日志需提炼进 `MEMORY.md` 后删除。

### 10.2 用户级记忆（跨项目，位于用户主目录）

目录：`C:\Users\28271\.workbuddy\`

| 文件 | 内容 | 说明 |
|------|------|------|
| `SOUL.md` | AI 助手的人格定义、价值观、行为边界 | 人格源头，改动需告知用户 |
| `IDENTITY.md` | AI 助手身份卡（姓名/性质/风格/emoji） | 模板未填写 |
| `USER.md` | 用户档案：万振威，哈工程通信工程，保研南科大卓工夏令营 | 用户基本信息 |
| `BOOTSTRAP.md` | 身份引导脚本 | 未执行完成，可忽略 |
| `memory\e18ff65d-5b36-44f3-ba51-64361e0659eb_memory.md` | 云端生成的用户画像缓存 | ⚠️ 只读，服务器管理，本地改动会被覆盖 |

> 注：跨项目用户级 `MEMORY.md`（`C:\Users\28271\.workbuddy\MEMORY.md`）当前未创建，按规则用于记录跨项目偏好。

### 10.3 云端历史会话检索

- 通过 `conversation_search` 工具可检索用户所有历史对话（服务器端排名），用于回溯特定事件或决策
- 本项目完整开发历程（从选题到实验完成）已在 `2026-07-11.md` 日志中摘要记录，通常无需额外检索
- 检索时 query 必须自包含（工具无法访问当前对话），需重述当前任务和缺失的上下文

### 10.4 接手 AI 读取顺序建议

1. **先读** `PROJECT_GUIDE.md`（本文件）——掌握项目全貌和当前状态
2. **再读** `D:\Code\nlp_mianshi\.workbuddy\memory\MEMORY.md`——理解核心约定和用户偏好
3. **按需读** `2026-07-11.md`——了解踩坑细节和决策过程
4. **最后看** `ppt/ppt_content.md` 和 `speech/speech_3min.md`——确认交付物内容
5. 身份文件（SOUL/IDENTITY/USER）仅在与用户持续交互时才需要关注
