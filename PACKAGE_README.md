# 面试项目 AI 学习精简包

这个文件夹是从原项目复制出来的安全精简副本，用于上传给网页版 ChatGPT、Gemini，或推送到 GitHub 作为学习材料。

## 包含内容

- `LEARN_ME.md`：给网页版 AI 的学习提示词和阅读顺序
- `PROJECT_GUIDE.md`：项目交接说明，包含实验结果和关键结论
- `README.md`：项目结构说明
- `prepare_data.py`：数据集划分脚本
- `train.py`：YOLOv8 训练脚本
- `distill.py`：伪标签蒸馏脚本
- `distill_train_simple.py`：最终使用的蒸馏训练脚本
- `analyze.py`：实验分析脚本
- `inference.py`：推理脚本
- `demo.py`：Streamlit 演示脚本
- `ppt/ppt_content.md`：3 页 PPT 文案
- `speech/speech_3min.md`：3 分钟讲稿
- `figures/`：关键图表
- `docs/`：项目记忆和工作日志

## 未包含内容

为了避免文件过大或误传敏感材料，本包没有包含：

- 原始数据集 `dataset/`
- 模型权重 `*.pt`
- 训练输出目录 `runs/`
- 大型训练日志 `*.log`
- Python 缓存文件

## 推荐用法

把整个文件夹压缩后上传给网页版 AI，或推送到 GitHub 私有仓库。然后把 `LEARN_ME.md` 里的内容作为第一条消息，让 AI 按面试准备的角度带你学习。
