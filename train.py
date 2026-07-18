"""
训练脚本：4组对比实验
  ① YOLOv8n 全量训练 (Student上界)
  ② YOLOv8m 全量训练 (Teacher)
  ③ YOLOv8n 小样本训练 (退化)
  ④ YOLOv8n 小样本+伪标签蒸馏 (恢复) — 在 distill.py 中完成
"""
import sys
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
DATA_FEWSHOT_YAML = os.path.join(BASE_DIR, "dataset", "data_fewshot.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# 训练参数
EPOCHS = 100
IMG_SIZE = 512
BATCH_SIZE = 8  # 减小batch避免内存不足
DEVICE = 0  # GPU
WORKERS = 2   # 减少数据加载进程数

def train_experiment(model_name, data_yaml, exp_name, epochs=EPOCHS):
    """训练一组实验"""
    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}")
    print(f"  模型: {model_name}")
    print(f"  数据: {data_yaml}")
    print(f"  Epochs: {epochs}, Batch: {BATCH_SIZE}, ImgSize: {IMG_SIZE}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=RUNS_DIR,
        name=exp_name,
        exist_ok=True,
        patience=30,  # 早停
        verbose=True,
        seed=42,
        amp=False,  # 禁用AMP检查（避免下载参考模型失败）
        workers=WORKERS,  # 减少数据加载进程
    )
    return results

def main():
    # 获取命令行参数选择训练哪个实验
    if len(sys.argv) < 2:
        print("用法: python train.py <experiment_name>")
        print("可选实验:")
        print("  v8n_full    - YOLOv8n 全量训练")
        print("  v8m_full    - YOLOv8m 全量训练 (Teacher)")
        print("  v8n_fewshot - YOLOv8n 小样本训练")
        print("  all         - 依次训练全部 (不含蒸馏)")
        return

    exp = sys.argv[1]

    if exp == "v8n_full":
        train_experiment("yolov8n.pt", DATA_YAML, "v8n_full")
    elif exp == "v8m_full":
        train_experiment("yolov8m.pt", DATA_YAML, "v8m_full")
    elif exp == "v8n_fewshot":
        train_experiment("yolov8n.pt", DATA_FEWSHOT_YAML, "v8n_fewshot")
    elif exp == "all":
        train_experiment("yolov8n.pt", DATA_YAML, "v8n_full")
        train_experiment("yolov8m.pt", DATA_YAML, "v8m_full")
        train_experiment("yolov8n.pt", DATA_FEWSHOT_YAML, "v8n_fewshot")
    else:
        print(f"未知实验: {exp}")

if __name__ == "__main__":
    main()
