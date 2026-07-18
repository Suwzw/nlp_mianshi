"""独立蒸馏训练脚本"""
import os
import sys

def main():
    from ultralytics import YOLO

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUNS_DIR = os.path.join(BASE_DIR, "runs")

    # 创建yaml
    yaml_content = f"""path: {BASE_DIR}/dataset
train: images/train_distill
val: images/val
test: images/test

names:
  0: WiFi
  1: Bluetooth
  2: ZigBee
  3: Lightbridge
  4: XPD
"""
    yaml_path = os.path.join(BASE_DIR, "dataset", "data_distill.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML: {yaml_path}")

    model = YOLO("yolov8n.pt")
    print("Starting distillation training...")
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=8,
        device=0,
        project=RUNS_DIR,
        name="v8n_distilled",
        exist_ok=True,
        patience=30,
        seed=42,
        amp=False,
        workers=0,  # Windows多进程兼容
    )
    print("DONE!")

if __name__ == '__main__':
    main()
