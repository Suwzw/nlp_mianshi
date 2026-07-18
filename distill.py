"""
伪标签知识蒸馏脚本

流程:
1. 加载训练好的 Teacher (YOLOv8m)
2. Teacher 对 few-shot 训练集推理，生成伪标签
3. 将伪标签与原始标签合并 (软标签增强)
4. Student (YOLOv8n) 在合并后的数据上训练

这是知识蒸馏的一种形式 (Label/Response Distillation):
  - Teacher 的检测知识通过伪标签传递给 Student
  - Student 不仅学习人工标注，还学习 Teacher 的检测模式
"""
import os
import sys
import shutil
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
DATA_FEWSHOT_YAML = os.path.join(BASE_DIR, "dataset", "data_fewshot.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# Teacher 模型路径 (YOLOv8m, 大模型)
TEACHER_PATH = os.path.join(RUNS_DIR, "v8m_full", "weights", "best.pt")
# Few-shot 训练集路径
FEWSHOT_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "train_fewshot")
FEWSHOT_LBL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train_fewshot")
# 蒸馏数据集目录 (few-shot + teacher伪标签)
DISTILL_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "train_distill")
DISTILL_LBL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train_distill")

# 训练参数
EPOCHS = 100
IMG_SIZE = 512
BATCH_SIZE = 8
DEVICE = 0
WORKERS = 2


def generate_pseudo_labels(teacher_model_path, img_dir, lbl_dir, conf_thresh=0.7):
    """
    用 Teacher 模型对图片推理，生成伪标签
    策略: 对于每张图片，将 Teacher 的预测与原始标签合并
          - 如果 Teacher 预测的框与原始标签 IoU 高，保留原始标签 (更准确)
          - 如果 Teacher 预测了新框 (原始标签没有)，作为伪标签加入
    """
    print(f"\n{'='*60}")
    print(f"  伪标签生成 (Teacher: {teacher_model_path})")
    print(f"  置信度阈值: {conf_thresh}")
    print(f"{'='*60}\n")

    teacher = YOLO(teacher_model_path)

    # 创建蒸馏数据目录
    os.makedirs(DISTILL_IMG_DIR, exist_ok=True)
    os.makedirs(DISTILL_LBL_DIR, exist_ok=True)

    # 获取 few-shot 图片列表
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"  待处理图片数: {len(img_files)}")

    # 对每张图片: Teacher 推理 → 合并标签
    results = teacher.predict(
        source=img_dir,
        conf=conf_thresh,
        save=False,
        verbose=False,
        imgsz=IMG_SIZE,
    )

    total_original_boxes = 0
    total_teacher_boxes = 0
    total_merged_boxes = 0
    images_with_new_boxes = 0

    for result in results:
        img_name = os.path.basename(result.path)
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))

        # 复制图片到蒸馏目录
        shutil.copy2(img_path, os.path.join(DISTILL_IMG_DIR, img_name))

        # 读取原始标签
        original_labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        original_labels.append({
                            'cls': int(parts[0]),
                            'cx': float(parts[1]),
                            'cy': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4]),
                        })
        total_original_boxes += len(original_labels)

        # 获取 Teacher 预测
        teacher_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf = result.boxes.conf[i].item()
                # XYXY → XYWH normalized
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                teacher_boxes.append({
                    'cls': cls_id,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'conf': conf,
                })
        total_teacher_boxes += len(teacher_boxes)

        # 合并策略: 以原始标签为主，加入 Teacher 检测到的"新"框
        # (与原始标签 IoU > 0.5 的视为重复，不加入)
        merged_labels = list(original_labels)
        new_boxes_added = 0

        for tbox in teacher_boxes:
            is_duplicate = False
            for obox in original_labels:
                # 简单IoU计算
                iou = compute_iou(tbox, obox)
                if iou > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_labels.append(tbox)
                new_boxes_added += 1

        if new_boxes_added > 0:
            images_with_new_boxes += 1

        total_merged_boxes += len(merged_labels)

        # 写入合并后的标签
        distill_lbl_path = os.path.join(DISTILL_LBL_DIR, img_name.replace('.jpg', '.txt'))
        with open(distill_lbl_path, 'w') as f:
            for box in merged_labels:
                f.write(f"{box['cls']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}\n")

    print(f"\n  统计:")
    print(f"    原始标注框: {total_original_boxes}")
    print(f"    Teacher检测框: {total_teacher_boxes}")
    print(f"    合并后总框数: {total_merged_boxes}")
    print(f"    新增框数: {total_merged_boxes - total_original_boxes}")
    print(f"    有新增框的图片: {images_with_new_boxes}/{len(img_files)}")
    print(f"    蒸馏数据已保存到: {DISTILL_IMG_DIR}")

    return total_merged_boxes, total_original_boxes


def compute_iou(box1, box2):
    """计算两个YOLO格式框的IoU"""
    # XYWH → XYXY
    b1_x1 = box1['cx'] - box1['w'] / 2
    b1_y1 = box1['cy'] - box1['h'] / 2
    b1_x2 = box1['cx'] + box1['w'] / 2
    b1_y2 = box1['cy'] + box1['h'] / 2

    b2_x1 = box2['cx'] - box2['w'] / 2
    b2_y1 = box2['cy'] - box2['h'] / 2
    b2_x2 = box2['cx'] + box2['w'] / 2
    b2_y2 = box2['cy'] + box2['h'] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1['w'] * box1['h']
    b2_area = box2['w'] * box2['h']

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def create_distill_yaml():
    """创建蒸馏数据集的 data.yaml"""
    yaml_content = f"""# 蒸馏训练数据集 (Few-shot + Teacher伪标签)
path: {os.path.join(BASE_DIR, 'dataset').replace(os.sep, '/')}
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
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"  data_distill.yaml 已生成: {yaml_path}")
    return yaml_path


def train_distilled_student(distill_yaml):
    """用蒸馏数据训练 Student"""
    print(f"\n{'='*60}")
    print(f"  蒸馏训练: YOLOv8n + Teacher伪标签")
    print(f"{'='*60}\n")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=distill_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=RUNS_DIR,
        name="v8n_distilled",
        exist_ok=True,
        patience=30,
        verbose=True,
        seed=42,
        amp=False,
        workers=WORKERS,
    )
    return results


def main():
    if len(sys.argv) < 2:
        print("用法: python distill.py <step>")
        print("  step 1: generate  - 生成伪标签")
        print("  step 2: train     - 蒸馏训练Student")
        print("  step 3: all       - 全部执行")
        return

    step = sys.argv[1]

    if not os.path.exists(TEACHER_PATH):
        print(f"错误: Teacher模型不存在: {TEACHER_PATH}")
        print("请先运行: python train.py v8m_full")
        return

    if step in ("generate", "all"):
        generate_pseudo_labels(TEACHER_PATH, FEWSHOT_IMG_DIR, FEWSHOT_LBL_DIR)

    if step in ("train", "all"):
        yaml_path = create_distill_yaml()
        train_distilled_student(yaml_path)

    print("\n✅ 蒸馏流程完成!")


if __name__ == "__main__":
    main()
