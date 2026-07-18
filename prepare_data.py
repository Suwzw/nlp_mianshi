"""
数据集处理脚本：按录制场次划分 train/val/test，并生成 few-shot 子集
"""
import os
import shutil
import random
from collections import defaultdict

# ============ 配置 ============
SRC_DIR = "../低空五类宽带信号数据集/rot90_compress_anechoic"
DST_DIR = "./dataset"
CLASS_NAMES = {0: "WiFi", 1: "Bluetooth", 2: "ZigBee", 3: "Lightbridge", 4: "XPD"}
FEWSHOT_PER_CLASS = 50  # 每类50张图片
RANDOM_SEED = 42

# 按录制场次划分（避免数据泄露）
# Train: 6场, Val: 2场, Test: 1场
SPLIT_MAP = {
    "rec_154": "train", "rec_155": "train",
    "rec_180": "train", "rec_181": "train",
    "rec_182": "train", "rec_183": "train",
    "rec_184": "val",   "rec_185": "val",
    "rec_186": "test",
}

random.seed(RANDOM_SEED)

def get_rec_id(filename):
    """从文件名提取录制编号，如 rec_154_pic_0_rot90.jpg -> rec_154"""
    return filename.split("_pic_")[0]

def parse_label(label_path):
    """解析YOLO标签文件，返回 [(class_id, ...), ...]"""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(int(parts[0]))
    return boxes

def main():
    src_dir = os.path.join(os.path.dirname(__file__), SRC_DIR)
    dst_dir = os.path.join(os.path.dirname(__file__), DST_DIR)

    # 统计
    split_stats = {"train": 0, "val": 0, "test": 0}
    rec_stats = defaultdict(lambda: {"images": 0, "classes": defaultdict(int)})

    # 按录制场次分类文件
    files_by_rec = defaultdict(list)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith('.jpg'):
            rec_id = get_rec_id(f)
            files_by_rec[rec_id].append(f)

    print("=" * 60)
    print("数据集按录制场次划分")
    print("=" * 60)
    print(f"{'录制编号':<12} {'划分':<8} {'图片数':<8} {'各类别标注数'}")
    print("-" * 60)

    # 复制文件到对应目录
    for rec_id, split in SPLIT_MAP.items():
        if rec_id not in files_by_rec:
            print(f"警告: {rec_id} 不存在于数据集中")
            continue

        img_dst = os.path.join(dst_dir, "images", split)
        lbl_dst = os.path.join(dst_dir, "labels", split)

        for img_file in files_by_rec[rec_id]:
            # 复制图片
            shutil.copy2(
                os.path.join(src_dir, img_file),
                os.path.join(img_dst, img_file)
            )
            # 复制标签
            lbl_file = img_file.replace('.jpg', '.txt')
            lbl_src = os.path.join(src_dir, lbl_file)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(lbl_dst, lbl_file))
                # 统计类别
                classes = parse_label(lbl_src)
                for cls in classes:
                    rec_stats[rec_id]["classes"][cls] += 1

            rec_stats[rec_id]["images"] += 1
            split_stats[split] += 1

        classes_str = ", ".join([f"{CLASS_NAMES[k]}:{v}" for k, v in sorted(rec_stats[rec_id]["classes"].items())])
        print(f"{rec_id:<12} {split:<8} {rec_stats[rec_id]['images']:<8} {classes_str}")

    print("-" * 60)
    print(f"{'总计':<12} {'':<8} {sum(split_stats.values()):<8}")
    for split, count in split_stats.items():
        print(f"  {split}: {count} 张")
    print()

    # ============ 生成 few-shot 子集 ============
    print("=" * 60)
    print(f"生成 Few-shot 子集 (每类 {FEWSHOT_PER_CLASS} 张)")
    print("=" * 60)

    train_img_dir = os.path.join(dst_dir, "images", "train")
    train_lbl_dir = os.path.join(dst_dir, "labels", "train")
    fewshot_img_dir = os.path.join(dst_dir, "images", "train_fewshot")
    fewshot_lbl_dir = os.path.join(dst_dir, "labels", "train_fewshot")

    # 收集每类包含该类别的图片
    class_images = defaultdict(list)
    for lbl_file in os.listdir(train_lbl_dir):
        if not lbl_file.endswith('.txt'):
            continue
        lbl_path = os.path.join(train_lbl_dir, lbl_file)
        classes = parse_label(lbl_path)
        unique_classes = set(classes)
        img_file = lbl_file.replace('.txt', '.jpg')
        for cls in unique_classes:
            class_images[cls].append(img_file)

    fewshot_selected = set()
    for cls, imgs in sorted(class_images.items()):
        selected = random.sample(imgs, min(FEWSHOT_PER_CLASS, len(imgs)))
        fewshot_selected.update(selected)
        print(f"  {CLASS_NAMES[cls]:<12}: 可选 {len(imgs)} 张, 选中 {len(selected)} 张")

    # 复制 few-shot 文件
    for img_file in fewshot_selected:
        shutil.copy2(
            os.path.join(train_img_dir, img_file),
            os.path.join(fewshot_img_dir, img_file)
        )
        lbl_file = img_file.replace('.jpg', '.txt')
        lbl_src = os.path.join(train_lbl_dir, lbl_file)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(fewshot_lbl_dir, lbl_file))

    print(f"\n  Few-shot 总图片数: {len(fewshot_selected)} 张")
    print(f"  (注意: 一张图片可能包含多个类别目标，因此总数 < 5 × {FEWSHOT_PER_CLASS})")

    # ============ 生成 data.yaml ============
    yaml_content = f"""# 低空五类宽带信号检测数据集
# 按录制场次划分，避免数据泄露
path: {os.path.abspath(dst_dir).replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

# 类别
names:
  0: WiFi
  1: Bluetooth
  2: ZigBee
  3: Lightbridge
  4: XPD
"""
    yaml_path = os.path.join(dst_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\n  data.yaml 已生成: {yaml_path}")

    # few-shot yaml
    yaml_fewshot = yaml_content.replace("train: images/train", "train: images/train_fewshot")
    yaml_fewshot_path = os.path.join(dst_dir, "data_fewshot.yaml")
    with open(yaml_fewshot_path, 'w', encoding='utf-8') as f:
        f.write(yaml_fewshot)
    print(f"  data_fewshot.yaml 已生成: {yaml_fewshot_path}")

    print("\n✅ 数据集处理完成!")

if __name__ == "__main__":
    main()
