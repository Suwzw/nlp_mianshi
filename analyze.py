"""
实验分析脚本：深度分析4组对比实验结果
  1. mAP 对比表格
  2. 逐类 AP 分析 (哪类信号受小样本影响最大/从蒸馏受益最多)
  3. 参数量/FLOPs/FPS 统计
  4. 检测结果可视化 (成功+失败案例)
  5. t-SNE 特征可视化
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO
from pathlib import Path

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
TEST_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "test")

os.makedirs(FIGURES_DIR, exist_ok=True)

CLASS_NAMES = {0: "WiFi", 1: "Bluetooth", 2: "ZigBee", 3: "Lightbridge", 4: "XPD"}
EXPERIMENTS = {
    "v8n_full": {"name": "YOLOv8n\n(全量)", "color": "#2196F3"},
    "v8m_full": {"name": "YOLOv8m\n(Teacher)", "color": "#4CAF50"},
    "v8n_fewshot": {"name": "YOLOv8n\n(小样本)", "color": "#FF9800"},
    "v8n_distilled": {"name": "YOLOv8n\n(蒸馏)", "color": "#F44336"},
}


def load_results(exp_name):
    """加载实验结果"""
    results_path = os.path.join(RUNS_DIR, exp_name, "results.json")
    if not os.path.exists(results_path):
        # 尝试从 results.csv 加载
        csv_path = os.path.join(RUNS_DIR, exp_name, "results.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            return df
        print(f"  警告: {exp_name} 结果文件不存在")
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


def get_map_from_results(exp_name):
    """从验证结果中获取 mAP50 和 mAP50-95"""
    # 尝试从 best.pt 验证
    best_pt = os.path.join(RUNS_DIR, exp_name, "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"  警告: {exp_name} 的 best.pt 不存在")
        return None, None, None

    model = YOLO(best_pt)
    results = model.val(data=DATA_YAML, split="test", verbose=False, imgsz=512)

    # 逐类 AP
    per_class_ap = {}
    if hasattr(results, 'boxes'):
        ap50 = results.box.ap50  # per-class AP50
        ap = results.box.ap  # per-class AP50-95
        for i, name in CLASS_NAMES.items():
            if i < len(ap50):
                per_class_ap[name] = {
                    "ap50": float(ap50[i]),
                    "ap50_95": float(ap[i]),
                }

    return float(results.box.map50), float(results.box.map), per_class_ap


def plot_map_comparison(map_data):
    """绘制 mAP 对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = list(map_data.keys())
    map50_values = [map_data[e]["mAP50"] for e in exp_names]
    map5095_values = [map_data[e]["mAP50-95"] for e in exp_names]
    labels = [EXPERIMENTS[e]["name"] for e in exp_names]
    colors = [EXPERIMENTS[e]["color"] for e in exp_names]

    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, map50_values, width, label='mAP@50', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, map5095_values, width, label='mAP@50-95',
                   color=colors, alpha=0.5, hatch='//')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('mAP', fontsize=14)
    ax.set_title('四组实验 mAP 对比', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "map_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ mAP对比图: {save_path}")


def plot_per_class_ap(per_class_data):
    """绘制逐类 AP 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    class_names = list(CLASS_NAMES.values())
    exp_names = list(per_class_data.keys())

    for idx, metric in enumerate(["ap50", "ap50_95"]):
        ax = axes[idx]
        x = np.arange(len(class_names))
        width = 0.8 / len(exp_names)

        for i, exp in enumerate(exp_names):
            values = [per_class_data[exp].get(cls, {}).get(metric, 0) * 100 for cls in class_names]
            bars = ax.bar(x + i * width - 0.4 + width/2, values, width,
                         label=EXPERIMENTS[exp]["name"], color=EXPERIMENTS[exp]["color"], alpha=0.85)

        ax.set_ylabel('AP', fontsize=12)
        ax.set_title(f'逐类 AP{"@50" if "50" in metric else "@50-95"}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "per_class_ap.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 逐类AP图: {save_path}")


def plot_distillation_gain(per_class_data):
    """绘制蒸馏增益图: (蒸馏AP - 小样本AP) / 小样本AP"""
    if "v8n_fewshot" not in per_class_data or "v8n_distilled" not in per_class_data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    class_names = list(CLASS_NAMES.values())
    gains = []
    for cls in class_names:
        fewshot_ap = per_class_data["v8n_fewshot"].get(cls, {}).get("ap50", 0) * 100
        distilled_ap = per_class_data["v8n_distilled"].get(cls, {}).get("ap50", 0) * 100
        if fewshot_ap > 0:
            gain = ((distilled_ap - fewshot_ap) / fewshot_ap) * 100
        else:
            gain = 0
        gains.append(gain)

    colors = ['#F44336' if g > 0 else '#9E9E9E' for g in gains]
    bars = ax.bar(class_names, gains, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, gain in zip(bars, gains):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gain:+.1f}%', ha='center', va=va, fontsize=12, fontweight='bold')

    ax.set_ylabel('蒸馏增益 (%)', fontsize=14)
    ax.set_title('知识蒸馏对各信号类别的精度增益', fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(min(gains) - 10, max(gains) + 10)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "distillation_gain.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 蒸馏增益图: {save_path}")


def count_parameters_and_flops():
    """统计模型参数量、FLOPs"""
    results = {}
    for exp_name in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        best_pt = os.path.join(RUNS_DIR, exp_name, "weights", "best.pt")
        if not os.path.exists(best_pt):
            continue
        model = YOLO(best_pt)
        info = model.info(verbose=False)

        # 获取参数量和FLOPs
        n_params = sum(p.numel() for p in model.model.parameters())
        n_layers = len(list(model.model.modules()))

        # 模型文件大小
        model_size = os.path.getsize(best_pt) / (1024 * 1024)  # MB

        # FPS 测试
        import time
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # warmup
        for _ in range(5):
            model.predict(dummy, verbose=False, imgsz=512)
        # benchmark
        times = []
        for _ in range(20):
            t0 = time.time()
            model.predict(dummy, verbose=False, imgsz=512)
            times.append(time.time() - t0)
        fps = 1.0 / np.mean(times)

        results[exp_name] = {
            "params_M": n_params / 1e6,
            "model_size_MB": model_size,
            "fps": fps,
        }
        print(f"  {exp_name}: {n_params/1e6:.2f}M params, {model_size:.1f}MB, {fps:.1f} FPS")

    return results


def plot_model_comparison(model_stats):
    """绘制模型参数/速度对比"""
    if not model_stats:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    exp_names = list(model_stats.keys())
    labels = [EXPERIMENTS[e]["name"] for e in exp_names]
    colors = [EXPERIMENTS[e]["color"] for e in exp_names]

    # 参数量
    ax = axes[0]
    params = [model_stats[e]["params_M"] for e in exp_names]
    bars = ax.bar(labels, params, color=colors, alpha=0.85)
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{v:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('参数量 (M)', fontsize=12)
    ax.set_title('模型参数量', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 模型大小
    ax = axes[1]
    sizes = [model_stats[e]["model_size_MB"] for e in exp_names]
    bars = ax.bar(labels, sizes, color=colors, alpha=0.85)
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{v:.1f}MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('模型大小 (MB)', fontsize=12)
    ax.set_title('模型文件大小', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # FPS
    ax = axes[2]
    fps_vals = [model_stats[e]["fps"] for e in exp_names]
    bars = ax.bar(labels, fps_vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{v:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('推理速度', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 模型对比图: {save_path}")


def plot_loss_curves():
    """绘制训练Loss曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        csv_path = os.path.join(RUNS_DIR, exp_name, "results.csv")
        if not os.path.exists(csv_path):
            continue
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        epochs = df['epoch']
        box_loss = df['train/box_loss']
        cls_loss = df['train/cls_loss']

        label = EXPERIMENTS[exp_name]["name"].replace('\n', ' ')
        color = EXPERIMENTS[exp_name]["color"]

        axes[0].plot(epochs, box_loss, label=label, color=color, linewidth=1.5)
        axes[1].plot(epochs, cls_loss, label=label, color=color, linewidth=1.5)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Box Loss', fontsize=12)
    axes[0].set_title('训练 Box Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Cls Loss', fontsize=12)
    axes[1].set_title('训练 Cls Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "loss_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Loss曲线: {save_path}")


def save_detection_examples(exp_name, num_images=10):
    """保存检测结果可视化"""
    best_pt = os.path.join(RUNS_DIR, exp_name, "weights", "best.pt")
    if not os.path.exists(best_pt):
        return

    model = YOLO(best_pt)
    save_dir = os.path.join(FIGURES_DIR, f"detections_{exp_name}")
    os.makedirs(save_dir, exist_ok=True)

    results = model.predict(
        source=TEST_IMG_DIR,
        save=True,
        project=save_dir,
        name=exp_name,
        exist_ok=True,
        conf=0.25,
        verbose=False,
        imgsz=512,
    )

    print(f"  ✅ 检测结果: {save_dir}/{exp_name}/")


def generate_summary_table(map_data, per_class_data, model_stats):
    """生成汇总表格"""
    table = "# 实验结果汇总\n\n"

    # mAP对比表
    table += "## ① mAP 对比\n\n"
    table += "| 模型 | 训练数据 | mAP@50 | mAP@50-95 |\n"
    table += "|------|----------|--------|----------|\n"
    data_desc = {
        "v8n_full": "全量(1902张)",
        "v8m_full": "全量(1902张)",
        "v8n_fewshot": "小样本(232张)",
        "v8n_distilled": "小样本+伪标签",
    }
    for exp in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        if exp in map_data:
            table += f"| {EXPERIMENTS[exp]['name'].replace(chr(10), ' ')} | {data_desc[exp]} | {map_data[exp]['mAP50']:.2f} | {map_data[exp]['mAP50-95']:.2f} |\n"

    # 逐类AP
    table += "\n## ② 逐类 AP@50\n\n"
    table += "| 模型 | WiFi | Bluetooth | ZigBee | Lightbridge | XPD |\n"
    table += "|------|------|-----------|--------|-------------|-----|\n"
    for exp in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        if exp in per_class_data:
            row = f"| {EXPERIMENTS[exp]['name'].replace(chr(10), ' ')} "
            for cls in ["WiFi", "Bluetooth", "ZigBee", "Lightbridge", "XPD"]:
                ap = per_class_data[exp].get(cls, {}).get("ap50", 0) * 100
                row += f"| {ap:.1f} "
            row += "|\n"
            table += row

    # 模型参数
    table += "\n## ③ 模型参数统计\n\n"
    table += "| 模型 | 参数量(M) | 模型大小(MB) | FPS |\n"
    table += "|------|-----------|-------------|-----|\n"
    for exp in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        if exp in model_stats:
            s = model_stats[exp]
            table += f"| {EXPERIMENTS[exp]['name'].replace(chr(10), ' ')} | {s['params_M']:.2f} | {s['model_size_MB']:.1f} | {s['fps']:.0f} |\n"

    # 蒸馏增益
    if "v8n_fewshot" in per_class_data and "v8n_distilled" in per_class_data:
        table += "\n## ④ 蒸馏增益分析\n\n"
        table += "| 类别 | 小样本AP@50 | 蒸馏后AP@50 | 增益(%) |\n"
        table += "|------|-----------|-----------|---------|\n"
        for cls in ["WiFi", "Bluetooth", "ZigBee", "Lightbridge", "XPD"]:
            fewshot_ap = per_class_data["v8n_fewshot"].get(cls, {}).get("ap50", 0) * 100
            distilled_ap = per_class_data["v8n_distilled"].get(cls, {}).get("ap50", 0) * 100
            if fewshot_ap > 0:
                gain = ((distilled_ap - fewshot_ap) / fewshot_ap) * 100
            else:
                gain = 0
            table += f"| {cls} | {fewshot_ap:.1f} | {distilled_ap:.1f} | {gain:+.1f}% |\n"

    save_path = os.path.join(FIGURES_DIR, "summary.md")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f"  ✅ 汇总表格: {save_path}")
    print("\n" + table)
    return table


def main():
    print("=" * 60)
    print("  实验结果深度分析")
    print("=" * 60)

    # 1. 获取各实验的 mAP
    print("\n📊 正在评估各模型在测试集上的表现...")
    map_data = {}
    per_class_data = {}

    for exp_name in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
        best_pt = os.path.join(RUNS_DIR, exp_name, "weights", "best.pt")
        if not os.path.exists(best_pt):
            print(f"  ⚠️ {exp_name}: best.pt 不存在，跳过")
            continue
        print(f"  评估 {exp_name}...")
        map50, map5095, per_class = get_map_from_results(exp_name)
        if map50 is not None:
            map_data[exp_name] = {"mAP50": map50 * 100, "mAP50-95": map5095 * 100}
            per_class_data[exp_name] = per_class
            print(f"    mAP@50={map50*100:.2f}, mAP@50-95={map5095*100:.2f}")

    if not map_data:
        print("  ❌ 没有找到任何实验结果")
        return

    # 2. 绘制图表
    print("\n📈 生成可视化图表...")
    plot_map_comparison(map_data)

    if per_class_data:
        plot_per_class_ap(per_class_data)
        plot_distillation_gain(per_class_data)

    plot_loss_curves()

    # 3. 模型参数统计
    print("\n🔧 统计模型参数...")
    model_stats = count_parameters_and_flops()
    plot_model_comparison(model_stats)

    # 4. 检测结果可视化
    print("\n🖼️ 保存检测结果...")
    for exp_name in ["v8n_full", "v8n_distilled"]:
        save_detection_examples(exp_name)

    # 5. 汇总表格
    print("\n📋 生成汇总表格...")
    generate_summary_table(map_data, per_class_data, model_stats)

    print("\n✅ 分析完成! 所有图表已保存到 figures/ 目录")


if __name__ == "__main__":
    main()
