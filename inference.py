"""
推理脚本：对单张或多张图片进行信号检测
"""
import os
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

CLASS_NAMES = {0: "WiFi", 1: "Bluetooth", 2: "ZigBee", 3: "Lightbridge", 4: "XPD"}

# 信号特征描述（用于AI解释模块）
SIGNAL_PROFILES = {
    "WiFi": {
        "modulation": "OFDM",
        "bandwidth": "20-160 MHz",
        "frequency": "2.4/5 GHz",
        "feature": "宽带OFDM信号，频谱占用较宽，通常呈连续块状",
        "application": "无线局域网通信",
    },
    "Bluetooth": {
        "modulation": "GFSK/DQPSK",
        "bandwidth": "1 MHz",
        "frequency": "2.402-2.480 GHz",
        "feature": "窄带跳频信号，频谱呈离散短脉冲，79个信道快速切换",
        "application": "短距离无线通信",
    },
    "ZigBee": {
        "modulation": "DSSS/O-QPSK",
        "bandwidth": "2 MHz",
        "frequency": "2.4 GHz",
        "feature": "窄带DSSS信号，频谱集中且持续时间短",
        "application": "物联网传感器通信",
    },
    "Lightbridge": {
        "modulation": "OFDM",
        "bandwidth": "10 MHz",
        "frequency": "2.4 GHz",
        "feature": "无人机图传信号，频谱特征与WiFi类似但带宽较窄",
        "application": "无人机通信链路",
    },
    "XPD": {
        "modulation": "FM",
        "bandwidth": "6 MHz",
        "frequency": "2.4 GHz",
        "feature": "无线麦克风FM信号，频谱呈连续窄带",
        "application": "无线音频传输",
    },
}


def detect(model_path, image_path, conf=0.25, save_dir=None):
    """对图片进行检测"""
    model = YOLO(model_path)

    # 推理
    t0 = time.time()
    results = model.predict(source=image_path, conf=conf, verbose=False, imgsz=512)
    inference_time = (time.time() - t0) * 1000  # ms

    # 解析结果
    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf_val = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                detections.append({
                    "class": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "confidence": conf_val,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

    return detections, inference_time, results


def generate_ai_explanation(detections):
    """根据检测结果生成AI解释"""
    if not detections:
        return "未检测到任何已知信号。可能频谱空闲，或存在未知类型的信号。"

    # 统计检测到的信号类型
    class_counts = {}
    for det in detections:
        cls_name = det["class"]
        if cls_name not in class_counts:
            class_counts[cls_name] = 0
        class_counts[cls_name] += 1

    # 生成解释
    explanation = "## 频谱检测结果分析\n\n"
    explanation += f"共检测到 **{len(detections)}** 个信号目标，涉及 **{len(class_counts)}** 种信号类型。\n\n"

    # 各类信号分析
    explanation += "### 信号详情\n\n"
    for cls_name, count in class_counts.items():
        profile = SIGNAL_PROFILES.get(cls_name, {})
        explanation += f"**{cls_name}** (×{count})\n"
        explanation += f"- 调制方式: {profile.get('modulation', '未知')}\n"
        explanation += f"- 带宽: {profile.get('bandwidth', '未知')}\n"
        explanation += f"- 特征: {profile.get('feature', '未知')}\n"
        explanation += f"- 应用: {profile.get('application', '未知')}\n\n"

    # 频谱共存分析
    if len(class_counts) > 1:
        explanation += "### 频谱共存分析\n\n"
        detected_classes = list(class_counts.keys())
        explanation += f"当前频谱存在多协议共存: {', '.join(detected_classes)}。\n"

        if "WiFi" in detected_classes and "Bluetooth" in detected_classes:
            explanation += "- WiFi与Bluetooth共存于2.4GHz ISM频段，存在潜在干扰风险。\n"
        if "Lightbridge" in detected_classes:
            explanation += "- 检测到无人机图传信号(Lightbridge)，建议关注低空飞行器活动。\n"
        if "ZigBee" in detected_classes and "WiFi" in detected_classes:
            explanation += "- ZigBee与WiFi频谱重叠，ZigBee可能受WiFi干扰影响。\n"
        explanation += "\n建议: 适用于低空频谱监测与干扰识别场景。\n"
    else:
        cls_name = list(class_counts.keys())[0]
        explanation += f"### 分析结论\n\n当前频谱以{cls_name}信号为主。\n"

    return explanation


def main():
    if len(sys.argv) < 3:
        print("用法: python inference.py <model_path> <image_path> [conf]")
        print("示例: python inference.py runs/v8n_distilled/weights/best.pt test.jpg 0.25")
        print("\n可用模型:")
        for exp in ["v8n_full", "v8m_full", "v8n_fewshot", "v8n_distilled"]:
            pt = os.path.join(RUNS_DIR, exp, "weights", "best.pt")
            status = "✅" if os.path.exists(pt) else "❌"
            print(f"  {status} {exp}: {pt}")
        return

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        return

    detections, inference_time, results = detect(model_path, image_path, conf)

    print(f"\n{'='*50}")
    print(f"  检测结果")
    print(f"{'='*50}")
    print(f"  推理时间: {inference_time:.1f} ms")
    print(f"  检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(f"    {i+1}. {det['class']} (置信度: {det['confidence']:.3f}, "
              f"位置: {det['bbox']})")

    explanation = generate_ai_explanation(detections)
    print(f"\n{'='*50}")
    print("  AI 分析")
    print(f"{'='*50}")
    print(explanation)


if __name__ == "__main__":
    main()
