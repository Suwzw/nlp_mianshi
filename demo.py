"""
Streamlit Demo: 低空宽带信号检测系统
上传图片 → 检测 → 显示结果 + AI解释
"""
import os
import time
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

CLASS_NAMES = {0: "WiFi", 1: "Bluetooth", 2: "ZigBee", 3: "Lightbridge", 4: "XPD"}
CLASS_COLORS = {
    "WiFi": (255, 0, 0),       # 红
    "Bluetooth": (0, 0, 255),   # 蓝
    "ZigBee": (0, 255, 0),      # 绿
    "Lightbridge": (0, 165, 255), # 橙
    "XPD": (128, 0, 128),       # 紫
}

SIGNAL_PROFILES = {
    "WiFi": {"modulation": "OFDM", "bandwidth": "20-160 MHz", "feature": "宽带OFDM信号，频谱占用较宽", "application": "无线局域网"},
    "Bluetooth": {"modulation": "GFSK", "bandwidth": "1 MHz", "feature": "窄带跳频信号，79个信道快速切换", "application": "短距离无线通信"},
    "ZigBee": {"modulation": "DSSS", "bandwidth": "2 MHz", "feature": "窄带DSSS信号，持续时间短", "application": "物联网传感器"},
    "Lightbridge": {"modulation": "OFDM", "bandwidth": "10 MHz", "feature": "无人机图传信号", "application": "无人机通信链路"},
    "XPD": {"modulation": "FM", "bandwidth": "6 MHz", "feature": "无线麦克风FM信号", "application": "无线音频传输"},
}


@st.cache_resource
def load_model(model_path):
    """加载模型（缓存）"""
    return YOLO(model_path)


def get_available_models():
    """获取可用模型列表"""
    models = {}
    for exp_name, label in [
        ("v8n_full", "YOLOv8n (全量训练)"),
        ("v8m_full", "YOLOv8m (Teacher)"),
        ("v8n_fewshot", "YOLOv8n (小样本)"),
        ("v8n_distilled", "YOLOv8n (蒸馏)"),
    ]:
        pt_path = os.path.join(RUNS_DIR, exp_name, "weights", "best.pt")
        if os.path.exists(pt_path):
            models[label] = pt_path
    return models


def draw_detections(image_np, detections):
    """在图片上绘制检测框"""
    import cv2
    img = image_np.copy()

    for det in detections:
        cls_name = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 标签
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def generate_explanation(detections, inference_time):
    """生成AI解释"""
    if not detections:
        return "🔍 **未检测到已知信号**\n\n可能原因：\n- 频谱空闲\n- 存在未知类型信号\n- 信号SNR过低\n\n建议：增大采集带宽或提高增益。"

    class_counts = {}
    for det in detections:
        cls_name = det["class"]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    exp = f"📊 **检测统计**\n"
    exp += f"- 总目标数: **{len(detections)}**\n"
    exp += f"- 信号类型: **{len(class_counts)}** 种\n"
    exp += f"- 推理耗时: **{inference_time:.1f} ms**\n\n"

    exp += "📡 **信号详情**\n"
    for cls_name, count in class_counts.items():
        profile = SIGNAL_PROFILES.get(cls_name, {})
        exp += f"\n**{cls_name}** (×{count})\n"
        exp += f"- 调制: {profile.get('modulation', '未知')}\n"
        exp += f"- 带宽: {profile.get('bandwidth', '未知')}\n"
        exp += f"- 特征: {profile.get('feature', '未知')}\n"
        exp += f"- 应用: {profile.get('application', '未知')}\n"

    if len(class_counts) > 1:
        exp += f"\n⚠️ **频谱共存分析**\n"
        exp += f"当前2.4GHz ISM频段存在 **{', '.join(class_counts.keys())}** 多协议共存。\n"
        if "WiFi" in class_counts and "Bluetooth" in class_counts:
            exp += "- WiFi与Bluetooth共存，存在潜在干扰\n"
        if "Lightbridge" in class_counts:
            exp += "- 检测到无人机图传信号，建议关注低空飞行器活动\n"
        if "ZigBee" in class_counts and "WiFi" in class_counts:
            exp += "- ZigBee可能受WiFi宽带信号干扰\n"
        exp += "\n✅ 适用场景: 低空频谱监测、干扰识别、无人机检测"
    else:
        cls = list(class_counts.keys())[0]
        exp += f"\n✅ 当前频谱以 **{cls}** 信号为主，频谱环境相对单一。"

    return exp


def main():
    st.set_page_config(page_title="低空宽带信号检测系统", page_icon="📡", layout="wide")

    # 标题
    st.title("📡 低空宽带信号检测系统")
    st.markdown("基于大小模型协同的小样本宽带通信信号检测")

    # 侧边栏
    st.sidebar.header("⚙️ 设置")

    available_models = get_available_models()

    if not available_models:
        st.sidebar.error("未找到训练好的模型！请先运行训练。")
        st.sidebar.info("运行: `python train.py v8n_full`")
        return

    selected_model_label = st.sidebar.selectbox(
        "选择模型",
        list(available_models.keys()),
        index=len(available_models) - 1  # 默认选最后一个（蒸馏模型）
    )
    model_path = available_models[selected_model_label]

    conf_threshold = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)

    # 模型信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 模型信息")
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    st.sidebar.metric("模型大小", f"{model_size:.1f} MB")
    st.sidebar.metric("检测类别", "5类")

    # 主区域
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 上传图片")
        uploaded_file = st.file_uploader(
            "选择频谱图图片",
            type=['jpg', 'jpeg', 'png'],
            help="上传低空宽带信号时频图"
        )

        # 示例图片
        test_dir = os.path.join(BASE_DIR, "dataset", "images", "test")
        if os.path.exists(test_dir):
            test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
            if test_images and st.button("📸 使用示例图片"):
                uploaded_file = open(os.path.join(test_dir, test_images[0]), 'rb')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="输入图片", use_container_width=True)

            if st.button("🔍 检测", type="primary"):
                # 加载模型
                with st.spinner("加载模型..."):
                    model = load_model(model_path)

                # 推理
                import cv2
                img_array = np.array(image.convert('RGB'))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                t0 = time.time()
                results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False, imgsz=512)
                inference_time = (time.time() - t0) * 1000

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
                                "confidence": conf_val,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            })

                # 绘制检测框
                if detections:
                    annotated = draw_detections(img_array, detections)
                    st.session_state['result_image'] = annotated
                    st.session_state['detections'] = detections
                    st.session_state['inference_time'] = inference_time
                else:
                    st.session_state['result_image'] = img_array
                    st.session_state['detections'] = []
                    st.session_state['inference_time'] = inference_time

    with col2:
        st.header("📊 检测结果")

        if 'result_image' in st.session_state:
            st.image(st.session_state['result_image'], caption="检测结果", use_container_width=True)

            detections = st.session_state['detections']
            inference_time = st.session_state['inference_time']

            # 检测统计
            st.metric("推理时间", f"{inference_time:.1f} ms")
            st.metric("检测目标数", len(detections))

            if detections:
                # 检测结果表格
                st.markdown("### 检测详情")
                for i, det in enumerate(detections):
                    st.markdown(
                        f"**{i+1}. {det['class']}** "
                        f"| 置信度: {det['confidence']:.3f} "
                        f"| 位置: {det['bbox']}"
                    )
        else:
            st.info("👆 上传图片并点击「检测」按钮")

    # AI解释模块
    if 'detections' in st.session_state:
        st.markdown("---")
        st.header("🤖 AI 频谱分析")
        explanation = generate_explanation(
            st.session_state['detections'],
            st.session_state.get('inference_time', 0)
        )
        st.markdown(explanation)


if __name__ == "__main__":
    main()
