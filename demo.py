"""
Streamlit demo for the interview project:
基于大小模型协同的小样本低空宽带信号检测
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUNS_DIR = BASE_DIR / "runs"
DATASET_TEST_DIR = BASE_DIR / "dataset" / "images" / "test"
SAMPLE_DIR = PROJECT_ROOT / "sample_images"

CLASS_NAMES = {0: "WiFi", 1: "Bluetooth", 2: "ZigBee", 3: "Lightbridge", 4: "XPD"}
CLASS_COLORS = {
    "WiFi": (239, 68, 68),
    "Bluetooth": (59, 130, 246),
    "ZigBee": (34, 197, 94),
    "Lightbridge": (245, 158, 11),
    "XPD": (168, 85, 247),
}

MODEL_CATALOG = {
    "v8n_distilled": {
        "label": "YOLOv8n 蒸馏 Student",
        "role": "232 张小样本 + Teacher 伪标签训练的轻量模型",
        "map50": "94.9%",
        "map": "78.0%",
        "params": "3.0M",
        "fps": "156",
    },
    "v8n_fewshot": {
        "label": "YOLOv8n 小样本 Student",
        "role": "仅 232 张人工标注，展示小样本基线",
        "map50": "95.2%",
        "map": "78.3%",
        "params": "3.0M",
        "fps": "153",
    },
    "v8n_full": {
        "label": "YOLOv8n 全量 Student",
        "role": "1902 张全量训练，小模型精度上界",
        "map50": "96.4%",
        "map": "81.5%",
        "params": "3.0M",
        "fps": "99",
    },
    "v8m_full": {
        "label": "YOLOv8m Teacher",
        "role": "全量训练的大模型教师，用于生成伪标签",
        "map50": "96.4%",
        "map": "81.9%",
        "params": "25.9M",
        "fps": "42",
    },
}

SIGNAL_PROFILES = {
    "WiFi": {
        "modulation": "OFDM",
        "bandwidth": "20-160 MHz",
        "feature": "宽带块状能量分布，常见于 2.4GHz/5GHz 无线局域网",
        "application": "无线局域网、热点接入",
    },
    "Bluetooth": {
        "modulation": "GFSK/DQPSK",
        "bandwidth": "1 MHz",
        "feature": "窄带跳频，时频图上常呈离散短脉冲",
        "application": "耳机、遥控器、短距设备",
    },
    "ZigBee": {
        "modulation": "DSSS/O-QPSK",
        "bandwidth": "2 MHz",
        "feature": "窄带、低功耗，容易和 WiFi 频谱共存",
        "application": "物联网传感器、智能家居",
    },
    "Lightbridge": {
        "modulation": "OFDM",
        "bandwidth": "10 MHz",
        "feature": "无人机图传链路，带宽较 WiFi 窄但持续性强",
        "application": "无人机低空通信与图传",
    },
    "XPD": {
        "modulation": "FM",
        "bandwidth": "6 MHz",
        "feature": "连续窄带能量轨迹，常用于无线音频传输",
        "application": "无线麦克风、音频链路",
    },
}

EXPERIMENT_ROWS = [
    {"模型": "YOLOv8n 全量", "训练数据": "1902 张", "mAP@50": "96.4%", "mAP@50-95": "81.5%", "参数量": "3.0M", "FPS": "99"},
    {"模型": "YOLOv8m Teacher", "训练数据": "1902 张", "mAP@50": "96.4%", "mAP@50-95": "81.9%", "参数量": "25.9M", "FPS": "42"},
    {"模型": "YOLOv8n 小样本", "训练数据": "232 张", "mAP@50": "95.2%", "mAP@50-95": "78.3%", "参数量": "3.0M", "FPS": "153"},
    {"模型": "YOLOv8n 蒸馏", "训练数据": "232 张 + 伪标签", "mAP@50": "94.9%", "mAP@50-95": "78.0%", "参数量": "3.0M", "FPS": "156"},
]

PER_CLASS_AP_ROWS = [
    {"模型": "YOLOv8n 全量", "WiFi": 91.9, "Bluetooth": 92.8, "ZigBee": 98.4, "Lightbridge": 99.5, "XPD": 99.5},
    {"模型": "YOLOv8m Teacher", "WiFi": 91.3, "Bluetooth": 93.5, "ZigBee": 98.4, "Lightbridge": 99.5, "XPD": 99.5},
    {"模型": "YOLOv8n 小样本", "WiFi": 89.4, "Bluetooth": 90.2, "ZigBee": 97.7, "Lightbridge": 99.5, "XPD": 99.4},
    {"模型": "YOLOv8n 蒸馏", "WiFi": 86.4, "Bluetooth": 90.9, "ZigBee": 98.2, "Lightbridge": 99.5, "XPD": 99.5},
]

DISTILLATION_GAIN_ROWS = [
    {"类别": "WiFi", "小样本 AP@50": "89.4", "蒸馏 AP@50": "86.4", "变化": "-3.4%"},
    {"类别": "Bluetooth", "小样本 AP@50": "90.2", "蒸馏 AP@50": "90.9", "变化": "+0.8%"},
    {"类别": "ZigBee", "小样本 AP@50": "97.7", "蒸馏 AP@50": "98.2", "变化": "+0.5%"},
    {"类别": "Lightbridge", "小样本 AP@50": "99.5", "蒸馏 AP@50": "99.5", "变化": "+0.0%"},
    {"类别": "XPD", "小样本 AP@50": "99.4", "蒸馏 AP@50": "99.5", "变化": "+0.0%"},
]


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.10), transparent 26rem),
                linear-gradient(180deg, #f8fafc 0%, #eef6f7 46%, #f8fafc 100%);
            color: #172033;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8e3e7;
        }
        .hero {
            border: 1px solid #d6e5e8;
            border-radius: 8px;
            padding: 18px 20px;
            background: linear-gradient(135deg, #ffffff 0%, #edf7f7 100%);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 30px;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: #475569;
            font-size: 15px;
        }
        .metric-card {
            border: 1px solid #d8e3e7;
            border-radius: 8px;
            background: #ffffff;
            padding: 13px 14px;
            min-height: 88px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
        }
        .metric-card .label {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 8px;
        }
        .metric-card .value {
            color: #0f172a;
            font-size: 23px;
            font-weight: 700;
            line-height: 1.1;
        }
        .metric-card .sub {
            color: #64748b;
            font-size: 12px;
            margin-top: 8px;
        }
        .signal-chip {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #e6f6f5;
            color: #0f766e;
            border: 1px solid #bce5df;
            font-size: 13px;
            margin: 0 7px 7px 0;
        }
        .insight {
            border-left: 4px solid #0f766e;
            background: #ffffff;
            border-radius: 8px;
            padding: 14px 16px;
            color: #334155;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #0f172a;
            margin: 12px 0 10px 0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d8e3e7;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def discover_models() -> dict[str, dict[str, str]]:
    available = {}
    for exp_name, meta in MODEL_CATALOG.items():
        weight_path = RUNS_DIR / exp_name / "weights" / "best.pt"
        if weight_path.exists():
            available[meta["label"]] = {**meta, "exp_name": exp_name, "path": str(weight_path)}
    return available


def discover_samples() -> dict[str, str]:
    samples: dict[str, str] = {}

    if DATASET_TEST_DIR.exists():
        for index, path in enumerate(sorted(DATASET_TEST_DIR.glob("*.jpg"))[:24], start=1):
            samples[f"推荐测试场景 {index:02d} - {path.stem}"] = str(path)

    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.glob("*.jpg")):
            label = path.stem.replace("class_", "裁剪样例 ").replace("_", " ")
            samples[f"{label}"] = str(path)

    return samples


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_image(uploaded_file, sample_path: str | None) -> tuple[Image.Image | None, str]:
    if uploaded_file is not None:
        return Image.open(uploaded_file).convert("RGB"), uploaded_file.name
    if sample_path:
        path = Path(sample_path)
        return Image.open(path).convert("RGB"), path.name
    return None, ""


def build_input_signature(
    model_label: str,
    image_name: str,
    sample_path: str | None,
    uploaded_file,
    conf_threshold: float,
) -> tuple:
    upload_id = None
    if uploaded_file is not None:
        upload_id = (uploaded_file.name, getattr(uploaded_file, "size", None))

    return (
        model_label,
        image_name,
        str(sample_path or ""),
        upload_id,
        round(float(conf_threshold), 3),
    )


def clear_stale_result(session_state, current_signature: tuple) -> None:
    previous_signature = session_state.get("demo_input_signature")
    if previous_signature != current_signature:
        session_state.pop("demo_result", None)
        session_state["demo_input_signature"] = current_signature


def run_detection(model_info: dict[str, str], image: Image.Image, conf_threshold: float) -> tuple[list[dict], float, np.ndarray]:
    import cv2

    model = load_model(model_info["path"])
    rgb_image = np.array(image)

    start = time.perf_counter()
    results = model.predict(source=rgb_image, conf=conf_threshold, verbose=False, imgsz=512)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for idx in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[idx].item())
            conf_val = float(result.boxes.conf[idx].item())
            x1, y1, x2, y2 = result.boxes.xyxy[idx].tolist()
            detections.append(
                {
                    "class": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "confidence": conf_val,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    annotated = draw_detections(rgb_image, detections)
    return detections, elapsed_ms, annotated


def draw_detections(image_np: np.ndarray, detections: list[dict]) -> np.ndarray:
    import cv2

    canvas = image_np.copy()
    overlay = canvas.copy()

    for det in detections:
        cls_name = det["class"]
        color = CLASS_COLORS.get(cls_name, (20, 184, 166))
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_y1 = max(0, y1 - label_h - 12)
        cv2.rectangle(canvas, (x1, label_y1), (x1 + label_w + 14, label_y1 + label_h + 12), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 7, label_y1 + label_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def detection_dataframe(detections: list[dict]) -> pd.DataFrame:
    rows = []
    ranked = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    for i, det in enumerate(ranked[:25], start=1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            {
                "序号": i,
                "类别": det["class"],
                "置信度": f"{det['confidence']:.3f}",
                "中心点": f"({(x1 + x2) // 2}, {(y1 + y2) // 2})",
                "框尺寸": f"{x2 - x1} x {y2 - y1}",
            }
        )
    frame = pd.DataFrame(rows)
    if len(ranked) > 25:
        frame.loc[len(frame)] = {
            "序号": "...",
            "类别": f"其余 {len(ranked) - 25} 个目标",
            "置信度": "已省略",
            "中心点": "详见检测框",
            "框尺寸": "详见检测框",
        }
    return frame


def render_signal_explanation(detections: list[dict], elapsed_ms: float) -> None:
    if not detections:
        st.warning("未检测到五类已知信号。可以适当降低置信度阈值，或把它解释为开集识别问题的引入。")
        st.markdown(
            """
            <div class="insight">
            若出现未知协议或低信噪比信号，闭集检测器可能无法给出可靠类别；
            后续可结合开集识别或异常检测模块提升未知信号处理能力。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    class_counts: dict[str, int] = {}
    for det in detections:
        class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

    chips = "".join([f'<span class="signal-chip">{name} x{count}</span>' for name, count in class_counts.items()])
    st.markdown(chips, unsafe_allow_html=True)

    for cls_name, count in class_counts.items():
        profile = SIGNAL_PROFILES[cls_name]
        with st.expander(f"{cls_name} 信号解释  x{count}", expanded=True):
            col_a, col_b = st.columns(2)
            col_a.markdown(f"**调制方式：** {profile['modulation']}")
            col_a.markdown(f"**典型带宽：** {profile['bandwidth']}")
            col_b.markdown(f"**应用场景：** {profile['application']}")
            col_b.markdown(f"**图像特征：** {profile['feature']}")

    detected_names = list(class_counts.keys())
    if "Lightbridge" in detected_names:
        conclusion = "检测到无人机图传相关信号，适合引出低空安全、无人机监测和频谱管控场景。"
    elif len(detected_names) >= 2:
        conclusion = "当前图中存在多协议共存，可用于说明 2.4GHz 频谱拥挤和干扰识别需求。"
    else:
        conclusion = f"当前主要检测到 {detected_names[0]}，可用于说明时频图目标检测能定位具体信号区域。"

    st.markdown(
        f"""
        <div class="insight">
        <b>频谱场景分析：</b>{conclusion}<br>
        <b>推理耗时：</b>{elapsed_ms:.1f} ms，本机实时演示时可能受首次加载和显卡状态影响。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_summary() -> None:
    st.markdown('<div class="section-title">整体性能对比</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(EXPERIMENT_ROWS), hide_index=True, width="stretch")
    st.markdown('<div class="section-title">逐类 AP@50 对比</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(PER_CLASS_AP_ROWS), hide_index=True, width="stretch")
    st.markdown('<div class="section-title">蒸馏前后逐类变化</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(DISTILLATION_GAIN_ROWS), hide_index=True, width="stretch")
    st.markdown(
        """
        <div class="insight">
        <b>结果观察：</b>在测试集上，蒸馏模型整体精度与小样本模型接近；
        从类别维度看，Bluetooth 和 ZigBee 略有提升，WiFi 下降更明显，
        表明伪标签质量和类别差异会影响知识迁移效果。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(models: dict[str, dict[str, str]], samples: dict[str, str]):
    st.sidebar.title("演示控制台")

    model_labels = list(models.keys())
    default_model = model_labels.index("YOLOv8n 蒸馏 Student") if "YOLOv8n 蒸馏 Student" in model_labels else 0
    selected_model = st.sidebar.selectbox("检测模型", model_labels, index=default_model)

    conf_threshold = st.sidebar.slider("置信度阈值", 0.10, 0.90, 0.25, 0.05)
    st.sidebar.caption("只显示置信度不低于该值的检测框；调高更严格、误检更少，调低更敏感、检测框更多。")

    sample_name = None
    if samples:
        sample_options = ["使用上传图片"] + list(samples.keys())
        default_sample = 1 if len(sample_options) > 1 else 0
        sample_name = st.sidebar.selectbox("内置样例", sample_options, index=default_sample)
    else:
        st.sidebar.warning("未找到内置样例，只能上传图片。")

    uploaded_file = st.sidebar.file_uploader("上传时频图", type=["jpg", "jpeg", "png"])
    run_button = st.sidebar.button("开始检测", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    selected_info = models[selected_model]
    st.sidebar.markdown("#### 当前模型")
    st.sidebar.write(selected_info["role"])
    st.sidebar.metric("测试集 mAP@50", selected_info["map50"])
    st.sidebar.metric("参数量", selected_info["params"])

    sample_path = None
    if sample_name and sample_name != "使用上传图片":
        sample_path = samples[sample_name]

    return selected_info, uploaded_file, sample_path, sample_name or "", conf_threshold, run_button


def main() -> None:
    st.set_page_config(page_title="低空宽带信号检测 Demo", page_icon="📡", layout="wide")
    apply_page_style()

    models = discover_models()
    samples = discover_samples()

    if not models:
        st.error(f"未找到模型权重。请确认权重位于：{RUNS_DIR}\\*/weights/best.pt")
        st.stop()

    selected_info, uploaded_file, sample_path, sample_name, conf_threshold, run_button = render_sidebar(models, samples)
    image, image_name = read_image(uploaded_file, sample_path)
    input_signature = build_input_signature(
        model_label=selected_info["label"],
        image_name=image_name,
        sample_path=sample_path,
        uploaded_file=uploaded_file,
        conf_threshold=conf_threshold,
    )
    clear_stale_result(st.session_state, input_signature)

    st.markdown(
        """
        <div class="hero">
          <h1>低空宽带信号智能检测 Demo</h1>
          <p>A2 时频图宽带信号检测 + B7 大小模型协同 + C2 小样本学习。面向无人机、物联网与 2.4GHz 频谱监测场景。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    top_cols = st.columns(5)
    model_size = Path(selected_info["path"]).stat().st_size / (1024 * 1024)
    with top_cols[0]:
        metric_card("当前模型", selected_info["label"].replace("YOLOv8n ", "").replace("YOLOv8m ", ""), selected_info["role"])
    with top_cols[1]:
        metric_card("mAP@50", selected_info["map50"], "测试集结果")
    with top_cols[2]:
        metric_card("mAP@50-95", selected_info["map"], "更关注定位质量")
    with top_cols[3]:
        metric_card("模型大小", f"{model_size:.1f} MB", f"{selected_info['params']} 参数")
    with top_cols[4]:
        metric_card("参考 FPS", selected_info["fps"], "训练后统计")

    if image is None:
        st.info("请在左侧选择内置样例或上传一张时频图，然后点击“开始检测”。")
        render_research_summary()
        return

    if run_button:
        try:
            with st.spinner("模型推理中，首次运行会加载权重..."):
                detections, elapsed_ms, annotated = run_detection(selected_info, image, conf_threshold)
            st.session_state["demo_result"] = {
                "input_signature": input_signature,
                "image_name": image_name,
                "model_label": selected_info["label"],
                "detections": detections,
                "elapsed_ms": elapsed_ms,
                "annotated": annotated,
                "original": np.array(image),
            }
        except Exception as exc:
            st.error(f"推理失败：{exc}")
            st.caption(f"模型路径：{selected_info['path']}")

    result = st.session_state.get("demo_result")
    if result is None:
        st.markdown('<div class="section-title">待检测时频图</div>', unsafe_allow_html=True)
        st.image(image, caption=image_name or sample_name, width="stretch")
        st.info("点击左侧“开始检测”后，这里会展示检测框、信号解释和实验对比。")
        render_research_summary()
        return

    st.markdown('<div class="section-title">检测画面</div>', unsafe_allow_html=True)
    img_col, result_col = st.columns([1.35, 1.0], vertical_alignment="top")

    with img_col:
        before, after = st.tabs(["原始时频图", "检测结果"])
        before.image(result["original"], caption=f"输入：{result['image_name']}", width="stretch")
        after.image(result["annotated"], caption=f"输出：{result['model_label']}", width="stretch")

    with result_col:
        small_cols = st.columns(3)
        with small_cols[0]:
            metric_card("检测目标", str(len(result["detections"])), "个候选信号")
        with small_cols[1]:
            type_count = len({det["class"] for det in result["detections"]})
            metric_card("信号类型", str(type_count), "已知类别")
        with small_cols[2]:
            metric_card("本次耗时", f"{result['elapsed_ms']:.1f} ms", "含前处理/推理")

        st.markdown('<div class="section-title">频谱分析</div>', unsafe_allow_html=True)
        render_signal_explanation(result["detections"], result["elapsed_ms"])

        if result["detections"]:
            st.markdown('<div class="section-title">检测明细</div>', unsafe_allow_html=True)
            st.dataframe(detection_dataframe(result["detections"]), hide_index=True, width="stretch")

    render_research_summary()


if __name__ == "__main__":
    main()
