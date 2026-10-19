"""
Tesla Autopilot Clone - AI-Powered Road Object Detection System
Interactive Streamlit Dashboard simulating autonomous driving perception.
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Automotive Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tesla Autopilot Clone | Road Vision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Automotive Dark CSS Styling
st.markdown("""
<style>
    /* Dark automotive theme background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top header styling */
    .autopilot-banner {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .autopilot-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .autopilot-badge {
        background-color: #e82127;
        color: white;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .autopilot-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00d2ff;
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d2ff;
        margin-bottom: 4px;
    }
    .metric-lbl {
        color: #8b949e;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(232, 33, 39, 0.08);
        border: 1px solid rgba(232, 33, 39, 0.3);
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 14px;
        color: #f85149;
        font-size: 0.85rem;
    }

    /* Custom sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 8px 18px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path Directories & Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMG_DIR = os.path.join(BASE_DIR, "data", "sample_images")
SAMPLE_VID_DIR = os.path.join(BASE_DIR, "data", "sample_videos")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "outputs", "annotated_images")
OUTPUT_VID_DIR = os.path.join(BASE_DIR, "outputs", "annotated_videos")

for p in [SAMPLE_IMG_DIR, SAMPLE_VID_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR]:
    os.makedirs(p, exist_ok=True)

# Target Autonomous Driving Classes from COCO
ROAD_CLASSES = {
    0: ("Person", "VRU"),
    1: ("Bicycle", "VRU"),
    2: ("Car", "Vehicle"),
    3: ("Motorcycle", "Vehicle"),
    5: ("Bus", "Vehicle"),
    7: ("Truck", "Vehicle"),
    9: ("Traffic Light", "Signal"),
    11: ("Stop Sign", "Signal"),
    12: ("Parking Meter", "Infrastructure"),
    10: ("Fire Hydrant", "Infrastructure")
}

# ---------------------------------------------------------
# Model Loading with Robust Fallback & Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO Neural Network Weights...")
def load_yolo_model(model_name: str = "yolov8n.pt"):
    """
    Load YOLO model safely with Ultralytics.
    Caches model in memory to prevent reload delays.
    """
    try:
        from ultralytics import YOLO
        # Search locally first, otherwise Ultralytics auto-downloads official weights
        model_path = os.path.join(BASE_DIR, model_name)
        if not os.path.exists(model_path):
            model_path = model_name
        model = YOLO(model_path)
        return model, "OK"
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# Core Perception Helper Functions
# ---------------------------------------------------------
def run_image_inference(model, image_input, conf_threshold: float = 0.35, road_filter: bool = False):
    """
    Execute object detection on a PIL Image or numpy array.
    Returns:
        annotated_image (RGB np.ndarray),
        results_df (pd.DataFrame),
        inference_time_ms (float)
    """
    start_time = time.time()
    
    # Run YOLO prediction
    results = model.predict(image_input, conf=conf_threshold, verbose=False)[0]
    inference_time_ms = (time.time() - start_time) * 1000.0

    boxes = results.boxes
    detections = []
    
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
        class_name = results.names[cls_id]

        category = "Other"
        if cls_id in ROAD_CLASSES:
            display_name, category = ROAD_CLASSES[cls_id]
        else:
            display_name = class_name.title()

        if road_filter and cls_id not in ROAD_CLASSES:
            continue

        detections.append({
            "ID": i + 1,
            "Class": display_name,
            "Category": category,
            "Confidence": round(conf * 100, 1),
            "X1": int(xyxy[0]),
            "Y1": int(xyxy[1]),
            "X2": int(xyxy[2]),
            "Y2": int(xyxy[3]),
            "BBox Area (px)": int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
        })

    # Render annotated image (BGR to RGB)
    annotated_bgr = results.plot(labels=True, conf=True)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    results_df = pd.DataFrame(detections)

    return annotated_rgb, results_df, inference_time_ms

def get_sample_images():
    """List available sample images in data directory."""
    if not os.path.exists(SAMPLE_IMG_DIR):
        return []
    valid_exts = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(SAMPLE_IMG_DIR) if f.lower().endswith(valid_exts)]

def get_sample_videos():
    """List available sample videos in data directory."""
    if not os.path.exists(SAMPLE_VID_DIR):
        return []
    valid_exts = ('.mp4', '.avi', '.mov')
    return [f for f in os.listdir(SAMPLE_VID_DIR) if f.lower().endswith(valid_exts)]

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ AUTOPILOT OS v12.4")
    st.caption("Neural Perception Subsystem")
    
    nav_option = st.radio(
        "Navigation Module",
        ["Dashboard Overview", "Image Perception", "Video Stream Perception", "Model Analytics", "Architecture & Docs"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Perception Settings")
    conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    road_only = st.checkbox("Focus Road Classes Only", value=True, help="Filter out non-traffic objects like indoor items")
    show_labels = st.checkbox("Display Bounding Labels", value=True)

    st.markdown("---")
    st.markdown("#### 🧠 Model Hardware")
    st.caption("**Model**: YOLOv8n (Lightweight Nano)")
    st.caption("**Input Resolution**: 640x640")
    st.caption("**Execution Mode**: CPU / Torch Inference")

# ---------------------------------------------------------
# Main Header Banner (Across All Pages)
# ---------------------------------------------------------
st.markdown("""
<div class="autopilot-banner">
    <div class="autopilot-title">
        TESLA AUTOPILOT CLONE <span class="autopilot-badge">SIMULATION</span>
    </div>
    <div class="autopilot-subtitle">
        AI-Powered Road Object Perception System • Pretrained Neural Vision Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# Check model availability
model, model_status = load_yolo_model("yolov8n.pt")
if model is None:
    st.error(f"⚠️ Model Initialization Notice: {model_status}. Ensure 'ultralytics' is installed and network connectivity allows loading weights.")

# =========================================================
# MODULE 1: DASHBOARD OVERVIEW
# =========================================================
if nav_option == "Dashboard Overview":
    st.markdown("### 📌 System Overview & Mission")
    st.markdown("""
    Welcome to the **Tesla Autopilot Clone Perception System**. This educational application reproduces the core 
    visual perception layer used in autonomous driving architectures. Using state-of-the-art computer vision 
    (YOLOv8 Nano), it processes real-time front-facing camera feeds to identify critical road objects:
    vehicles, vulnerable road users (pedestrians & cyclists), and traffic infrastructure.
    """)

    sample_imgs = get_sample_images()
    sample_vids = get_sample_videos()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(sample_imgs)}</div>
            <div class="metric-lbl">Sample Road Images</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(sample_vids)}</div>
            <div class="metric-lbl">Demo Video Feeds</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">80 / 10</div>
            <div class="metric-lbl">COCO / Road Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">~30-50 ms</div>
            <div class="metric-lbl">Avg Latency (CPU)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚦 Supported Road Perception Targets")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🚗 **Motor Vehicles**\n- Passenger Cars\n- Heavy Trucks\n- Public Buses\n- Motorcycles")
    with c2:
        st.warning("🚶 **Vulnerable Road Users**\n- Pedestrians / Crosswalk users\n- Bicycles & Cyclists\n- Stationary Obstacles")
    with c3:
        st.success("🚥 **Traffic Infrastructure**\n- Traffic Signals (Red/Yellow/Green)\n- Stop Signs\n- Hydrants & Curbs")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT EDUCATIONAL DISCLAIMER:</strong> This is an academic research simulation using a lightweight 
        publicly available YOLO neural network. It is <strong>NOT</strong> Tesla's proprietary Full Self-Driving (FSD) software 
        and is strictly intended for demonstration and academic study. Never use this for real-world automated vehicle control.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MODULE 2: IMAGE PERCEPTION
# =========================================================
elif nav_option == "Image Perception":
    st.markdown("### 📷 Front-Camera Road Scene Perception")
    
    tab_sample, tab_upload = st.tabs(["📁 Bundled Sample Images", "📤 Upload Custom Image"])
    selected_img = None

    with tab_sample:
        sample_files = get_sample_images()
        if sample_files:
            chosen_file = st.selectbox("Select Road Scene Sample", sample_files)
            sample_path = os.path.join(SAMPLE_IMG_DIR, chosen_file)
            if os.path.exists(sample_path):
                selected_img = Image.open(sample_path).convert("RGB")
        else:
            st.info("No sample images detected in data/sample_images. You can upload an image in the next tab or run prepare_data.py.")

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a road photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_img = Image.open(uploaded_file).convert("RGB")

    if selected_img is not None and model is not None:
        st.markdown("---")
        run_btn = st.button("🚀 Run Autopilot Vision Inference", use_container_width=True)

        if run_btn:
            with st.spinner("Processing frame through YOLOv8 Perception Engine..."):
                annotated_img, results_df, latency_ms = run_image_inference(
                    model=model,
                    image_input=selected_img,
                    conf_threshold=conf_thresh,
                    road_filter=road_only
                )

            # Metrics
            total_objs = len(results_df)
            unique_classes = results_df["Class"].nunique() if total_objs > 0 else 0
            v_count = (results_df["Category"] == "Vehicle").sum() if total_objs > 0 else 0
            p_count = (results_df["Category"] == "VRU").sum() if total_objs > 0 else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Objects Detected", total_objs)
            m2.metric("Unique Classes", unique_classes)
            m3.metric("Vehicles", v_count)
            m4.metric("Pedestrians / VRU", p_count)
            m5.metric("Inference Time", f"{latency_ms:.1f} ms")

            # Side-by-side view
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("📷 Raw Camera Input")
                st.image(selected_img, use_container_width=True)
            with col_right:
                st.subheader("🎯 Neural Detections Overlay")
                st.image(annotated_img, use_container_width=True)

            # Tabular & Chart Breakdown
            if total_objs > 0:
                st.markdown("### 📊 Detection Distribution & Objects Log")
                col_chart, col_table = st.columns([1, 1])

                with col_chart:
                    class_counts = results_df["Class"].value_counts().reset_index()
                    class_counts.columns = ["Class", "Count"]

                    fig = px.bar(
                        class_counts,
                        x="Class",
                        y="Count",
                        color="Class",
                        title="Detected Road Objects Breakdown",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig.update_layout(
                        plot_bgcolor="#161b22",
                        paper_bgcolor="#161b22",
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    st.markdown("**Detailed Object Attributes**")
                    st.dataframe(
                        results_df[["ID", "Class", "Category", "Confidence", "X1", "Y1", "X2", "Y2"]],
                        use_container_width=True,
                        height=280
                    )

                # Download Annotated Image
                out_filename = f"detected_{int(time.time())}.jpg"
                save_path = os.path.join(OUTPUT_IMG_DIR, out_filename)
                Image.fromarray(annotated_img).save(save_path)

                with open(save_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Annotated Vision Frame",
                        data=f,
                        file_name=out_filename,
                        mime="image/jpeg"
                    )
            else:
                st.info("No road objects detected above the selected confidence threshold. Try lowering the threshold slider.")

# =========================================================
# MODULE 3: VIDEO STREAM PERCEPTION
# =========================================================
elif nav_option == "Video Stream Perception":
    st.markdown("### 🎥 Continuous Video Stream Perception")
    st.markdown("""
    Autonomous driving systems process video frames in sequential sequence. This module executes frame-by-frame 
    perception, bounding box tracking, and latency profiling on dashcam footage.
    """)

    vid_tab1, vid_tab2 = st.tabs(["📁 Bundled Demonstration Video", "📤 Upload Custom Video"])
    video_source_path = None

    with vid_tab1:
        sample_vids = get_sample_videos()
        if sample_vids:
            selected_vid = st.selectbox("Choose Demo Video", sample_vids)
            video_source_path = os.path.join(SAMPLE_VID_DIR, selected_vid)
        else:
            st.info("No demo video found in data/sample_videos. Run prepare_data.py to generate one or upload an MP4.")

    with vid_tab2:
        uploaded_vid = st.file_uploader("Upload Driving Video (.mp4)", type=["mp4"])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_vid.read())
            video_source_path = tfile.name

    if video_source_path and os.path.exists(video_source_path) and model is not None:
        st.video(video_source_path)
        
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            frame_limit = st.slider("Maximum Frames to Process", min_value=10, max_value=120, value=40, step=10,
                                    help="Limit frame count for fast demonstration on standard CPU hardware.")
        with col_ctrl2:
            frame_skip = st.slider("Frame Stride (Skip)", min_value=1, max_value=5, value=2,
                                   help="Process every Nth frame to optimize throughput.")

        start_vid_btn = st.button("🎬 Process Video Stream with Autopilot Vision", use_container_width=True)

        if start_vid_btn:
            cap = cv2.VideoCapture(video_source_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.write(f"🎞️ Stream Metadata: {width}x{height} @ {fps:.1f} FPS | Total Stream Frames: {total_vid_frames}")

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()

            processed_count = 0
            frame_idx = 0
            out_frames = []
            latencies = []
            detected_classes_all = []

            while cap.isOpened() and processed_count < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    
                    t0 = time.time()
                    annotated_rgb, df, t_ms = run_image_inference(
                        model=model,
                        image_input=pil_frame,
                        conf_threshold=conf_thresh,
                        road_filter=road_only
                    )
                    latencies.append(t_ms)
                    if len(df) > 0:
                        detected_classes_all.extend(df["Class"].tolist())

                    out_frames.append(annotated_rgb)
                    processed_count += 1

                    # Live frame preview in UI
                    frame_display.image(annotated_rgb, caption=f"Frame {frame_idx} | Latency: {t_ms:.1f} ms", use_container_width=True)
                    progress = int((processed_count / frame_limit) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Frame {processed_count}/{frame_limit}...")

                frame_idx += 1

            cap.release()
            progress_bar.progress(100)
            status_text.success("🎉 Video perception sequence completed successfully!")

            # Summary stats
            avg_lat = np.mean(latencies) if latencies else 0
            st.markdown("### 📈 Video Processing Benchmark Summary")
            k1, k2, k3 = st.columns(3)
            k1.metric("Processed Frames", len(out_frames))
            k2.metric("Mean Latency", f"{avg_lat:.1f} ms")
            k3.metric("Effective FPS", f"{(1000.0 / avg_lat):.1f} FPS" if avg_lat > 0 else "N/A")

            # Save annotated video sample
            if len(out_frames) > 0:
                out_vid_name = f"annotated_stream_{int(time.time())}.mp4"
                out_vid_path = os.path.join(OUTPUT_VID_DIR, out_vid_name)
                
                # Write with OpenCV
                h, w, _ = out_frames[0].shape
                writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps/frame_skip), (w, h))
                for f in out_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()

                if os.path.exists(out_vid_path):
                    with open(out_vid_path, "rb") as vf:
                        st.download_button(
                            "📥 Download Annotated Video Clip",
                            data=vf,
                            file_name=out_vid_name,
                            mime="video/mp4"
                        )

# =========================================================
# MODULE 4: MODEL ANALYTICS & BENCHMARKS
# =========================================================
elif nav_option == "Model Analytics":
    st.markdown("### 📊 Model Architecture & Quantitative Performance")
    
    st.markdown("""
    #### 🔍 Pretrained Model Specification: YOLOv8 Nano (`yolov8n.pt`)
    - **Backbone**: Modified CSPDarknet53 with C2f modules.
    - **Neck**: Path Aggregation Network (PANet) feature pyramid.
    - **Head**: Anchor-free decoupled head for simultaneous classification and bounding box regression.
    - **Parameters**: ~3.2 Million weights (ultra-compact for edge/onboard automotive hardware).
    - **Training Dataset**: Microsoft COCO 2017 (80 categories, over 118,000 training images).
    """)

    st.markdown("---")
    st.markdown("### ⏱️ Latency Benchmark Profiler")
    st.caption("Measure actual inference latency on your current local CPU / hardware:")

    if st.button("⚡ Run 10-Iteration Latency Benchmark"):
        if model is not None:
            # Create synthetic test frame
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            benchmark_times = []
            
            with st.spinner("Benchmarking inference speed..."):
                # Warm-up pass
                _ = model.predict(dummy_frame, verbose=False)
                for _ in range(10):
                    t_start = time.perf_counter()
                    _ = model.predict(dummy_frame, verbose=False)
                    t_elapsed = (time.perf_counter() - t_start) * 1000.0
                    benchmark_times.append(t_elapsed)

            mean_lat = np.mean(benchmark_times)
            p95_lat = np.percentile(benchmark_times, 95)
            min_lat = np.min(benchmark_times)

            b1, b2, b3 = st.columns(3)
            b1.metric("Mean Latency", f"{mean_lat:.2f} ms")
            b2.metric("P95 Latency", f"{p95_lat:.2f} ms")
            b3.metric("Peak Inferences / Sec", f"{(1000.0/min_lat):.1f} FPS")

            # Chart of iteration latency
            df_bench = pd.DataFrame({
                "Iteration": list(range(1, 11)),
                "Latency (ms)": benchmark_times
            })
            fig_lat = px.line(
                df_bench, x="Iteration", y="Latency (ms)",
                markers=True, title="Inference Latency Across 10 Test Iterations",
                template="plotly_dark"
            )
            fig_lat.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#161b22")
            st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚠️ Perception Limitations & Edge Cases")
    st.markdown("""
    | Operational Factor | Challenge & Behavior | Mitigation Strategy |
    | :--- | :--- | :--- |
    | **Adverse Weather (Rain / Fog)** | Water droplets and light scattering attenuate high-frequency features, reducing detection recall. | Multi-modal sensor fusion (Radar / LiDAR) + temporal tracking. |
    | **Night / Low Illumination** | Low signal-to-noise ratio in camera sensors causes missed small objects. | HDR imaging sensors, infrared illumination, or contrast enhancement. |
    | **Severe Occlusion** | Partially blocked vehicles or pedestrians produce lower confidence scores. | Temporal tracking (Kalman filtering / DeepSORT) across successive frames. |
    | **Out-of-Distribution Obstacles** | Debris or exotic road objects outside COCO's 80 classes may not be detected. | Occupancy grid networks (3D voxel prediction) independent of semantics. |
    """)

# =========================================================
# MODULE 5: ARCHITECTURE & GUIDE
# =========================================================
elif nav_option == "Architecture & Docs":
    st.markdown("### 🏗️ Autonomous Driving Perception Architecture")
    
    st.markdown("""
    In a production autonomous driving stack (such as Tesla Vision), raw photon counts from camera CMOS sensors 
    pass through several layers before vehicle control decisions are made:
    """)

    st.code("""
    [Front/Side/Rear Cameras]
             │
             ▼
    [Raw Video Ingestion & Rectification (OpenCV)]
             │
             ▼
    [Neural Feature Backbone: CSPDarknet / RegNet]
             │
             ▼
    [Multi-Scale Feature Aggregator: PANet / BiFPN]
             │
             ▼
    [Decoupled Prediction Heads]
    ┌────────────────┬────────────────┐
    │ Classification │ Bounding Box   │
    │ (Car/Ped/Sign) │ Regression (xy)│
    └────────────────┴────────────────┘
             │
             ▼
    [Non-Maximum Suppression (NMS)]
             │
             ▼
    [Autopilot Planning & Trajectory Control]
    """, language="text")

    st.markdown("### 📚 Student Academic Review Guide")
    st.markdown("""
    Use these talking points during academic project reviews:
    1. **Why YOLOv8 Nano?** It balances high frame-rate edge performance (~30 FPS on consumer hardware) with competitive mean Average Precision (mAP), making it ideal for real-time safety critical robotics.
    2. **Confidence Threshold vs. Accuracy:** A confidence threshold controls precision-recall trade-offs. Setting it high reduces false alarms (ghost braking) but might miss faint distant objects.
    3. **Intersection-over-Union (IoU) & NMS:** Non-Maximum Suppression suppresses overlapping redundant candidate bounding boxes by calculating the overlap ratio between proposal boxes.
    """)

st.markdown("---")
st.caption("Tesla Autopilot Clone • Academic Simulation Project • Developed for Computer Vision & Autonomous Driving Demonstration")
