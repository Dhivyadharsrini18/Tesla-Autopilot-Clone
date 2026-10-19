"""
Dataset Preparation Script for Tesla Autopilot Clone
Autonomous download, fallback generation, and sample video creation.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np

# Ensure UTF-8 output if possible
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
VIDEOS_DIR = os.path.join(DATA_DIR, "sample_videos")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ANNOTATED_IMG_DIR = os.path.join(OUTPUTS_DIR, "annotated_images")
ANNOTATED_VID_DIR = os.path.join(OUTPUTS_DIR, "annotated_videos")
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")

for d in [IMAGES_DIR, VIDEOS_DIR, ANNOTATED_IMG_DIR, ANNOTATED_VID_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Legitimate public domain and test assets
SAMPLE_IMAGE_URLS = {
    "urban_street.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "highway_traffic.jpg": "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=1080&q=80",
    "city_crosswalk.jpg": "https://images.unsplash.com/photo-1508974239320-0a029497e820?w=1080&q=80",
    "suburban_road.jpg": "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=1080&q=80"
}

def create_synthetic_road_scene(filename: str, style: str = "day") -> str:
    """Generate a clean procedural road scene if internet download is unavailable."""
    width, height = 960, 540
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky
    if style == "day":
        img[:220, :] = [235, 206, 135]  # Light sky blue
    else:
        img[:220, :] = [40, 25, 20]     # Night sky

    # Ground / Horizon
    img[220:250, :] = [80, 110, 80]     # Distant trees/hills

    # Road surface (trapezoid perspective)
    road_pts = np.array([[380, 240], [580, 240], [920, 540], [40, 540]], np.int32)
    cv2.fillPoly(img, [road_pts], (50, 50, 50))

    # Lane markings (dashed center lines)
    for y in range(260, 540, 45):
        scale = (y - 240) / 300.0
        w_dash = int(4 + scale * 8)
        h_dash = int(12 + scale * 18)
        cx = int(480 + (scale * 0))
        cv2.rectangle(img, (cx - w_dash // 2, y), (cx + w_dash // 2, y + h_dash), (255, 255, 255), -1)

    # Left solid line, Right solid line
    cv2.line(img, (400, 240), (70, 540), (255, 255, 255), 3)
    cv2.line(img, (560, 240), (890, 540), (0, 215, 255), 3)

    # Lead Vehicle
    lead_x, lead_y, lead_w, lead_h = 520, 310, 90, 65
    cv2.rectangle(img, (lead_x, lead_y), (lead_x + lead_w, lead_y + lead_h), (30, 30, 180), -1)
    cv2.rectangle(img, (lead_x + 10, lead_y - 25), (lead_x + lead_w - 10, lead_y), (60, 60, 60), -1)
    cv2.circle(img, (lead_x + 15, lead_y + 40), 8, (0, 0, 220), -1)
    cv2.circle(img, (lead_x + lead_w - 15, lead_y + 40), 8, (0, 0, 220), -1)

    # Left Lane Vehicle
    car2_x, car2_y, car2_w, car2_h = 240, 350, 130, 85
    cv2.rectangle(img, (car2_x, car2_y), (car2_x + car2_w, car2_y + car2_h), (180, 150, 40), -1)
    cv2.rectangle(img, (car2_x + 15, car2_y - 35), (car2_x + car2_w - 15, car2_y), (70, 70, 70), -1)
    cv2.circle(img, (car2_x + 20, car2_y + 55), 10, (0, 0, 240), -1)
    cv2.circle(img, (car2_x + car2_w - 20, car2_y + 55), 10, (0, 0, 240), -1)

    # Traffic Light
    cv2.rectangle(img, (870, 180), (880, 320), (70, 70, 70), -1)
    cv2.rectangle(img, (860, 140), (890, 200), (30, 30, 30), -1)
    cv2.circle(img, (875, 152), 6, (0, 0, 255), -1)
    cv2.circle(img, (875, 170), 6, (0, 255, 255), -1)
    cv2.circle(img, (875, 188), 6, (0, 255, 0), -1)

    # Autopilot Vision Label Overlay
    cv2.putText(img, "SYNTHETIC ROAD SCENE DEMONSTRATION", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    target_path = os.path.join(IMAGES_DIR, filename)
    cv2.imwrite(target_path, img)
    return target_path

def download_or_generate_images():
    """Attempt downloading real road images; fall back gracefully to procedural samples."""
    print("[*] Preparing sample road images...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for name, url in SAMPLE_IMAGE_URLS.items():
        target = os.path.join(IMAGES_DIR, name)
        if os.path.exists(target) and os.path.getsize(target) > 5000:
            print(f"  [OK] Existing image found: {name}")
            continue

        success = False
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 5000:
                    with open(target, "wb") as f:
                        f.write(data)
                    test_mat = cv2.imread(target)
                    if test_mat is not None and test_mat.size > 0:
                        print(f"  [OK] Downloaded: {name} ({len(data)//1024} KB)")
                        success = True
                    else:
                        if os.path.exists(target):
                            os.remove(target)
        except Exception as e:
            print(f"  [!] Download note for {name}: {e}")

        if not success:
            print(f"  [*] Generating procedural synthetic road scene for: {name}")
            create_synthetic_road_scene(name, style="night" if "night" in name else "day")
            print(f"  [OK] Synthetic scene created: {name}")

def generate_sample_video():
    """Generate a 4-second demonstration video with moving vehicle perspective."""
    video_path = os.path.join(VIDEOS_DIR, "road_sample_demo.mp4")
    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
        print(f"  [OK] Existing demo video found: {video_path}")
        return video_path

    print("[*] Generating demonstration road video clip (road_sample_demo.mp4)...")
    fps = 20
    duration_sec = 4
    total_frames = fps * duration_sec
    w, h = 640, 360

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky
        frame[:150, :] = [210, 180, 120]
        # Distant terrain
        frame[150:180, :] = [60, 90, 60]

        # Road
        road_pts = np.array([[260, 170], [380, 170], [610, 360], [30, 360]], np.int32)
        cv2.fillPoly(frame, [road_pts], (45, 45, 45))

        # Lane markings that move with time
        offset = (i * 8) % 40
        for y_base in range(180, 360, 40):
            y = y_base + offset
            if y < 355:
                scale = (y - 170) / 190.0
                dash_w = max(2, int(scale * 6))
                dash_h = max(4, int(scale * 14))
                cx = int(320 + scale * 0)
                cv2.rectangle(frame, (cx - dash_w, y), (cx + dash_w, y + dash_h), (255, 255, 255), -1)

        # Curbs
        cv2.line(frame, (260, 170), (30, 360), (255, 255, 255), 2)
        cv2.line(frame, (380, 170), (610, 360), (0, 200, 255), 2)

        # Vehicle ahead smoothly moving forward/wobbling slightly
        lead_y = int(220 + np.sin(i * 0.1) * 8)
        lead_x = int(340 + np.cos(i * 0.05) * 15)
        lead_scale = (lead_y - 170) / 190.0
        car_w = int(45 * lead_scale)
        car_h = int(30 * lead_scale)

        cv2.rectangle(frame, (lead_x, lead_y), (lead_x + car_w, lead_y + car_h), (20, 20, 170), -1)
        cv2.rectangle(frame, (lead_x + 5, lead_y - int(car_h*0.4)), (lead_x + car_w - 5, lead_y), (40, 40, 40), -1)
        cv2.circle(frame, (lead_x + 6, lead_y + car_h - 6), max(2, int(3*lead_scale)), (0, 0, 255), -1)
        cv2.circle(frame, (lead_x + car_w - 6, lead_y + car_h - 6), max(2, int(3*lead_scale)), (0, 0, 255), -1)

        cv2.putText(frame, "AUTOPILOT VISION SIMULATOR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"SPEED: {int(55 + np.sin(i*0.2)*3)} MPH | FRAME: {i+1}/{total_frames}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"  [OK] Demo video saved: {video_path}")
    return video_path

if __name__ == "__main__":
    download_or_generate_images()
    generate_sample_video()
    print("[OK] Dataset preparation completed.")
