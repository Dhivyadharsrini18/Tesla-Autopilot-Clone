"""
Generate a sample annotated video output using the pretrained YOLO model.
"""

import os
import cv2
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "yolov8n.pt")
input_video = os.path.join(base_dir, "data", "sample_videos", "road_sample_demo.mp4")
output_video = os.path.join(base_dir, "outputs", "annotated_videos", "sample_annotated_video.mp4")

if not os.path.exists(input_video):
    print("Input video not found!")
    exit(1)

model = YOLO(model_path)
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_count = 0
while cap.isOpened() and frame_count < 40:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.predict(frame, conf=0.35, verbose=False)[0]
    annotated = res.plot()
    writer.write(annotated)
    frame_count += 1

cap.release()
writer.release()
print(f"[OK] Generated sample annotated video: {output_video} ({frame_count} frames)")
