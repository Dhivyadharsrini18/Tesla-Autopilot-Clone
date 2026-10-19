"""
Automated Test Suite for Tesla Autopilot Clone Object Detection System
Verifies dataset integrity, model loading, perception inference, data aggregation,
output generation, and web dashboard syntax.
"""

import os
import sys
import unittest
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestAutopilotSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PROJECT_ROOT, "data")
        cls.img_dir = os.path.join(cls.data_dir, "sample_images")
        cls.vid_dir = os.path.join(cls.data_dir, "sample_videos")
        cls.output_dir = os.path.join(PROJECT_ROOT, "outputs")
        cls.model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")

    def test_01_dataset_paths_and_assets(self):
        """Test dataset path structure and verify asset presence."""
        self.assertTrue(os.path.exists(self.img_dir), "Sample images directory must exist.")
        self.assertTrue(os.path.exists(self.vid_dir), "Sample videos directory must exist.")
        
        images = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.assertGreater(len(images), 0, "There should be at least one sample road scene image.")
        
        videos = [f for f in os.listdir(self.vid_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        self.assertGreater(len(videos), 0, "There should be at least one demonstration video.")

    def test_02_model_initialization(self):
        """Test model weights initialization with Ultralytics."""
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        self.assertIsNotNone(model, "YOLO model should initialize successfully.")
        self.assertEqual(model.task, "detect", "Model task must be 'detect'.")
        self.assertGreater(len(model.names), 0, "Model should expose COCO class names.")

    def test_03_image_validation(self):
        """Test that images can be decoded and have positive 3-channel dimensions."""
        images = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        first_img_path = os.path.join(self.img_dir, images[0])
        
        # Test OpenCV reading
        cv_mat = cv2.imread(first_img_path)
        self.assertIsNotNone(cv_mat, f"OpenCV should successfully decode {first_img_path}")
        self.assertEqual(len(cv_mat.shape), 3, "Image must have 3 dimensions (H, W, C)")
        self.assertEqual(cv_mat.shape[2], 3, "Image must have 3 color channels")

        # Test PIL reading
        pil_img = Image.open(first_img_path)
        self.assertGreater(pil_img.width, 100, "Image width should be greater than 100px")
        self.assertGreater(pil_img.height, 100, "Image height should be greater than 100px")

    def test_04_detection_inference_and_results_processing(self):
        """Test model forward pass, bounding box extraction, and dataframe construction."""
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        
        images = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_path = os.path.join(self.img_dir, images[0])
        
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        self.assertIsNotNone(results.boxes, "Inference output must contain bounding boxes tensor")
        
        # Test converting to structured records
        records = []
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            records.append({
                "Class": results.names[cls_id],
                "Confidence": conf,
                "BBox": xyxy
            })
            
        df = pd.DataFrame(records)
        self.assertIsInstance(df, pd.DataFrame, "Results should form a valid Pandas DataFrame.")
        if not df.empty:
            self.assertIn("Class", df.columns)
            self.assertIn("Confidence", df.columns)
            self.assertTrue((df["Confidence"] >= 0.25).all(), "All detections should exceed threshold")

    def test_05_object_count_aggregation(self):
        """Test aggregating detection counts by class category."""
        dummy_df = pd.DataFrame({
            "Class": ["Car", "Car", "Truck", "Person", "Traffic Light"],
            "Category": ["Vehicle", "Vehicle", "Vehicle", "VRU", "Signal"],
            "Confidence": [89.5, 76.2, 85.0, 68.3, 91.0]
        })
        
        counts = dummy_df["Class"].value_counts()
        self.assertEqual(counts["Car"], 2)
        self.assertEqual(counts["Truck"], 1)
        self.assertEqual(counts["Person"], 1)
        
        v_count = (dummy_df["Category"] == "Vehicle").sum()
        self.assertEqual(v_count, 3, "Vehicle count should be 3")

    def test_06_output_generation_validation(self):
        """Verify that outputs directory contains valid annotated image, chart, and table."""
        annotated_img = os.path.join(self.output_dir, "annotated_images", "sample_detection_result.jpg")
        self.assertTrue(os.path.exists(annotated_img), "sample_detection_result.jpg should exist")
        self.assertGreater(os.path.getsize(annotated_img), 5000, "Annotated image should not be empty")

        chart_path = os.path.join(self.output_dir, "charts", "class_counts_chart.png")
        self.assertTrue(os.path.exists(chart_path), "class_counts_chart.png should exist")
        self.assertGreater(os.path.getsize(chart_path), 5000, "Chart image should not be empty")

        csv_path = os.path.join(self.output_dir, "detection_summary.csv")
        self.assertTrue(os.path.exists(csv_path), "detection_summary.csv should exist")
        self.assertGreater(os.path.getsize(csv_path), 50, "Summary CSV should have content")

    def test_07_app_file_syntax_and_structure(self):
        """Test app.py compiles cleanly without Python syntax errors."""
        app_file = os.path.join(PROJECT_ROOT, "app.py")
        self.assertTrue(os.path.exists(app_file), "app.py must exist.")
        
        with open(app_file, "r", encoding="utf-8") as f:
            code_content = f.read()
        
        # Compile syntax check
        compiled = compile(code_content, app_file, 'exec')
        self.assertIsNotNone(compiled, "app.py should compile without any syntax errors.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
