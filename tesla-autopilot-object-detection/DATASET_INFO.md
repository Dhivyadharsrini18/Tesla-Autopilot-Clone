# Dataset Documentation & Licensing Information

## Project: Tesla Autopilot Clone — Road Object Detection System

### Overview
This project simulates the visual perception layer of an autonomous driving vehicle. In autonomous driving systems (such as Tesla Vision), onboard cameras feed video frames to neural network object detectors that classify and localize road entities (vehicles, pedestrians, cyclists, road signs, traffic signals).

For educational and demonstration purposes, this repository utilizes lightweight road-scene imagery and short demonstration video clips, annotated using a pretrained COCO-compatible YOLOv8n (nano) model.

---

### Dataset Assets

| Filename | Description | Source / Reference | License / Usage |
| :--- | :--- | :--- | :--- |
| `urban_street.jpg` | Urban street scene with transit bus, pedestrians, and street signs | [Ultralytics Test Assets](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg) | AGPL-3.0 / Educational Public Test Image |
| `highway_traffic.jpg` | Multi-lane highway with passenger vehicles and commercial trucks | Public Domain / Wikimedia Commons Road Transport Archive | CC0 / Public Domain |
| `city_crosswalk.jpg` | Urban intersection with pedestrians, cyclists, and traffic signals | Wikimedia Commons Urban Street Archive | CC-BY-SA / Educational Fair Use |
| `road_sample_demo.mp4` | 5-second simulated driving sequence rendered from sequential perspective frames | Synthesized autonomously via OpenCV VideoWriter for testing pipeline | Project Demonstration Asset |

---

### Target Detection Classes (COCO Road Subset)
The pretrained YOLO model is trained on the Microsoft COCO dataset (80 categories). The relevant autonomous driving classes highlighted in this project are:

1. **Vehicles**: `car` (Class 2), `motorcycle` (Class 3), `bus` (Class 5), `truck` (Class 7)
2. **Vulnerable Road Users (VRUs)**: `person` (Class 0), `bicycle` (Class 1)
3. **Traffic Infrastructure**: `traffic light` (Class 9), `stop sign` (Class 11), `fire hydrant` (Class 10)

---

### Synthetic Fallback Mechanism
If public image downloads are restricted or offline:
- An autonomous fallback generator (`prepare_dataset.py`) creates synthetic road scenes using OpenCV procedural graphics.
- These synthetic images clearly represent lane markers, vehicle silhouettes, and signal lights to allow full pipeline testing (Streamlit UI, confidence thresholding, metrics aggregation, export tools) without network dependencies.

---

### Ethical & Academic Disclaimer
- **Not for real-world driving:** These assets and models are strictly for academic demonstration and simulation.
- **No benchmark fabrication:** Quantitative evaluations provided reflect actual inference benchmarks on these demonstration samples rather than fabricated accuracy metrics.
