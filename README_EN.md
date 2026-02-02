# Real-Time Multi-Person Tracking System - Autonomous Quadcopter Navigation

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<p align="center">
  <img src="demo.gif" alt="System Demo" width="800"/>
</p>

## 📋 Project Overview

A **real-time human detection and persistent tracking** system on a simulated quadcopter platform in Gazebo Sim. Utilizing YOLOv8-based object detection, SORT (Simple Online and Realtime Tracking) algorithm, and color histogram-based re-identification to maintain continuous identity tracking in dynamic environments.

### 🎯 Key Features

- ✅ **Real-time Object Detection**: 30 FPS performance with YOLOv8n
- ✅ **Multi-Object Tracking**: Simultaneous tracking of multiple targets with SORT
- ✅ **Persistent ID Assignment**: Feature-based re-identification for continuous identity
- ✅ **Temporal Smoothing**: EMA-based bounding box stabilization
- ✅ **Interactive Selection**: Mouse-based target selection and manual tracking
- ✅ **Gazebo Integration**: Full integration with Harmonic (v8)

---

## 🏗️ System Architecture
```
┌──────────────────────────────────────────────────────────────────┐
│                      Gazebo Simulation Layer                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Quadcopter │  │   Actors    │  │    Environment           │  │
│  │  + Camera  │  │(Pedestrians)│  │ (Buildings + Textures)   │  │
│  └─────┬──────┘  └─────────────┘  └──────────────────────────┘  │
└────────┼─────────────────────────────────────────────────────────┘
         │ Topic: /quadcopter/camera/image (640x480 @ 30Hz)
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Perception & Tracking Pipeline                    │
│                                                                   │
│  ┌──────────┐      ┌──────────┐      ┌────────────────────┐    │
│  │  YOLOv8  │─────▶│   SORT   │─────▶│  Re-Identification │    │
│  │ Detector │      │ Tracker  │      │  (HSV Histogram)   │    │
│  └──────────┘      └──────────┘      └────────────────────┘    │
│                                                                   │
│  Input:  RGB Image (H×W×3)                                       │
│  Output: Tracks [(x₁,y₁,x₂,y₂,ID)ₙ]                             │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Persistent Person IDs
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Visualization & Control                        │
│                                                                   │
│  ┌─────────────┐          ┌──────────────────────────────┐      │
│  │   OpenCV    │          │    Control Publisher         │      │
│  │  Rendering  │          │ Topic: /model/quadcopter/... │      │
│  └─────────────┘          └──────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Algorithm Details

### 1. YOLOv8 Object Detection

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n (5.3M parameters) |
| Input Size | 640×480 |
| Confidence Threshold | 0.35 |
| Class Filter | Person (COCO class 0) |
| Inference Speed | ~30 FPS (CPU) |

### 2. SORT Tracking

**State Vector (7D Kalman Filter):**
```
x = [x_center, y_center, area, aspect_ratio, dx, dy, darea]
```

**Parameters:**
- **IoU Threshold**: 0.2 (flexible matching)
- **Max Age**: 30 frames (loss tolerance)
- **Min Hits**: 1 frame (fast initialization)

**Matching:** Hungarian Algorithm (O(n³))

### 3. Feature-based Re-Identification

**Feature Extraction:**
```python
HSV Color Space → Histogram (50×60 bins) → Normalization
Similarity Metric: cv2.compareHist(HISTCMP_CORREL)
```

**Re-ID Logic:**
```
IF similarity > 0.65 AND last_seen < 100 frames:
    THEN assign existing person_id
ELSE:
    create new person_id
```

### 4. Temporal Smoothing (EMA)
```python
bbox_smoothed = α × bbox_current + (1-α) × bbox_previous
α = 0.7  # smoothing factor
history_window = 5 frames
```

---

## 🚀 Installation

### System Requirements
```
OS:      Ubuntu 22.04 LTS (recommended)
Python:  3.10+
Gazebo:  Gazebo Sim 8 (Harmonic)
RAM:     8 GB minimum, 16 GB recommended
CPU:     4+ cores (Intel i5/i7 or equivalent)
GPU:     Optional (CUDA 11.8+ recommended)
```

# Install dependencies
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0

# Gazebo Python bindings
sudo apt-get install python3-gz-transport13 python3-gz-msgs10
```

### Step 3: Clone Project
```bash
git clone <repository-url>
cd real_time_perception_system
```

---

## 📂 Directory Structure
```
real_time_perception_system/
├── sim_worlds/
│   ├── city.world                    # Main simulation world
│   ├── materials/
│   ├── meshes/                       # 3D model files (optional)
│   │   └── walk.dae
│   └── scripts/
│       ├── camera_and_detector.py    # Main tracking system
│       ├── sort.py                   # SORT implementation
│       └── move_people.py            # Pedestrian movement script
├── requirements.txt
├── README_TR.md
├── README_EN.md
└── demo.gif
```

---

## 🎮 Usage

### Quick Start
```bash
# Terminal 1: Start Gazebo simulation
source rt_env/bin/activate
gz sim ~/real_time_perception_system/sim_worlds/city.world

# Terminal 2: Run tracking system
cd ~/real_time_perception_system/sim_worlds/scripts
python3 camera_and_detector.py
```

### Control Scheme

| Key/Action | Function | Description |
|------------|----------|-------------|
| **W** | Up | Quadcopter Z+ |
| **S** | Down | Quadcopter Z- |
| **A** | Left | Quadcopter Y+ |
| **D** | Right | Quadcopter Y- |
| **I** | Forward | Quadcopter X+ |
| **K** | Backward | Quadcopter X- |
| **T** | Auto-Track Toggle | Autonomous tracking ON/OFF |
| **Mouse Click** | Person Selection | Click on person → start tracking |
| **Q** | Quit | Close application |

### GUI Information

**Top-Right Panel:**
- **Detected**: Number of detected persons
- **Selected**: Selected person ID
- **Total IDs**: Total IDs created

**Bounding Box Colors:**
- **Green (thick)**: Tracked person
- **Colored (thin)**: Other detected persons
- Each ID has a consistent color

---

## 🔧 Configuration

### Adjusting SORT Parameters

**In `camera_and_detector.py`:**
```python
tracker = PersonTracker()
# Adjustable parameters:
# - max_age: Keep lost track (default: 30)
# - min_hits: Frames before ID assignment (default: 1)
# - iou_threshold: Matching sensitivity (default: 0.2)
```

### YOLO Confidence Adjustment
```python
results = model(
    img_bgr,
    conf=0.35,  # Adjustable (0.1-0.9)
    classes=[0],
    verbose=False
)
```

### Re-ID Threshold
```python
self.feature_similarity_threshold = 0.65  # Try 0.5-0.8
```

---

## 🐛 Troubleshooting

### 1. "No module named 'gz.transport13'"
```bash
sudo apt-get install python3-gz-transport13 python3-gz-msgs10
```

### 2. No camera feed
```bash
# Check topics
gz topic -l | grep camera

# Expected:
# /quadcopter/camera/image
# /quadcopter/camera_info
```

## 🚧 Ongoing Development

- [x] YOLOv8 integration
- [x] SORT tracking
- [x] Feature-based Re-ID
- [x] Manual quadcopter control
- [ ] **Autonomous tracking control** (in development)
- [ ] **Bbox rendering optimization**
- [ ] Multi-camera fusion
- [ ] Path prediction
- [ ] ROS 2 bridge

---

## 📚 References

### Academic Sources

1. **YOLO**: Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection"
2. **SORT**: Bewley, A., et al. (2016). "Simple Online and Realtime Tracking"
3. **DeepSORT**: Wojke, N., et al. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric"

### Libraries Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Gazebo Sim](https://gazebosim.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.
