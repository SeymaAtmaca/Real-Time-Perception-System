# Gerçek Zamanlı Çok Kişi Takip Sistemi - Otonom Quadcopter Navigasyonu

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<p align="center">
  <img src="demo.gif" alt="System Demo" width="800"/>
</p>

## 📋 Proje Özeti

Gazebo Sim ortamında simüle edilen quadcopter platformu üzerinde **gerçek zamanlı insan tespiti ve kalıcı takip** sistemi. YOLOv8 tabanlı nesne tespiti, SORT (Simple Online and Realtime Tracking) algoritması ve renk histogram tabanlı Re-identification kullanılarak dinamik ortamlarda kesintisiz kimlik koruması sağlanmaktadır.

### 🎯 Temel Özellikler

- ✅ **Real-time Object Detection**: YOLOv8n ile 30 FPS performans
- ✅ **Multi-Object Tracking**: SORT algoritması ile eş zamanlı birden fazla hedef takibi
- ✅ **Persistent ID Assignment**: Feature-based re-identification ile kalıcı kimlik
- ✅ **Temporal Smoothing**: EMA tabanlı bounding box stabilizasyonu
- ✅ **Interactive Selection**: Mouse ile hedef seçimi ve manuel takip başlatma
- ✅ **Gazebo Integration**: Harmonic (v8) tam entegrasyonu

---

## 🏗️ Sistem Mimarisi
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

## 🧮 Algoritma Detayları

### 1. YOLOv8 Object Detection

| Parametre | Değer |
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

**Parametreler:**
- **IoU Threshold**: 0.2 (esnek eşleştirme)
- **Max Age**: 30 frames (kayıp toleransı)
- **Min Hits**: 1 frame (hızlı başlatma)

**Eşleştirme:** Hungarian Algorithm (O(n³))

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

## 🚀 Kurulum

### Sistem Gereksinimleri
```
OS:      Ubuntu 22.04 LTS (önerilen)
Python:  3.10+
Gazebo:  Gazebo Sim 8 (Harmonic)
RAM:     8 GB minimum, 16 GB önerilen
CPU:     4+ cores (Intel i5/i7 veya eşdeğeri)
GPU:     Opsiyonel (CUDA 11.8+ önerilir)
```

# Gazebo Python bağlantıları
sudo apt-get install python3-gz-transport13 python3-gz-msgs10
```

### Adım 3: Proje Klonlama
```bash
git clone <repository-url>
cd real_time_perception_system
```

---

## 📂 Dizin Yapısı
```
real_time_perception_system/
├── sim_worlds/
│   ├── city.world                    # Ana simülasyon dünyası
│   ├── materials/
│   ├── meshes/                       # 3D model dosyaları (opsiyonel)
│   │   └── walk.dae
│   └── scripts/
│       ├── camera_and_detector.py    # Ana tracking sistemi
│       ├── sort.py                   # SORT implementasyonu
│       └── move_people.py            # Pedestrian hareket scripti
├── requirements.txt
├── README_TR.md
├── README_EN.md
└── demo.gif
```

---

## 🎮 Kullanım

### Hızlı Başlangıç
```bash
# Terminal 1: Gazebo Simülasyonunu başlat
source rt_env/bin/activate
gz sim ~/real_time_perception_system/sim_worlds/city.world

# Terminal 2: Tracking sistemini çalıştır
cd ~/real_time_perception_system/sim_worlds/scripts
python3 camera_and_detector.py
```

### Kontrol Şeması

| Tuş/Aksiyon | Fonksiyon | Açıklama |
|-------------|-----------|----------|
| **W** | Yukarı | Quadcopter Z+ |
| **S** | Aşağı | Quadcopter Z- |
| **A** | Sol | Quadcopter Y+ |
| **D** | Sağ | Quadcopter Y- |
| **I** | İleri | Quadcopter X+ |
| **K** | Geri | Quadcopter X- |
| **T** | Auto-Track Toggle | Otomatik takip ON/OFF |
| **Mouse Click** | Person Selection | Kişiye tıkla → takibe al |
| **Q** | Quit | Uygulamayı kapat |

### GUI Bilgileri

**Sağ Üst Panel:**
- **Detected**: Algılanan kişi sayısı
- **Selected**: Seçili kişi ID'si
- **Total IDs**: Toplam oluşturulan ID sayısı

**Bounding Box Renkleri:**
- **Yeşil (kalın)**: Takip edilen kişi
- **Renkli (ince)**: Diğer algılanan kişiler
- Her ID için sabit renk atanır

---


## 🔧 Yapılandırma

### SORT Parametrelerini Ayarlama

**`camera_and_detector.py` içinde:**
```python
tracker = PersonTracker()
# Değiştirebileceğiniz parametreler:
# - max_age: Kayıp track'i sakla (default: 30)
# - min_hits: ID atamadan önce kaç frame (default: 1)
# - iou_threshold: Eşleştirme hassasiyeti (default: 0.2)
```

### YOLO Confidence Ayarlama
```python
results = model(
    img_bgr,
    conf=0.35,  # Burası değiştirilebilir (0.1-0.9)
    classes=[0],
    verbose=False
)
```

### Re-ID Threshold
```python
self.feature_similarity_threshold = 0.65  # 0.5-0.8 arası deneyin
```

---

## 🐛 Sorun Giderme

### 1. "No module named 'gz.transport13'"
```bash
sudo apt-get install python3-gz-transport13 python3-gz-msgs10
```

### 2. Kamera görüntüsü gelmiyor
```bash
# Topic kontrolü
gz topic -l | grep camera

# Beklenen:
# /quadcopter/camera/image
# /quadcopter/camera_info
```

## 🚧 Devam Eden Geliştirmeler

- [x] YOLOv8 entegrasyonu
- [x] SORT tracking
- [x] Feature-based Re-ID
- [x] Manual quadcopter control
- [ ] **Autonomous tracking control** (geliştirme aşamasında)
- [ ] **Bbox rendering optimization**
- [ ] Multi-camera fusion
- [ ] Path prediction
- [ ] ROS 2 bridge

---

## 📚 Referanslar

### Akademik Kaynaklar

1. **YOLO**: Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection"
2. **SORT**: Bewley, A., et al. (2016). "Simple Online and Realtime Tracking"
3. **DeepSORT**: Wojke, N., et al. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric"

### Kullanılan Kütüphaneler

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Gazebo Sim](https://gazebosim.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)


---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.


