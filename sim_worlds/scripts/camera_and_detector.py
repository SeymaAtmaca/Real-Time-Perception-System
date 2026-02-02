#!/usr/bin/env python3

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import sys
import termios
import tty
from collections import defaultdict, deque

# SORT tracker import
from sort import Sort

model = YOLO("yolov8n.pt")

# Global değişkenler
latest_frame = None
frame_lock = threading.Lock()

# Gelişmiş tracking sistemi
class PersonTracker:
    def __init__(self):
        self.sort = Sort(max_age=30, min_hits=1, iou_threshold=0.2)
        self.person_features = {}
        self.sort_to_person_map = {}
        self.next_person_id = 1
        self.frame_count = 0
        self.feature_similarity_threshold = 0.65
        
        # Label stabilizasyonu için
        self.stable_labels = {}  # {person_id: {"pos": (x,y), "smoothing": deque}}
        
    def extract_features(self, image, bbox):
        """Kişiden renk histogramı çıkar"""
        x1, y1, x2, y2 = map(int, bbox)
        
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        person_roi = image[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def compare_features(self, hist1, hist2):
        """İki histogram arasındaki benzerlik"""
        if hist1 is None or hist2 is None:
            return 0.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def find_matching_person(self, feature):
        """Mevcut kişilerden en benzeri bul"""
        if feature is None:
            return None, 0
            
        best_match = None
        best_score = 0
        
        current_frame = self.frame_count
        
        for person_id, data in self.person_features.items():
            if current_frame - data["last_seen"] > 100:
                continue
                
            score = self.compare_features(feature, data["histogram"])
            
            if score > best_score and score > self.feature_similarity_threshold:
                best_score = score
                best_match = person_id
        
        return best_match, best_score
    
    def smooth_bbox(self, person_id, bbox, alpha=0.7):
        """Bbox pozisyonunu yumuşat (temporal smoothing)"""
        if person_id not in self.stable_labels:
            self.stable_labels[person_id] = {
                "bbox": np.array(bbox),  # NumPy array olarak kaydet
                "history": deque(maxlen=5)
            }
        
        self.stable_labels[person_id]["history"].append(bbox)
        
        # Ortalama bbox hesapla
        history = np.array(self.stable_labels[person_id]["history"])
        smoothed = np.mean(history, axis=0)
        
        # Exponential moving average
        old_bbox = np.array(self.stable_labels[person_id]["bbox"])  # NumPy array'e çevir
        new_bbox = alpha * smoothed + (1 - alpha) * old_bbox
        
        self.stable_labels[person_id]["bbox"] = new_bbox
        
        return new_bbox
    
    def update(self, image, detections):
        """Ana güncelleme fonksiyonu"""
        self.frame_count += 1
        
        sort_tracks = self.sort.update(detections)
        
        person_tracks = []
        
        for track in sort_tracks:
            x1, y1, x2, y2, sort_id = track
            sort_id = int(sort_id)
            
            # Feature çıkar
            feature = self.extract_features(image, [x1, y1, x2, y2])
            
            # SORT ID → Person ID mapping
            if sort_id in self.sort_to_person_map:
                person_id = self.sort_to_person_map[sort_id]
                
                # Person ID'nin feature dictionary'de olup olmadığını kontrol et
                if person_id in self.person_features and feature is not None:
                    self.person_features[person_id]["histogram"] = feature
                    self.person_features[person_id]["last_seen"] = self.frame_count
                elif feature is not None:
                    # Yoksa oluştur
                    self.person_features[person_id] = {
                        "histogram": feature,
                        "last_seen": self.frame_count
                    }
            else:
                matching_person, similarity = self.find_matching_person(feature)
                
                if matching_person:
                    person_id = matching_person
                    print(f"[REID] SORT ID {sort_id} → Person ID {person_id} (sim: {similarity:.2f})")
                else:
                    person_id = self.next_person_id
                    self.next_person_id += 1
                    print(f"[NEW] Person ID {person_id} created")
                
                self.sort_to_person_map[sort_id] = person_id
                
                if feature is not None:
                    self.person_features[person_id] = {
                        "histogram": feature,
                        "last_seen": self.frame_count
                    }
            
            # Bbox smoothing
            smoothed_bbox = self.smooth_bbox(person_id, [x1, y1, x2, y2])
            
            person_tracks.append([*smoothed_bbox, person_id])
        
        # Temizlik
        active_sort_ids = set([int(t[4]) for t in sort_tracks])
        self.sort_to_person_map = {
            k: v for k, v in self.sort_to_person_map.items() 
            if k in active_sort_ids
        }
        
        return np.array(person_tracks) if person_tracks else np.empty((0, 5))

# Global tracker
tracker = PersonTracker()

# Takip sistemi
selected_id = None
tracked_persons = {}
id_colors = {}

def get_color_for_id(track_id):
    """Her ID için tutarlı renk"""
    if track_id not in id_colors:
        np.random.seed(track_id * 123)
        id_colors[track_id] = tuple(np.random.randint(100, 255, 3).tolist())
    return id_colors[track_id]

def mouse_callback(event, x, y, flags, param):
    """Mouse tıklaması"""
    global selected_id, tracked_persons
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for track_id, data in tracked_persons.items():
            bbox = data["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                if selected_id == track_id:
                    selected_id = None
                    print(f"\n[INFO] Person #{track_id} takipten çıkarıldı")
                else:
                    selected_id = track_id
                    print(f"\n[INFO] Person #{track_id} takibe alındı!")
                break

def camera_callback(msg: Image):
    """Kamera görüntüsünü işle"""
    global latest_frame, tracker, tracked_persons, selected_id
    
    width = msg.width
    height = msg.height
    
    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    
    # YOLO
    results = model(
        img_bgr,
        conf=0.35,
        classes=[0],
        verbose=False
    )
    
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append([x1, y1, x2, y2, conf])
    
    detections = np.array(detections) if detections else np.empty((0, 5))
    
    # Tracker güncelle
    tracks = tracker.update(img_bgr, detections)
    
    current_tracked = {}
    annotated = img_bgr.copy()
    
    for track in tracks:
        x1, y1, x2, y2, person_id = track
        person_id = int(person_id)
        
        color = get_color_for_id(person_id)
        
        # Geçmiş
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if person_id in tracked_persons and "history" in tracked_persons[person_id]:
            history = tracked_persons[person_id]["history"]
        else:
            history = []
        
        history.append((cx, cy))
        if len(history) > 30:
            history = history[-30:]
        
        current_tracked[person_id] = {
            "bbox": [x1, y1, x2, y2],
            "color": color,
            "history": history
        }
        
        # Çizim
        if person_id == selected_id:
            thickness = 4
            box_color = (0, 255, 0)
            label = f"TRACKING #{person_id}"
            
            # Trail
            for i in range(1, len(history)):
                alpha = i / len(history)
                thick = int(2 + alpha * 2)
                pt1 = history[i-1]
                pt2 = history[i]
                cv2.line(annotated, pt1, pt2, (0, int(255*alpha), 0), thick)
        else:
            thickness = 2
            box_color = color
            label = f"Person #{person_id}"
        
        # Bbox - smoothed coordinates kullanılıyor
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), 
                     box_color, thickness)
        
        # Label - Sabit pozisyon (bbox tepesi)
        label_y = int(y1) - 8
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        # Label arka plan (semi-transparent)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (int(x1), label_y - label_h - 4), 
                     (int(x1) + label_w + 8, label_y + 4), box_color, -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Label text
        cv2.putText(annotated, label, (int(x1) + 4, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # Merkez nokta
        cv2.circle(annotated, (cx, cy), 4, box_color, -1)
        cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)
    
    tracked_persons = current_tracked
    
    # Info panel - Sağ üst köşe
    panel_x = width - 250
    panel_y = 20
    
    # Semi-transparent panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y - 5), 
                 (width - 10, panel_y + 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    cv2.putText(annotated, f"Detected: {len(tracks)}", 
               (panel_x, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, f"Selected: {selected_id if selected_id else 'None'}", 
               (panel_x, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if selected_id else (200, 200, 200), 1)
    cv2.putText(annotated, f"Total IDs: {tracker.next_person_id - 1}", 
               (panel_x, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(annotated, "Click to track", 
               (panel_x, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    with frame_lock:
        latest_frame = annotated

def display_thread():
    global latest_frame
    
    cv2.namedWindow("YOLO - Quadcopter Camera")
    cv2.setMouseCallback("YOLO - Quadcopter Camera", mouse_callback)
    
    while True:
        with frame_lock:
            if latest_frame is not None:
                cv2.imshow("YOLO - Quadcopter Camera", latest_frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def control_thread(pub):
    speed = 2.0
    auto_track_mode = False
    
    print("\n=== Quadcopter Kontrol ===")
    print("W/S: Yukarı/Aşağı")
    print("A/D: Sol/Sağ")
    print("I/K: İleri/Geri")
    print("T: Otomatik takip ON/OFF")
    print("Q: Çıkış")
    print("=" * 30)
    
    while True:
        key = get_key()
        
        msg = Twist()
        
        if key == 'w':
            msg.linear.z = speed
            auto_track_mode = False
            print("↑ Yukarı    ", end='\r')
        elif key == 's':
            msg.linear.z = -speed
            auto_track_mode = False
            print("↓ Aşağı     ", end='\r')
        elif key == 'a':
            msg.linear.y = speed
            auto_track_mode = False
            print("← Sol       ", end='\r')
        elif key == 'd':
            msg.linear.y = -speed
            auto_track_mode = False
            print("→ Sağ       ", end='\r')
        elif key == 'i':
            msg.linear.x = speed
            auto_track_mode = False
            print("↑ İleri     ", end='\r')
        elif key == 'k':
            msg.linear.x = -speed
            auto_track_mode = False
            print("↓ Geri      ", end='\r')
        elif key == 't':
            auto_track_mode = not auto_track_mode
            print(f"\n🎯 Auto-track: {'ON' if auto_track_mode else 'OFF'}")
        elif key == 'q':
            print("\n\nÇıkış...")
            cv2.destroyAllWindows()
            sys.exit(0)
        else:
            if not auto_track_mode:
                msg.linear.x = 0
                msg.linear.y = 0
                msg.linear.z = 0
        
        # Otomatik takip
        if auto_track_mode and selected_id and selected_id in tracked_persons:
            bbox = tracked_persons[selected_id]["bbox"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            offset_x = (cx - 320) / 320
            offset_y = (cy - 240) / 240
            
            msg.linear.y = -offset_x * speed * 0.4
            msg.linear.z = -offset_y * speed * 0.4
            
            print(f"🎯 Tracking Person #{selected_id}    ", end='\r')
        
        pub.publish(msg)

def main():
    node = Node()
    
    camera_topic = "/quadcopter/camera/image"
    if not node.subscribe(Image, camera_topic, camera_callback):
        print(f"✗ Bağlantı hatası!")
        return
    
    print(f"✓ Kamera: OK")
    
    pub = node.advertise("/model/quadcopter/cmd_vel", Twist)
    print(f"✓ Kontrol: OK")
    print("\n✅ 3 yürüyen insan eklendi!")
    print("✅ Stabil label sistemi aktif")
    
    display_t = threading.Thread(target=display_thread, daemon=True)
    display_t.start()
    
    try:
        control_thread(pub)
    except KeyboardInterrupt:
        print("\n\nKapatılıyor...")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()