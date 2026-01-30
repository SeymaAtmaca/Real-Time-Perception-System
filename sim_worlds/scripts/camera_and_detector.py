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

model = YOLO("yolov8n.pt")

# Global değişkenler
latest_frame = None
frame_lock = threading.Lock()

def camera_callback(msg: Image):
    """Kamera görüntüsünü işle"""
    global latest_frame
    
    width = msg.width
    height = msg.height
    
    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    
    # YOLO inference
    results = model(
        img_bgr,
        conf=0.4,
        classes=[0],  # person
        verbose=False
    )
    
    annotated = results[0].plot()
    
    with frame_lock:
        latest_frame = annotated

def display_thread():
    """Görüntüyü göster"""
    global latest_frame
    
    while True:
        with frame_lock:
            if latest_frame is not None:
                cv2.imshow("YOLO - Quadcopter Camera", latest_frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC tuşu
            break

def get_key():
    """Klavye tuşunu oku"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def control_thread(pub):
    """Quadcopter kontrolü"""
    speed = 2.0
    
    print("\n=== Quadcopter Kontrol ===")
    print("W/S: Yukarı/Aşağı")
    print("A/D: Sol/Sağ")
    print("I/K: İleri/Geri")
    print("Q: Çıkış")
    print("=" * 30)
    
    while True:
        key = get_key()
        
        msg = Twist()
        
        if key == 'w':
            msg.linear.z = speed
            print("↑ Yukarı    ", end='\r')
        elif key == 's':
            msg.linear.z = -speed
            print("↓ Aşağı     ", end='\r')
        elif key == 'a':
            msg.linear.y = speed
            print("← Sol       ", end='\r')
        elif key == 'd':
            msg.linear.y = -speed
            print("→ Sağ       ", end='\r')
        elif key == 'i':
            msg.linear.x = speed
            print("↑ İleri     ", end='\r')
        elif key == 'k':
            msg.linear.x = -speed
            print("↓ Geri      ", end='\r')
        elif key == 'q':
            print("\n\nÇıkış...")
            cv2.destroyAllWindows()
            sys.exit(0)
        else:
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = 0
        
        pub.publish(msg)

def main():
    # Gazebo Transport Node
    node = Node()
    
    # Kamera subscriber
    camera_topic = "/quadcopter/camera/image"
    if not node.subscribe(Image, camera_topic, camera_callback):
        print(f"✗ '{camera_topic}' topic'ine abone olunamadı!")
        return
    
    print(f"✓ Kamera topic'ine bağlandı: {camera_topic}")
    
    # Kontrol publisher
    pub = node.advertise("/model/quadcopter/cmd_vel", Twist)
    print(f"✓ Kontrol topic'i hazır: /model/quadcopter/cmd_vel")
    
    # Display thread'ini başlat
    display_t = threading.Thread(target=display_thread, daemon=True)
    display_t.start()
    
    # Kontrol thread'ini başlat (ana thread'de çalışsın)
    try:
        control_thread(pub)
    except KeyboardInterrupt:
        print("\n\nKapatılıyor...")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()