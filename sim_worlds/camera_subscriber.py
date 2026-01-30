#!/usr/bin/env python3

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import cv2

def camera_callback(msg: Image):
    """Kamera görüntüsünü işle"""
    # Görüntü boyutları
    width = msg.width
    height = msg.height
    
    print(f"Görüntü alındı: {width}x{height}, format: {msg.pixel_format_type}")
    
    # RGB verisini numpy array'e çevir
    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    
    # BGR'ye çevir (OpenCV için)
    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    
    # Göster
    cv2.imshow('Quadcopter Camera', img_bgr)
    cv2.waitKey(1)

def main():
    # Node oluştur
    node = Node()
    
    # Topic'e abone ol
    topic = "/quadcopter/camera/image"
    
    print(f"'{topic}' topic'ine abone oluyor...")
    
    if node.subscribe(Image, topic, camera_callback):
        print(f"✓ '{topic}' topic'ine başarıyla abone olundu")
        print("Görüntüler gelmeye başlayacak...")
        print("Çıkmak için Ctrl+C basın")
    else:
        print(f"✗ '{topic}' topic'ine abone olunamadı!")
        print("\nMevcut topic'leri kontrol edin:")
        print("  gz topic -l | grep camera")
        return
    
    try:
        # Sonsuz döngü
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nKapatılıyor...")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()