#!/usr/bin/env python3

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

def camera_callback(msg: Image):
    width = msg.width
    height = msg.height

    img_data = np.frombuffer(msg.data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    # YOLO inference (sadece insan)
    results = model(
        img_bgr,
        conf=0.4,
        classes=[0],   # person
        verbose=False
    )

    annotated = results[0].plot()
    cv2.imshow("YOLO - Quadcopter Camera", annotated)
    cv2.waitKey(1)


def main():
    node = Node()
    
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