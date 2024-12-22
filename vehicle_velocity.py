import cv2
import numpy as np
from ultralytics import YOLO
import math

def avg_speed(speed_list):
    avg_speed = sum(speed_list)/len(speed_list)
    return avg_speed

def velocity():
    model = YOLO("yolov8n.pt")

    video_path = "/Users/emre/Desktop/detection-velocity/velocity/video/cctv052x2004080516x01638.avi"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS değeri
    frame_time = 1 / fps  # Her bir frame'in zamanı

    # Araç takibi için bir veri yapısı
    vehicle_positions = {}

    avg_speed_ = 0
    speed_ = []
    # Pikselden metreye dönüştürme oranı (kamera kalibrasyonuna göre ayarlanmalı)
    pixel_to_meter_ratio = 0.05  # Örneğin, her piksel 0.05 metreye denk

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO ile tespit yap
        results = model(frame)
        detections = results[0].boxes  # Tespit edilen nesneler

        for det in detections:
            cls = int(det.cls[0])  # Sınıf ID'si
            if cls != 2:  # '2' genelde araç (car) sınıfıdır (COCO dataset)
                continue

            # Bounding box koordinatları
            x1, y1, x2, y2 = map(int, det.xyxy[0].numpy())
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Araç ID'si (Takip için her araca bir ID atanabilir)
            vehicle_id = None
            for vid, pos in vehicle_positions.items():
                prev_x, prev_y = pos[-1]
                dist = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                if dist < 50:  # Eğer araç yakınsa aynı araç olarak kabul et
                    vehicle_id = vid
                    break

            if vehicle_id is None:  # Yeni bir araç tespit edilirse
                vehicle_id = len(vehicle_positions) + 1
                vehicle_positions[vehicle_id] = []

            # Pozisyonu kaydet
            vehicle_positions[vehicle_id].append((center_x, center_y))

            # Hız hesaplama
            if len(vehicle_positions[vehicle_id]) > 1:
                prev_x, prev_y = vehicle_positions[vehicle_id][-2]
                dist_pixels = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                dist_meters = dist_pixels * pixel_to_meter_ratio
                speed = dist_meters / frame_time * 3.6
                speed_.append(speed)

            else:
                speed = 0

            # Görüntüye çizin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

        avg_speed_ = avg_speed(speed_)
        print(avg_speed_)

        cv2.imshow("Vehicle Detection and Speed Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    velocity()
