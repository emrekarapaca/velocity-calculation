import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import requests

url = 'http://192.168.2.222:8000/car_pass'

def avg_speed_calculation(speed_list):
    return sum(speed_list) / len(speed_list) if speed_list else 0

def velocity(video_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps

    vehicle_positions = {}
    vehicle_speeds = []  #

    pixel_to_meter_ratio = 0.05

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame)
        detections = results[0].boxes

        for det in detections:
            cls = int(det.cls[0])
            if cls != 2:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0].numpy())
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            vehicle_id = None
            for vid, pos in vehicle_positions.items():
                prev_x, prev_y = pos[-1]
                dist = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                if dist < 50:
                    vehicle_id = vid
                    break

            if vehicle_id is None:
                vehicle_id = len(vehicle_positions) + 1
                vehicle_positions[vehicle_id] = []

            vehicle_positions[vehicle_id].append((center_x, center_y))

            if len(vehicle_positions[vehicle_id]) > 1:
                prev_x, prev_y = vehicle_positions[vehicle_id][-2]
                dist_pixels = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                dist_meters = dist_pixels * pixel_to_meter_ratio
                speed = dist_meters / frame_time * 3.6
                vehicle_speeds.append((vehicle_id, speed))
            else:
                speed = 0

        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    avg_speed = avg_speed_calculation([speed for _, speed in vehicle_speeds])
    #cv2.putText(frame, f"Avg Speed: {avg_speed:.2f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #cv2.imshow("Vehicle Detection and Speed Estimation", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

    #cap.release()
    #cv2.destroyAllWindows()

    return avg_speed

def process_videos(video_directory):
    all_avg_speeds = {}
    video_counter = 1
    for video_file in os.listdir(video_directory):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_path = os.path.join(video_directory, video_file)
            print(f"Processing video: {video_path}")
            avg_speed = velocity(video_path)

            car_data = {'citypoint': video_counter, 'velocity': avg_speed}
            response = requests.post(url, json=car_data)
            print(response.text)

            all_avg_speeds[video_counter] = avg_speed
            video_counter += 1
    return all_avg_speeds

if __name__ == "__main__":
    video_directory = "/Users/emre/Desktop/detection-velocity/velocity/video"
    all_avg_speeds = process_videos(video_directory)
    print(all_avg_speeds)
