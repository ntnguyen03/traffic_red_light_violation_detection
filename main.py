import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from collections import defaultdict
import os

# Tạo thư mục lưu vi phạm
VIOLATION_DIR = 'violations'
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# Load các mô hình YOLOv8
vehicle_model = YOLO('yolov8m.pt')
plate_model = YOLO('models/license_plate/license_plate_detection.pt')
ocr_model = YOLO('models/license_plate/license_plate_ocr.pt')
traffic_light_model = YOLO('models/traffic_light/traffic_light.pt')

# Đọc video từ camera giao thông
cap = cv2.VideoCapture('data_test/test2.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize khung hình để tăng tốc
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = int(RESIZE_WIDTH * frame_height / frame_width)
scale_factor = frame_width / RESIZE_WIDTH

# Định nghĩa vùng vạch dừng (ROI)
STOP_LINE_Y = 450

# Lưu vị trí tâm của phương tiện qua các frame để kiểm tra hướng di chuyển
vehicle_positions = defaultdict(list)

# Tắt hiển thị khung hình để tăng tốc (True: hiển thị, False: không hiển thị)
DISPLAY_FRAME = True

# Hàm tính tâm của bounding box
def get_center(box):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

# Hàm kiểm tra vượt đèn đỏ (chỉ tính khi vượt từ dưới lên trên)
def is_red_light_violation(box, traffic_light_status, vehicle_history):
    center_x, center_y = get_center(box)
    
    # Nếu không có lịch sử (frame đầu tiên), không ghi nhận vi phạm
    if len(vehicle_history) < 2:
        return False

    # Kiểm tra nếu phương tiện từng ở dưới stopline
    was_below_stopline = any(pos[1] > STOP_LINE_Y for pos in vehicle_history[:-1])
    # Kiểm tra nếu hiện tại đã vượt qua stopline
    has_crossed = center_y < STOP_LINE_Y
    # Kiểm tra hướng di chuyển (center_y giảm qua các frame)
    prev_center_y = vehicle_history[-2][1]
    is_moving_up = prev_center_y > center_y

    # Ghi nhận vi phạm nếu: đèn đỏ, từng ở dưới stopline, hiện đã vượt qua, và đang di chuyển lên
    if (traffic_light_status == 'red' and was_below_stopline and has_crossed and is_moving_up):
        return True
    return False

# Hàm OCR
def ocr_license_plate(plate_img):
    results = ocr_model(plate_img)
    characters = []
    for result in results:
        for detection in result.boxes:
            label = result.names[int(detection.cls)]
            conf = float(detection.conf)
            if conf > 0.5:
                characters.append((detection.xyxy[0][0], label))
    characters.sort(key=lambda x: x[0])
    return ''.join([char[1] for char in characters])

# Biến để bỏ qua khung hình
frame_count = 0
FRAME_SKIP = 10  # Xử lý frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Resize khung hình để tăng tốc
    frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Vẽ vạch dừng lên khung hình
    cv2.line(frame_resized, (0, STOP_LINE_Y), (RESIZE_WIDTH, STOP_LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame_resized, f"Stop Line", (10, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Phát hiện xe máy và ô tô
    vehicle_results = vehicle_model(frame_resized)
    vehicles = []
    current_vehicles = {}
    for result in vehicle_results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            conf = float(box.conf)
            if label in ['car', 'motorcycle'] and conf > 0.5:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                box_coords = [x_min, y_min, x_max, y_max]
                vehicles.append((label, box_coords))
                current_vehicles[id(box)] = box_coords
                cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label} {conf:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Phát hiện biển số xe
    license_plates = []
    for vehicle in vehicles:
        label, box = vehicle
        vehicle_img = frame_resized[box[1]:box[3], box[0]:box[2]]
        plate_results = plate_model(vehicle_img)
        for result in plate_results:
            for plate_box in result.boxes:
                px_min, py_min, px_max, py_max = map(int, plate_box.xyxy[0])
                conf = float(plate_box.conf)
                if conf > 0.5:
                    abs_px_min = box[0] + px_min
                    abs_py_min = box[1] + py_min
                    abs_px_max = box[0] + px_max
                    abs_py_max = box[1] + py_max
                    license_plates.append([abs_px_min, abs_py_min, abs_px_max, abs_py_max])
                    cv2.rectangle(frame_resized, (abs_px_min, abs_py_min), (abs_px_max, abs_py_max),
                                  (255, 255, 0), 2)

    # OCR ký tự trong biển số
    plate_texts = []
    for plate_box in license_plates:
        plate_img = frame_resized[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
        plate_text = ocr_license_plate(plate_img)
        plate_texts.append(plate_text)
        cv2.putText(frame_resized, plate_text, (plate_box[0], plate_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Phát hiện trạng thái đèn giao thông
    traffic_light_results = traffic_light_model(frame_resized)
    traffic_light_status = None
    for result in traffic_light_results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.5:
                traffic_light_status = label
                cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame_resized, f"{label} {conf:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Kiểm tra vượt đèn đỏ và lưu vi phạm
    for i, vehicle in enumerate(vehicles):
        label, box = vehicle
        vehicle_id = id(box)
        center_x, center_y = get_center(box)

        # Lấy lịch sử vị trí của phương tiện
        vehicle_history = vehicle_positions[vehicle_id]

        # Kiểm tra vi phạm
        if is_red_light_violation(box, traffic_light_status, vehicle_history):
            plate_text = plate_texts[i] if i < len(plate_texts) else "Unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Phương tiện vi phạm: {label}, Biển số: {plate_text}, Thời gian: {timestamp}")

            # Cắt vùng phương tiện vi phạm từ khung hình gốc
            orig_box = [int(coord * scale_factor) for coord in box]
            vehicle_img = frame[orig_box[1]:orig_box[3], orig_box[0]:orig_box[2]]

            # Lưu hình ảnh vi phạm
            violation_filename = f"{VIOLATION_DIR}/{label}_{plate_text}_{timestamp}.jpg"
            cv2.imwrite(violation_filename, vehicle_img)

        # Cập nhật vị trí tâm
        vehicle_positions[vehicle_id].append((center_x, center_y))
        if len(vehicle_positions[vehicle_id]) > 10:
            vehicle_positions[vehicle_id].pop(0)

    # Hiển thị khung hình (nếu bật)
    if DISPLAY_FRAME:
        cv2.imshow('Traffic Violation Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()