import cv2
from ultralytics import YOLO
import numpy as np
import pytesseract
import easyocr
import re
from datetime import datetime
from collections import defaultdict
import os

# Tạo thư mục lưu ảnh debug
DEBUG_DIR = 'debug_plates'
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

# Cấu hình đường dẫn đến Tesseract nếu cần (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Khởi tạo EasyOCR làm backup
reader = easyocr.Reader(['en', 'vi'])

# Load các mô hình YOLOv8
vehicle_model = YOLO('yolov8m.pt')
plate_model = YOLO('models/license_plate/license_plate_detection.pt')
traffic_light_model = YOLO('models/traffic_light/traffic_light.pt')

# Đọc video
cap = cv2.VideoCapture('data_test/test2.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Định nghĩa vùng vạch dừng
STOP_LINE_Y = 450

# Lưu vị trí tâm của phương tiện
vehicle_positions = defaultdict(list)

DISPLAY_FRAME = True

# Hàm tính tâm của bounding box
def get_center(box):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

# Hàm kiểm tra vượt đèn đỏ
def is_red_light_violation(box, traffic_light_status, vehicle_history):
    center_x, center_y = get_center(box)
    if len(vehicle_history) < 2:
        return False
    was_below_stopline = any(pos[1] > STOP_LINE_Y for pos in vehicle_history[:-1])
    has_crossed = center_y < STOP_LINE_Y
    prev_center_y = vehicle_history[-2][1]
    is_moving_up = prev_center_y > center_y
    return traffic_light_status == 'red' and was_below_stopline and has_crossed and is_moving_up

# Mẫu định dạng biển số Việt Nam
plate_patterns = [
    r'^\d{2}[A-Z]\d{4,5}$',    # 59F12345
    r'^\d{2}[A-Z]-\d{4,5}$',    # 59F-12345
    r'^\d{2}[A-Z]\s\d{4,5}$',   # 59F 12345
    r'^\d{2}\s[A-Z]\s\d{4,5}$', # 59 F 12345
    r'^\d{2}-[A-Z]-\d{4,5}$'    # 59-F-12345
]

# Hàm kiểm tra biển số hợp lệ
def is_valid_plate(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    for pattern in plate_patterns:
        if re.match(pattern, cleaned_text):
            return True
    if re.match(r'.*\d+.*[A-Z]+.*\d+.*', cleaned_text) and 5 <= len(cleaned_text) <= 10:
        return True
    return False

# Hàm định dạng biển số
def format_plate_number(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if 5 <= len(cleaned_text) <= 10 and re.match(r'^\d{2}[A-Z]', cleaned_text[:3]):
        return f"{cleaned_text[:2]}{cleaned_text[2:3]}-{cleaned_text[3:]}"
    return cleaned_text

# Hàm OCR tối ưu hóa
def ocr_license_plate(plate_img, frame_count):
    if plate_img is None or plate_img.size == 0:
        return "Unknown"

    # Phóng to vùng biển số ngay từ đầu để giữ chi tiết
    plate_img = cv2.resize(plate_img, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

    # Lưu ảnh gốc để debug
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_raw_{frame_count}.jpg"), plate_img)

    # Tạo nhiều phiên bản tiền xử lý
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Tăng độ tương phản
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=-20)
    equalized = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Loại bỏ nhiễu
    dilated = cv2.dilate(cleaned, kernel, iterations=1)

    processed_images = [gray, equalized, thresh, cleaned, dilated]

    # Lưu các phiên bản tiền xử lý để debug
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_gray_{frame_count}.jpg"), gray)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_equalized_{frame_count}.jpg"), equalized)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_thresh_{frame_count}.jpg"), thresh)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_cleaned_{frame_count}.jpg"), cleaned)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"plate_dilated_{frame_count}.jpg"), dilated)

    all_results = []
    confidence_threshold = 0.4

    # Thử với pytesseract
    for proc_img in processed_images:
        try:
            plate_text = pytesseract.image_to_string(proc_img, config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
            cleaned_text = re.sub(r'[^A-Z0-9-]', '', plate_text.upper()).strip()
            if len(cleaned_text) >= 5:
                all_results.append((cleaned_text, 0.9))
        except Exception as e:
            print(f"Tesseract error: {e}")

    # Thử với EasyOCR làm backup
    for proc_img in processed_images:
        try:
            ocr_results = reader.readtext(proc_img)
            for (_, text, conf) in ocr_results:
                cleaned_text = re.sub(r'[^A-Z0-9-]', '', text.upper())
                if conf > confidence_threshold and len(cleaned_text) >= 5:
                    all_results.append((cleaned_text, conf))
        except Exception as e:
            print(f"EasyOCR error: {e}")

    # Sắp xếp theo độ tin cậy
    all_results.sort(key=lambda x: x[1], reverse=True)

    if not all_results:
        return "Unknown"

    best_text, _ = all_results[0]
    formatted_text = format_plate_number(best_text)
    if is_valid_plate(formatted_text):
        return formatted_text
    elif len(all_results) > 1:
        second_best, _ = all_results[1]
        formatted_second = format_plate_number(second_best)
        if is_valid_plate(formatted_second):
            return formatted_second
    return formatted_text

# Biến bỏ qua khung hình
frame_count = 0
FRAME_SKIP = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    # Vẽ vạch dừng
    cv2.line(frame, (0, STOP_LINE_Y), (frame_width, STOP_LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, "Stop Line", (10, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Phát hiện xe
    vehicle_results = vehicle_model(frame)
    vehicles = []
    for result in vehicle_results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            conf = float(box.conf)
            if label in ['car', 'motorcycle'] and conf > 0.5:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                box_coords = [x_min, y_min, x_max, y_max]
                vehicles.append((label, box_coords))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Phát hiện biển số
    license_plates = []
    for vehicle in vehicles:
        label, box = vehicle
        x_min, y_min, x_max, y_max = box
        vehicle_img = frame[y_min:y_max, x_min:x_max]
        plate_results = plate_model(vehicle_img)
        for result in plate_results:
            for plate_box in result.boxes:
                px_min, py_min, px_max, py_max = map(int, plate_box.xyxy[0])
                conf = float(plate_box.conf)
                if conf > 0.5:
                    abs_px_min = x_min + px_min
                    abs_py_min = y_min + py_min
                    abs_px_max = x_min + px_max
                    abs_py_max = y_min + py_max
                    license_plates.append([abs_px_min, abs_py_min, abs_px_max, abs_py_max])
                    plate_img = vehicle_img[py_min:py_max, px_min:px_max]
                    plate_text = ocr_license_plate(plate_img, frame_count)
                    cv2.rectangle(frame, (abs_px_min, abs_py_min), (abs_px_max, abs_py_max), (255, 0, 0), 2)
                    cv2.putText(frame, plate_text, (abs_px_min, abs_py_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Phát hiện trạng thái đèn giao thông
    traffic_light_results = traffic_light_model(frame)
    traffic_light_status = None
    for result in traffic_light_results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            conf = float(box.conf)
            if conf > 0.5:
                traffic_light_status = label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Kiểm tra vượt đèn đỏ
    for i, vehicle in enumerate(vehicles):
        label, box = vehicle
        vehicle_id = id(box)
        center_x, center_y = get_center(box)
        vehicle_history = vehicle_positions[vehicle_id]

        if is_red_light_violation(box, traffic_light_status, vehicle_history):
            plate_text = license_plates[i][4] if i < len(license_plates) and len(license_plates[i]) > 4 else "Unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Phương tiện vi phạm: {label}, Biển số: {plate_text}, Thời gian: {timestamp}")
            orig_box = [int(coord) for coord in box]
            vehicle_img = frame[orig_box[1]:orig_box[3], orig_box[0]:orig_box[2]]
            cv2.imwrite(f"violations/{label}_{plate_text}_{timestamp}.jpg", vehicle_img)

        vehicle_positions[vehicle_id].append((center_x, center_y))
        if len(vehicle_positions[vehicle_id]) > 10:
            vehicle_positions[vehicle_id].pop(0)

    if DISPLAY_FRAME:
        cv2.imshow('Traffic Violation Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()