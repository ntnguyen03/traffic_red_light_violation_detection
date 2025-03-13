import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# Đường dẫn tới các mô hình đã train
vehicle_model_path = "yolov8m.pt"  # Mô hình phát hiện phương tiện (YOLOv8m)
license_plate_model_path = "models/license_plate/license_plate_detection.pt"  # Mô hình phát hiện biển số
ocr_model_path = "models/license_plate/license_plate_ocr.pt"  # Mô hình OCR ký tự
traffic_light_model_path = "models/traffic_light/traffic_light.pt"  # Mô hình phát hiện đèn giao thông

# Khởi tạo các mô hình
vehicle_detector = YOLO(vehicle_model_path)
license_plate_detector = YOLO(license_plate_model_path)
ocr_detector = YOLO(ocr_model_path)  # Giả sử OCR của bạn dùng YOLO
traffic_light_detector = YOLO(traffic_light_model_path)

# Định nghĩa stopline (giả sử là một đường ngang cố định)
STOPLINE_Y = 500  # Tọa độ y của stopline (tùy chỉnh theo video)

# Thư mục lưu ảnh vi phạm
VIOLATION_DIR = "violations"
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# Hàm kiểm tra trạng thái đèn giao thông
def get_traffic_light_state(frame):
    results = traffic_light_detector(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    for box, cls in zip(boxes, classes):
        if cls == 0:
            return "green"
        elif cls == 1:
            return "red"
        elif cls == 2:
            return "yellow"
    return "unknown"  # Nếu không phát hiện đèn

# Hàm kiểm tra phương tiện vượt đèn đỏ
def check_red_light_violation(vehicle_box, traffic_light_state):
    # vehicle_box: [x_min, y_min, x_max, y_max]
    if traffic_light_state == "red" and vehicle_box[3] > STOPLINE_Y:  # Nếu đèn đỏ và xe vượt stopline
        return True
    return False

# Hàm tách ký tự từ biển số
def split_characters(lp_image):
    # Chuyển ảnh sang grayscale và binary
    gray = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]  # Lọc contour nhỏ
    
    # Sắp xếp từ trái sang phải
    char_boxes = sorted(char_boxes, key=lambda x: x[0])
    
    chars = []
    for (x, y, w, h) in char_boxes:
        char_img = lp_image[y:y+h, x:x+w]
        chars.append((char_img, (x, y)))  # Lưu ảnh ký tự và tọa độ
    return chars

# Hàm nhận diện ký tự bằng OCR thực tế
def recognize_characters(chars):
    lp_text = ""
    y_coords = []
    for char_img, (x, y) in chars:
        results = ocr_detector(char_img)
        if len(results[0].boxes) > 0:
            cls = int(results[0].boxes.cls[0].cpu().numpy())  # Lấy class của ký tự
            lp_text += str(cls) if cls < 10 else chr(65 + (cls - 10))  # 0-9, A-Z
            y_coords.append(y)
    return lp_text, y_coords

# Hàm kiểm tra biển số 1 dòng hay 2 dòng
def check_license_plate_lines(y_coords):
    if not y_coords:
        return "unknown"
    y_diff = max(y_coords) - min(y_coords)
    if y_diff > 20:  # Ngưỡng để xác định 2 dòng (tùy chỉnh)
        return "2 lines"
    return "1 line"

# Hàm chuẩn hóa biển số
def normalize_license_plate(lp_text):
    # Ví dụ: Chuẩn hóa theo định dạng Việt Nam (2 số - 1 chữ - 4/5 số)
    if len(lp_text) >= 7 and lp_text[0].isdigit() and lp_text[2].isalpha():
        return lp_text[:2] + "-" + lp_text[2] + "-" + lp_text[3:]
    return "ERROR"

# Hàm xử lý biển số xe
def process_license_plate(image, lp_box):
    x_min, y_min, x_max, y_max = map(int, lp_box)
    lp_image = image[y_min:y_max, x_min:x_max]
    
    # Tách và nhận diện ký tự
    chars = split_characters(lp_image)
    lp_text, y_coords = recognize_characters(chars)
    
    # Kiểm tra 1/2 dòng
    lp_type = check_license_plate_lines(y_coords)
    
    # Chuẩn hóa biển số
    lp_text = normalize_license_plate(lp_text)
    
    return lp_text, lp_image, lp_type

# Hàm vẽ bbox và text lên ảnh
def draw_annotations(image, box, text, color=(0, 0, 255)):
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Hàm lưu ảnh vi phạm
def save_violation(image, vehicle_id, license_plate_text, lp_type):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{VIOLATION_DIR}/{timestamp}_ID{vehicle_id}_{lp_type}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved violation: {filename}")

# Main processing loop
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Phát hiện phương tiện
        vehicle_results = vehicle_detector(frame)
        vehicles = vehicle_results[0].boxes.xyxy.cpu().numpy()
        
        # Phát hiện trạng thái đèn giao thông
        traffic_state = get_traffic_light_state(frame)
        
        # Vẽ stopline
        cv2.line(frame, (0, STOPLINE_Y), (frame.shape[1], STOPLINE_Y), (255, 0, 0), 2)
        
        # Xử lý từng phương tiện
        for i, vehicle_box in enumerate(vehicles):
            vehicle_id = i  # Thay bằng tracker thực tế
            
            if check_red_light_violation(vehicle_box, traffic_state):
                lp_results = license_plate_detector(frame)
                lp_boxes = lp_results[0].boxes.xyxy.cpu().numpy()
                
                for lp_box in lp_boxes:
                    if (lp_box[0] >= vehicle_box[0] and lp_box[2] <= vehicle_box[2] and
                        lp_box[1] >= vehicle_box[1] and lp_box[3] <= vehicle_box[3]):
                        lp_text, lp_image, lp_type = process_license_plate(frame, lp_box)
                        
                        draw_annotations(frame, vehicle_box, f"ID: {vehicle_id}")
                        draw_annotations(frame, lp_box, f"{lp_text} ({lp_type})")
                        
                        save_violation(frame.copy(), vehicle_id, lp_text, lp_type)
                        break
        
        cv2.imshow("Red Light Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data_test/test_video_1.mp4"
    process_video(video_path)