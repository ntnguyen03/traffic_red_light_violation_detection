import cv2
from ultralytics import YOLO

# Load model YOLOv8m pre-trained từ Ultralytics
model = YOLO('yolov8m.pt')  # Tải YOLOv8m pre-trained

# Đọc video đầu vào
video_path = 'data_test/test_video_2.mp4'  # Đường dẫn đến video thử nghiệm
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở được không
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Danh sách tên lớp từ COCO dataset (chỉ cần class ID 2 và 3)
class_names = {2: 'car', 3: 'motorcycle'}  # Chỉ lấy car và motorcycle

# Màu sắc cho bounding box (BGR format)
COLORS = {
    'car': (0, 255, 0),       # Màu xanh lá cho car
    'motorcycle': (0, 0, 255)  # Màu đỏ cho motorcycle
}

# Lặp qua từng frame của video
while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame
    if not ret:
        print("End of video or error reading frame.")
        break

    # Bước 1: Phát hiện phương tiện
    results = model(frame)

    # Lấy thông tin từ kết quả phát hiện
    for result in results:
        boxes = result.boxes  # Lấy danh sách các bounding boxes
        for box in boxes:
            # Lấy tọa độ và lớp
            class_id = int(box.cls[0])  # ID của lớp

            # Chỉ xử lý class ID 2 (car) và 3 (motorcycle)
            if class_id not in [2, 3]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Chuyển tọa độ về kiểu int
            confidence = box.conf[0]  # Độ tin cậy
            class_name = class_names[class_id]  # Tên lớp dựa trên ID

            print(f"Detected {class_name} with confidence: {confidence:.2f}")

            # Vẽ bounding box
            color = COLORS.get(class_name, (255, 255, 255))  # Màu mặc định nếu không tìm thấy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn và độ tin cậy
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Hiển thị frame đã xử lý
    cv2.imshow('Vehicle Detection (Car & Motorcycle)', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()