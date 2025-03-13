import cv2
from ultralytics import YOLO

# Load 2 model YOLOv8 đã train
detector_model = YOLO('models/license_plate/license_plate_detection.pt')  # Model phát hiện biển số
ocr_model = YOLO('models/license_plate/license_plate_ocr.pt')                   # Model nhận diện ký tự

# Đọc video đầu vào
video_path = 'data_test/test2.mp4'  # Đường dẫn đến video thử nghiệm
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở được không
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Màu đỏ cho bounding box và text (BGR format)
RED_COLOR = (0, 0, 255)

# Lặp qua từng frame của video
while cap.isOpened():
    ret, frame = cap.read()  # Đọc frame
    if not ret:
        print("End of video or error reading frame.")
        break

    # Bước 1: Phát hiện biển số xe
    detector_results = detector_model(frame)

    # Lấy thông tin từ kết quả phát hiện
    for result in detector_results:
        boxes = result.boxes  # Lấy danh sách các bounding boxes
        for box in boxes:
            # Lấy tọa độ của biển số
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Chuyển tọa độ về kiểu int
            confidence = box.conf[0]  # Độ tin cậy
            print(f"Detected license plate with confidence: {confidence}")

            # Cắt vùng ảnh chứa biển số
            license_plate_img = frame[y1:y2, x1:x2]

            # Bước 2: Nhận diện ký tự trên biển số đã cắt
            ocr_results = ocr_model(license_plate_img)

            # Xử lý kết quả OCR
            plate_text = ""
            for ocr_result in ocr_results:
                ocr_boxes = ocr_result.boxes
                for ocr_box in ocr_boxes:
                    char = ocr_box.cls  # Lớp ký tự (giả sử model trả về ký tự dưới dạng class)
                    plate_text += str(char)  # Ghép các ký tự lại

            # Tùy chỉnh text hiển thị (thêm ID và trạng thái vi phạm)
            display_text = f"{plate_text}" 
            print(f"License plate text: {display_text}")

            # Vẽ bounding box màu đỏ
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED_COLOR, 2)

            # Vẽ text màu đỏ bên trên bounding box
            cv2.putText(frame, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED_COLOR, 2)

    # Hiển thị frame đã xử lý
    cv2.imshow('License Plate Detection', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()