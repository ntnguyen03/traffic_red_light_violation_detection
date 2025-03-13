from ultralytics import YOLO
import cv2

# Load model YOLOv8 đã train
model = YOLO("models/traffic_light/traffic_light.pt")  # Thay bằng đường dẫn tới file .pt của bạn

# Hàm chạy thử trên ảnh
def test_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh!")
    
    # Dự đoán
    results = model(img)
    
    # Vẽ kết quả lên ảnh
    annotated_img = results[0].plot()  # Lấy ảnh đã annotate
    
    # Hiển thị
    cv2.imshow("Ket qua", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hàm chạy thử trên video
def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dự đoán
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Hiển thị
        cv2.imshow("Ket qua", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Chạy thử
if __name__ == "__main__":
    # Thay đường dẫn phù hợp
    # image_path = "path/to/your/test_image.jpg"
    video_path = "data_test/capture_video.mp4"
    
    # Test với ảnh
    # try:
    #     print("Đang xử lý ảnh...")
    #     test_image(image_path)
    # except Exception as e:
    #     print("Lỗi:", str(e))
    
    # Test với video (bỏ comment nếu muốn chạy)
    try:
        print("Đang xử lý video...")
        test_video(video_path)
    except Exception as e:
        print("Lỗi:", str(e))