# Hệ thống nhận diện phương tiện vượt đèn đỏ sử dụng YOLOv8 và AIoT 🎥

## Giới thiệu 💡💡💡
Dự án này là bài tập lớn môn AIoT, tập trung vào việc xây dựng một hệ thống thông minh để nhận diện và ghi nhận các phương tiện vượt đèn đỏ tại giao lộ. Hệ thống kết hợp công nghệ trí tuệ nhân tạo (AI) và Internet vạn vật (IoT), sử dụng mô hình YOLOv8 để xử lý video từ camera giao thông theo thời gian thực. Mục tiêu là nâng cao hiệu quả giám sát giao thông, giảm thiểu vi phạm và đảm bảo an toàn đường bộ.

Hệ thống bao gồm 4 thành phần chính:
- **Phát hiện phương tiện**: Sử dụng YOLOv8m pre-trained để nhận diện xe máy (`motorcycle`) và ô tô (`car`).
- **Phát hiện biển số xe**: Dùng mô hình YOLOv8 tùy chỉnh để xác định vị trí biển số trên phương tiện.
- **OCR biển số xe**: Trích xuất ký tự trên biển số bằng mô hình YOLOv8 hoặc các thư viện OCR.
- **Nhận diện đèn giao thông**: Phân loại trạng thái đèn (xanh, đỏ, vàng) với mô hình YOLOv8 tùy chỉnh.

Các phương tiện vi phạm (vượt đèn đỏ) sẽ được ghi nhận, bao gồm hình ảnh, biển số và thời gian vi phạm, được lưu vào thư mục `violations`. Hệ thống có khả năng tích hợp IoT để gửi dữ liệu vi phạm đến cơ quan quản lý giao thông hoặc lưu trữ trên đám mây.

## Tính năng chính 🚀🚀
- Nhận diện xe máy và ô tô với độ chính xác khá cao
- Phát hiện và đọc biển số xe theo thời gian thực.
- Nhận diện trạng thái đèn giao thông (xanh, đỏ, vàng).
- Ghi nhận phương tiện vi phạm vượt đèn đỏ từ dưới vạch dừng lên trên.
- Tối ưu tốc độ xử lý khung hình bằng cách resize video và bỏ qua khung hình không cần thiết.
- Lưu trữ hình ảnh vi phạm vào thư mục `violations` với thông tin chi tiết (loại phương tiện, biển số, thời gian).

## Công nghệ sử dụng 🖥🖥
- **YOLOv8**: Mô hình deep learning để phát hiện đối tượng và OCR.
- **OpenCV**: Xử lý video và hình ảnh.
- **Python**: Ngôn ngữ lập trình chính.

## Hình ảnh Demo 🎞🎞
![image](https://github.com/user-attachments/assets/cddb5c82-5d84-44d1-aa16-4461657c3305)
