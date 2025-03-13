# Hệ thống nhận diện phương tiện vượt đèn đỏ sử dụng YOLOv8 và AIoT 🎥
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>EJS</title><path d="m8.291 5.696-1.258-1.58 1.668-1.55 1.258 1.58-1.668 1.55zm2.34 2.048.205-1.55-5.412-.03-.204 1.55 3.945.022L7.8 17.852l-.839.77-.405-.004c.385.403.803.81 1.236 1.214l1.378-1.265 1.458-10.823h.004zm-6.757 7.254s2.925-.468 3.54.38c0 0-1.756-2.925-4.184-2.925 0-.074-.098-1.353 5.146-2.609l.206-1.53c-8.346 1.108-14.287 4.322.265 13.12 0 0-5.675-4.71-4.973-6.436zM13 6.223 11.216 7.86l-.526 4.037 1.316 1.638 5.675.058.556.702-.38 2.633-.713.685-.018.017h2.193l.556-4.037-1.345-1.638-5.646-.058-.556-.702.351-2.633.731-.702 5.032.058.556.673-.176 1.229h1.55l.264-1.902-1.317-1.667-6.318-.03zm2.882 11.908.545-.523-4.305-.035-.965-1.17-1.258 1.17 1.346 1.667 6.318.03 1.22-1.139h-2.901zM13.13 8.965a103.16 103.16 0 0 1 4.624-.554l-4.145-.048-.457.44-.022.162zm8.026-1.156-.025.179-.018.132c.92-.07 1.87-.139 2.887-.2 0 0-1.113-.067-2.844-.11zM1.914 18.392l1.404 1.784 2.66.02c-1.292-.875-2.393-1.708-3.296-2.499l-.768.695z"/></svg>
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

## Cách sử dụng ✅✅
- Tải models đã được pre-train cho các đặc trưng (phương tiện, biển số, đèn giao thông, ký tự biển số)
```bash
python main.py
```
- Phát triển để models chạy với độ chính xác cao hơn
- Kết hợp rtsp videostream để chạy video với Camera
