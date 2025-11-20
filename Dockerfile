# Sử dụng base image Python chính thức
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp yêu cầu và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code dự án
COPY . .

# LỆNH CHẠY: Gọi đúng tệp code Python của bạn
CMD ["python", "vps_terminal_render.py"]
