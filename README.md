# LLM Routing with Matrix Factorization

## Tổng quan (Overview)
Dự án này triển khai mô hình định tuyến cho các Large Language Models (LLMs) dựa trên bài báo "RouteLLM: Learning to Route LLMs with Preference Data". Mô hình giúp tự động đưa ra quyết định gửi truy vấn (query) của người dùng đến một mô hình LLM mạnh (độ chính xác cao, chi phí cao) hoặc một mô hình LLM yếu hơn (độ chính xác thấp hơn, chi phí thấp) để tối ưu hóa sự cân bằng giữa chất lượng câu trả lời và chi phí.

## Kiến trúc (Architecture)
Phương pháp định tuyến được sử dụng trong dự án này là **Matrix Factorization**. Mô hình dự đoán xác suất chiến thắng của một mô hình mạnh so với một mô hình yếu bằng cách kết hợp:
1. **Query Embedding**: Sử dụng các mô hình mã nguồn mở như `sentence-transformers` (hoặc thông qua HuggingFace Inference API).
2. **Model Embedding**: Mỗi LLM được biểu diễn bằng một vector đặc trưng ($v_m$) học được trong quá trình huấn luyện.

## Dữ liệu (Dataset)
Sử dụng bộ dữ liệu **`NPULH/LLMRouterBench`** từ HuggingFace. Đây là bộ dữ liệu chuyên dụng dành cho việc huấn luyện và đánh giá các thuật toán định tuyến LLM, được công khai và hoàn toàn miễn phí.

## Cấu trúc thư mục (Project Structure)
- `model.py`: Chứa định nghĩa kiến trúc mô hình `MatrixFactorizationRouter` bằng PyTorch.
- `train.py`: Script để bắt đầu quá trình huấn luyện mô hình với Adam optimizer và BCE Loss.
- `evaluate.py`: Script dùng để đánh giá độ chính xác của mô hình sau khi huấn luyện.
- `run.py`: Script chính dùng để chạy thử nghiệm pipeline thực tế.
- `requirements.txt`: Danh sách các thư viện cần thiết.

## Cài đặt (Installation)
1. Clone repository về máy:
```bash
git clone <repository_url>
cd Model
```
2. Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

## Sử dụng (Usage)
1. Huấn luyện mô hình:
```bash
python train.py
```
2. Đánh giá mô hình:
```bash
python evaluate.py
```
3. Chạy pipeline định tuyến:
```bash
python run.py
```
