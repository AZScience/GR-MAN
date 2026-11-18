# GR-MAN: Gated Residual Multimodal Attention Network
### Hệ Thống Trí Tuệ Nhân Tạo Học Sâu Đa Phương Thức Dự Báo Sớm Biến Cố Bất Lợi Trong Chăm Sóc Sức Khỏe (Clinical CDSS)

Đề tài: **Nghiên cứu mô hình học sâu đa phương thức cho dự báo sớm các biến cố bất lợi trong chăm sóc sức khỏe**

---

## 📌 Tổng Quan Giải Pháp GR-MAN
Hệ thống tích hợp kiến trúc tiên tiến **GR-MAN (Gated Residual Multimodal Attention Network)** giải quyết triệt để 2 thách thức lớn nhất của dữ liệu hồi sức cấp cứu (ICU):
1. **Residual Identity Highway:** Bảo toàn dòng tín hiệu số sinh hóa (Sinh hiệu, Creatinine, BUN, Kali, Natri...), loại bỏ nguy cơ văn bản bệnh án dài lấn át (Modality Dominance/Collapse).
2. **Gated Multimodal Unit (GMU):** Cổng điều tiết động đóng/mở tự thích ứng tùy thuộc độ đầy đủ của từng phương thái, giữ vững độ chính xác khi khuyết thiếu dữ liệu cấp cứu.

---

## 📊 Cơ Sở Dữ Liệu Nghiên Cứu Chuẩn Quốc Tế
- **MIMIC-IV / MIMIC-CXR:** Bộ dữ liệu mã nguồn mở lớn nhất thế giới của MIT/Harvard gồm dữ liệu EHR, ghi chú lâm sàng và ảnh X-quang của hàng chục nghìn bệnh nhân ICU.
- **eICU Collaborative Research Database:** Dữ liệu đa trung tâm về bệnh nhân chăm sóc tích cực.

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Chạy trên máy cục bộ (Local Environment):
```bash
# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Khởi chạy giao diện ứng dụng Streamlit
streamlit run app.py
# (hoặc: streamlit run app_medical_pipeline.py)
```
Ứng dụng sẽ mở tự động tại: `http://localhost:8501`

### 2. Triển khai lên Streamlit Community Cloud:
1. Đăng nhập vào [Streamlit Community Cloud](https://share.streamlit.io/).
2. Chọn **New app**.
3. Chọn Repository: `AZScience/GR-MAN`.
4. Chọn Branch: `kiemtranoibo`.
5. Đặt Main file path: `app.py` (hoặc `app_medical_pipeline.py`).
6. Nhấn **Deploy**!
