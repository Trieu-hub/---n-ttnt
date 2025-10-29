# 🎓 HỆ THỐNG GỢI Ý MÔN HỌC CÁ NHÂN HÓA  
### 🔍 Sử dụng thuật toán Collaborative Filtering (Neural Collaborative Filtering - NCF)

---

## 📘 GIỚI THIỆU DỰ ÁN

Hệ thống được xây dựng nhằm **gợi ý các môn học phù hợp cho từng sinh viên**, dựa trên:
- Lịch sử học tập (các môn đã học và điểm đánh giá),
- Mối quan hệ giữa sinh viên và các môn học,
- Kết hợp với thông tin ngành học (career path).

Thuật toán cốt lõi là **Neural Collaborative Filtering (NCF)** — một biến thể nâng cao của **Collaborative Filtering (CF)**, sử dụng **mạng nơ-ron nhân tạo** để học đặc trưng giữa sinh viên và môn học.

---

## 🚀 HƯỚNG DẪN CHẠY DEMO

### 🔧 Bước 1: Cài đặt thư viện cần thiết
```bash
pip install flask flask-cors pandas torch scikit-learn numpy
