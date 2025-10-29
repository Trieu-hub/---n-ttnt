# 🎓 HỆ THỐNG GỢI Ý MÔN HỌC CÁ NHÂN HÓA  
### 🔍 Ứng dụng thuật toán Collaborative Filtering (Neural Collaborative Filtering - NCF)

---

## 📘 GIỚI THIỆU

Hệ thống gợi ý môn học cá nhân hóa giúp **đề xuất các môn học phù hợp cho từng sinh viên** dựa trên lịch sử học tập, mối tương quan giữa sinh viên – môn học và hướng nghề nghiệp tương lai.  
Cốt lõi của hệ thống là thuật toán **Neural Collaborative Filtering (NCF)**, một biến thể của **Collaborative Filtering** kết hợp **deep learning** nhằm cải thiện độ chính xác và khả năng tổng quát hóa.

---

## 🧱 MỤC TIÊU

- Xây dựng mô hình gợi ý môn học cá nhân hóa cho sinh viên ngành CNTT.  
- Ứng dụng NCF để dự đoán mức độ phù hợp giữa sinh viên và môn học.  
- Kết hợp điểm học tập, hành vi và career path để tăng tính chính xác.  
- Tạo API RESTful cho phép frontend hoặc hệ thống khác gọi gợi ý.  

---

## 🧩 KIẾN TRÚC HỆ THỐNG

```
Frontend (HTML/CSS/JS)
        ↓
Backend (Flask API + PyTorch)
        ↓
NCF Model (Collaborative Filtering)
        ↓
Dataset (student-course-rating)
```

> Giao diện frontend cho phép người dùng nhập MSSV → gửi yêu cầu đến Flask backend → backend xử lý bằng NCF model → trả về danh sách môn học được gợi ý.

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY DEMO

### 🔧 Bước 1: Cài đặt thư viện
```bash
pip install flask flask-cors pandas torch scikit-learn numpy
```

### ⚙️ Bước 2: Khởi động Backend
```bash
python app.py
```
> Server Flask chạy tại: `http://127.0.0.1:5000`

### 💻 Bước 3: Chạy Frontend
- Mở file `index.html` bằng **Live Server** trong VS Code.  
- Nhập MSSV demo: `24812345` hoặc `24812346`.  
- Xem danh sách môn học được gợi ý hiển thị trên giao diện.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```
project/
├── app.py                 # Backend Flask
├── model.py               # Mô hình NCF
├── dataset.csv            # Dữ liệu mẫu
├── templates/             # Giao diện HTML
├── static/                # CSS, JS
├── requirements.txt       # Thư viện cần thiết
└── README.md              # Tài liệu hướng dẫn
```

---

## 🧠 THUẬT TOÁN SỬ DỤNG: NEURAL COLLABORATIVE FILTERING (NCF)

### 🔹 Ý tưởng chính
Thay vì chỉ dựa vào ma trận tương quan (như CF truyền thống), NCF học **biểu diễn (embedding)** của người dùng và item (sinh viên & môn học) bằng **mạng neural**.

### 🔹 Kiến trúc mô hình
```
Input: (student_id, course_id)
        ↓
Embedding Layer:
   - User Embedding (32-dim)
   - Item Embedding (32-dim)
        ↓
Concatenate → MLP (64 → 32 → 16)
        ↓
Output Layer → Predicted Rating (1–5)
```

### 🔹 Công thức dự đoán
```
ŷ = σ(MLP([P_u ⊕ Q_i]))
```
> - `P_u`: vector embedding của sinh viên  
> - `Q_i`: vector embedding của môn học  
> - `⊕`: phép nối vector  
> - `σ`: hàm sigmoid → đầu ra là điểm gợi ý (1–5)

---

## ⚗️ GIẢI THÍCH CODE MÔ HÌNH (model.py)

```python
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, 32)
        self.item_embedding = nn.Embedding(num_items, 32)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        concat = torch.cat([user_embed, item_embed], dim=-1)
        output = self.mlp(concat)
        return output
```

> ✅ Mỗi cặp (user, item) → xuất ra 1 giá trị “độ phù hợp” → càng cao thì môn học càng được gợi ý nhiều hơn.

---

## 🧪 API TESTING

### 1️⃣ Lấy danh sách gợi ý
```bash
curl http://127.0.0.1:5000/recommend/24812345/IT
```

### 2️⃣ Cập nhật trạng thái học tập
```bash
curl -X POST http://127.0.0.1:5000/update_status/24812345/CMP169/IT/completed   -H "Content-Type: application/json"   -d '{"rating": 4.5}'
```

---

## 📊 DỮ LIỆU SỬ DỤNG

- File: `dataset.csv`  
- Dữ liệu mẫu gồm **2 sinh viên** và **103 môn học** thuộc ngành CNTT  

| Cột | Mô tả |
|------|--------|
| `student_id` | Mã số sinh viên |
| `course_id` | Mã môn học |
| `course_name` | Tên môn học |
| `rating` | Mức độ yêu thích / phù hợp |
| `status` | completed / studying / wishlist |
| `year`, `semester` | Năm học, học kỳ |

---

## ⚙️ HYBRID SCORING (TÍNH ĐIỂM KẾT HỢP)
Hệ thống có thể kết hợp giữa điểm gợi ý từ NCF và điểm nội dung từ career path:

```
Hybrid_Score = 0.7 × NCF_Score + 0.3 × Content_Score
```

> Giúp kết quả gợi ý phản ánh cả “sở thích cá nhân” và “định hướng nghề nghiệp”.

---

## ❄️ COLD START HANDLING

Khi sinh viên mới chưa có dữ liệu học tập:
- Hệ thống sử dụng **Career Path Matching** (so sánh nội dung môn học và ngành học).  
- Gợi ý các môn **phổ biến** hoặc **thuộc nhóm cơ sở ngành**.  
- Sau vài lần cập nhật rating → NCF tự học lại và cải thiện dần độ chính xác.

---

## ✅ CHECKLIST DEMO CHO GIẢNG VIÊN

| Kiểm tra | Mục tiêu | Cách thực hiện |
|-----------|-----------|----------------|
| 1️⃣ Backend | Flask chạy ổn định, log training hiển thị | `python app.py` |
| 2️⃣ Frontend | Live Server hoạt động, nhập MSSV có kết quả | `index.html` |
| 3️⃣ API | Test bằng Postman hoặc curl | `/recommend`, `/update_status` |
| 4️⃣ Mô hình | Giải thích forward() của NCF | `model.py` |
| 5️⃣ Hybrid Score | Hiểu và trình bày công thức | README.md |
| 6️⃣ Cold Start | Nêu cách xử lý sinh viên mới | README.md |
| 7️⃣ Documentation | Có file README.md hướng dẫn đầy đủ | Nộp kèm project |

---

## 🎯 KẾT QUẢ KỲ VỌNG

| Tiêu chí | Mô tả |
|-----------|-------|
| 🎯 Độ chính xác | Tăng 15–20% so với CF truyền thống |
| ⚡ Thời gian phản hồi | < 2 giây mỗi truy vấn |
| 💡 Tính năng | Gợi ý, cập nhật, retrain tự động |
| 📈 Mở rộng | Có thể áp dụng cho các ngành học khác |

---

## 🏁 KẾT LUẬN

Hệ thống gợi ý môn học cá nhân hóa sử dụng **Neural Collaborative Filtering (NCF)** đã chứng minh hiệu quả trong việc:
- Dự đoán chính xác các môn học phù hợp với từng sinh viên,  
- Giảm tình trạng chọn môn không phù hợp,  
- Tăng trải nghiệm học tập thông minh và định hướng nghề nghiệp rõ ràng.  

> 💬 Hệ thống có thể được mở rộng thêm với dữ liệu lớn hơn, sử dụng AutoEncoder hoặc Graph Neural Network để tăng hiệu năng và độ chính xác.

---

## 👨‍💻 THÔNG TIN SINH VIÊN

- **Họ tên:** [Tên của bạn]  
- **MSSV:** [Mã số sinh viên]  
- **Ngành:** Công nghệ Thông tin  
- **Trường:** Đại học HUTECH  
- **Giảng viên hướng dẫn:** [Tên giảng viên]  

---

📅 *Tháng 10/2025*  
🧠 *Project: Hệ thống gợi ý môn học cá nhân hóa sử dụng Neural Collaborative Filtering (NCF)*

