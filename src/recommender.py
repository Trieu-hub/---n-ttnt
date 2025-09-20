# Cài thư viện trước (nếu chưa có):
# pip install scikit-surprise pandas

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# =========================
# 1. Tạo dữ liệu giả cho Khoa CNTT
# =========================
# user_id: mã sinh viên
# item_id: mã môn học
# rating: mức độ "hot" (sinh viên đánh giá từ 1-5)

data_dict = {
    "user_id": ["sv1", "sv1", "sv2", "sv2", "sv3", "sv3", "sv4", "sv5", "sv5", "sv6"],
    "item_id": [
        "Lập trình C", "Cơ sở dữ liệu", 
        "Lập trình C", "Java OOP", 
        "Cơ sở dữ liệu", "Mạng máy tính", 
        "Trí tuệ nhân tạo", 
        "Hệ điều hành", "Java OOP", 
        "Machine Learning"
    ],
    "rating": [5, 4, 4, 5, 5, 3, 4, 3, 4, 5]
}

df = pd.DataFrame(data_dict)

# =========================
# 2. Load dữ liệu vào Surprise
# =========================
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# =========================
# 3. Chia dữ liệu train/test
# =========================
trainset, testset = train_test_split(data, test_size=0.25)

# =========================
# 4. Huấn luyện bằng SVD (Matrix Factorization)
# =========================
model = SVD()
model.fit(trainset)

# =========================
# 5. Dự đoán & đánh giá
# =========================
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# =========================
# 6. Gợi ý top môn học hot cho 1 sinh viên
# =========================
def get_top_n(predictions, n=3):
    # Gom dự đoán cho từng user
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Lấy top n theo dự đoán cao nhất
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=3)

# =========================
# 7. In gợi ý
# =========================
for uid, user_ratings in top_n.items():
    print(f"\n📌 Gợi ý cho sinh viên {uid}:")
    for iid, est in user_ratings:
        print(f"   - {iid} (dự đoán hot: {est:.2f})")
