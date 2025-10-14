from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
from surprise import Dataset, Reader, SVD
import os

app = Flask(__name__)
CORS(app)

CSV_PATH = "dataset.csv"

# ======================================================
# 🧠 HÀM LOAD DATASET VÀ TRAIN MODEL
# ======================================================
def load_and_train():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(), None

    df = pd.read_csv(CSV_PATH)

    # Đảm bảo các cột cần thiết tồn tại
    required_cols = ["student_id", "course_id", "course_name", "rating", "faculty_id", "year", "status"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Điền giá trị mặc định cho cột status (tránh lỗi)
    df["status"] = df["status"].fillna("not_started")

    # Đảm bảo dữ liệu hợp lệ
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(1).astype(int)

    # Chỉ train với các môn đã hoàn thành
    train_df = df[df["status"].str.lower() == "completed"]

    if train_df.empty:
        return df, None  # Chưa có dữ liệu đủ để train

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["student_id", "course_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    return df, algo


# ======================================================
# 🔄 KHỞI TẠO BAN ĐẦU
# ======================================================
df, algo = load_and_train()


# ======================================================
# 🏠 HOME ROUTE
# ======================================================
@app.route("/")
def home():
    return "✅ Flask API - Gợi ý môn học (Realtime Update)"


# ======================================================
# 🎯 API: GỢI Ý MÔN HỌC
# ======================================================
@app.route("/recommend/<student_id>/<faculty_id>", defaults={"year": None})
@app.route("/recommend/<student_id>/<faculty_id>/<year>")
def recommend(student_id, faculty_id, year):
    global df, algo

    if df.empty:
        return jsonify({"error": "Dataset trống hoặc chưa tồn tại"}), 400

    student_id = student_id.upper().strip()
    faculty_id = str(faculty_id).strip().upper()

    # --- Lọc theo khoa ---
    filtered = df[df["faculty_id"].astype(str).str.upper() == faculty_id]

    # --- Lọc theo năm nếu có ---
    if year:
        try:
            filtered = filtered[filtered["year"].astype(float) == float(year)]
        except:
            pass

    # --- Map ID → Tên môn ---
    course_map = dict(zip(filtered["course_id"], filtered["course_name"]))

    # --- Danh sách môn đã học ---
    taken = df[
        (df["student_id"].astype(str).str.upper() == student_id)
        & (df["faculty_id"].astype(str).str.upper() == faculty_id)
        & (df["status"].astype(str).str.lower() == "completed")
    ]["course_id"].tolist()

    taken_names = [course_map.get(cid, cid) for cid in taken]

    # --- Nếu chưa có model hoặc chưa có dữ liệu completed ---
    if algo is None or df[df["status"].str.lower() == "completed"].empty:
        available_courses = filtered[
            filtered["status"].str.lower() != "completed"
        ][["course_id", "course_name"]].drop_duplicates().to_dict(orient="records")

        return jsonify({
            "student_id": student_id,
            "recommendations": available_courses[:5],
            "taken": taken_names
        })

    # --- Các môn chưa học ---
    available = filtered[
        ~filtered["course_id"].isin(taken)
    ]["course_id"].unique()

    predictions = []
    for cid in available:
        try:
            est = algo.predict(student_id, cid).est
            if pd.notna(est):
                predictions.append((cid, est))
        except:
            continue

    # --- Sắp xếp theo điểm dự đoán ---
    predictions.sort(key=lambda x: x[1], reverse=True)

    recommend_names = [{
        "course_id": cid,
        "course_name": course_map.get(cid, cid),
        "score": round(score, 2)
    } for cid, score in predictions[:5]]

    return jsonify({
        "student_id": student_id,
        "recommendations": recommend_names,
        "taken": taken_names
    })




# ======================================================
# 🧾 API: CẬP NHẬT TRẠNG THÁI MÔN HỌC
# ======================================================
@app.route("/update_status/<student_id>/<course_id>/<new_status>", methods=["POST"])
def update_status(student_id, course_id, new_status):
    global df, algo

    student_id = student_id.strip().upper()
    course_id = course_id.strip().upper()
    new_status = new_status.strip().lower()

    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Không tìm thấy file dataset.csv"}), 404

    # Đọc file
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Đảm bảo cột status tồn tại
    if "status" not in df.columns:
        df["status"] = "not_started"
    df["status"] = df["status"].fillna("not_started")

    # Cập nhật trạng thái
    mask = (
        (df["student_id"].astype(str).str.upper() == student_id)
        & (df["course_id"].astype(str).str.upper() == course_id)
    )

    if mask.sum() == 0:
        return jsonify({"error": "Không tìm thấy môn học của sinh viên này"}), 404

    df.loc[mask, "status"] = new_status

    # Lưu lại
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # Retrain CF model
    df, algo = load_and_train()

    # Trả lại dữ liệu sinh viên
    student_data = df[df["student_id"].str.upper() == student_id].to_dict(orient="records")

    return jsonify({
        "message": f"✅ Đã cập nhật {course_id} của {student_id} thành '{new_status}' và retrain model.",
        "updated_courses": student_data
    })


# ======================================================
# 🚀 CHẠY SERVER
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
