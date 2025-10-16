# recommender.py (Sửa & hoàn chỉnh)
from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
from surprise import Dataset, Reader, SVD
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

CSV_PATH = "dataset.csv"
# =======================================
# 🔧 HÀM LOAD VÀ TRAIN MODEL CF
# =======================================
def load_and_train():
    """
    Đọc dataset từ CSV, chuẩn hóa các cột cơ bản và train SVD
    Trả về: (df, algo) - algo=None nếu không đủ dữ liệu để train
    """
    if not os.path.exists(CSV_PATH):
        # Trả DataFrame rỗng cùng algo None nếu chưa có file
        empty_cols = ["student_id", "year", "semester", "course_id", "course_name", "rating", "faculty_id", "status"]
        return pd.DataFrame(columns=empty_cols), None

    df_local = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Đảm bảo các cột tồn tại
    required_cols = ["student_id", "course_id", "course_name", "rating", "faculty_id", "year", "status"]
    for col in required_cols:
        if col not in df_local.columns:
            df_local[col] = None

    # Chuẩn hóa cột
    df_local["student_id"] = df_local["student_id"].astype(str)
    df_local["faculty_id"] = df_local["faculty_id"].astype(str)
    df_local["course_id"] = df_local["course_id"].astype(str)
    df_local["course_name"] = df_local["course_name"].astype(str)
    df_local["status"] = df_local["status"].fillna("not_started").astype(str)
    df_local["rating"] = pd.to_numeric(df_local["rating"], errors="coerce").fillna(0.0)
    # year có thể float trong CSV, ép về int nếu có thể
    try:
        df_local["year"] = pd.to_numeric(df_local["year"], errors="coerce").fillna(0).astype(int)
    except:
        df_local["year"] = df_local["year"].fillna(0)

    # Train chỉ trên các bản ghi marked 'completed' (đã có rating hợp lệ)
    train_df = df_local[df_local["status"].astype(str).str.lower() == "completed"]

    if train_df.empty or len(train_df) < 2:
        # Nếu chưa có đủ dữ liệu để train thì trả dataset chuẩn hóa và algo None
        print("⚠️ Chưa có dữ liệu completed đủ để train model CF.")
        return df_local, None

    try:
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(train_df[["student_id", "course_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)
        print("✅ Đã train model CF thành công.")
        return df_local, algo
    except Exception as e:
        print("⚠️ Lỗi khi train CF:", e)
        return df_local, None


# =======================================
# 🚀 KHỞI TẠO BIẾN TOÀN CỤC (ngoài hàm)
# =======================================
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    df, algo = load_and_train()
else:
    df, algo = pd.DataFrame(), None



# =======================================
# 🏠 HOME ROUTE
# =======================================
@app.route("/")
def home():
    return "✅ Flask API - Gợi ý môn học (Realtime Update)"


# =======================================
# 🎯 HELPER: xác định start_year & học năm hiện tại từ MSSV
# =======================================
def parse_start_year_from_mssv(sid):
    """
    Lấy 2 chữ số đầu trong MSSV để suy start year (ví dụ '248...' -> '24' -> 2024).
    Nếu không parse được, fallback về current year.
    """
    digits = ''.join(ch for ch in str(sid) if ch.isdigit())
    if len(digits) >= 2:
        try:
            yy = int(digits[:2])
            start = 2000 + yy
            if 2000 <= start <= 2099:
                return start
        except:
            pass
    return datetime.now().year


def compute_student_year_from_start(start_year):
    """
    Từ start_year (ví dụ 2024) -> compute academic year (1..4) theo current year
    """
    current_year = datetime.now().year
    student_year = current_year - start_year + 1
    if student_year < 1:
        student_year = 1
    if student_year > 4:
        student_year = 4
    return student_year


# =======================================
# 🎯 API: GỢI Ý MÔN HỌC
# =======================================
@app.route("/recommend/<student_id>/<faculty_id>", defaults={"year": None})
@app.route("/recommend/<student_id>/<faculty_id>/<year>")
def recommend(student_id, faculty_id, year):
    global df, algo

    # Đọc lại CSV để luôn đồng bộ với thay đổi mới nhất
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Không tìm thấy file dataset.csv"}), 404
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Chuẩn hóa cột cho an toàn
    df["student_id"] = df["student_id"].astype(str)
    df["faculty_id"] = df["faculty_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)
    if "course_name" not in df.columns:
        df["course_name"] = df["course_id"]
    df["course_name"] = df["course_name"].astype(str)
    if "status" not in df.columns:
        df["status"] = "not_started"
    df["status"] = df["status"].fillna("not_started").astype(str)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    else:
        df["year"] = 0

    # chuẩn input
    student_id = str(student_id).strip()
    faculty_id = str(faculty_id).strip().upper()

    # xác định start_year và student_year từ MSSV
    start_year = parse_start_year_from_mssv(student_id)
    student_year = compute_student_year_from_start(start_year)

    # Lọc theo faculty
    # =======================================
    # 🎯 LỌC DỮ LIỆU THEO KHOA & NĂM
    # =======================================
    filtered = df[df["faculty_id"].fillna("").astype(str).str.upper() == faculty_id.upper()]

    # Ưu tiên năm được truyền hoặc năm học hiện tại
    if year:
        try:
            filtered = filtered[filtered["year"].astype(float) == float(year)]
        except:
            pass
    else:
        try:
            same_year = filtered[filtered["year"].astype(float) == float(student_year)]
            if not same_year.empty:
                filtered = same_year
        except:
            pass

    # Nếu rỗng -> fallback toàn khoa
    if filtered.empty:
        filtered = df[df["faculty_id"].fillna("").astype(str).str.upper() == faculty_id.upper()]

    # ✅ Không lọc bỏ completed ở đây
    # Vì ta vẫn cần completed để biết sinh viên đã học gì


    # map id -> name
    course_map = dict(zip(filtered["course_id"].astype(str), filtered["course_name"].astype(str)))

    # kiểm tra student tồn tại trong dataset
    student_exists = student_id.upper() in df["student_id"].astype(str).str.upper().unique()

    # nếu chưa có student thì thêm các môn (not_started) cho student đó (dựa vào filtered)
    if not student_exists:
        courses_for_new = filtered.copy()
        if courses_for_new.empty:
            # fallback: lấy tất cả các môn
            courses_for_new = df.copy()

        new_rows = []
        for _, row in courses_for_new.iterrows():
            new_rows.append({
                "student_id": student_id,
                "year": int(row.get("year") if pd.notna(row.get("year")) else student_year),
                "semester": row.get("semester", 1),
                "course_id": str(row.get("course_id")),
                "course_name": str(row.get("course_name", "")),
                "rating": 0.0,
                "faculty_id": faculty_id,
                "status": "not_started"
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            # lưu lại CSV và retrain
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")
            df, algo = load_and_train()

    # Lấy danh sách môn đã completed của student trong khoa
    taken_mask = (
        (df["student_id"].astype(str).str.upper() == student_id.upper()) &
        (df["faculty_id"].astype(str).str.upper() == faculty_id.upper()) &
        (df["status"].astype(str).str.lower() == "completed")
    )
    taken_courses = df.loc[taken_mask, "course_id"].astype(str).tolist()
    # đổi sang tên để frontend hiển thị
    taken_names = [course_map.get(cid, cid) for cid in taken_courses]

    # Nếu chưa có model (algo None) thì fallback trả 5 môn chưa học (object list)
    # Nếu chưa có model (algo None) thì fallback trả 5 môn chưa học (object list)
    if algo is None:
    # Hiển thị tất cả môn chưa completed
        available = filtered[~filtered["course_id"].isin(taken_courses)]
        recommend_names = available[["course_id", "course_name"]].head(10)
        recommend_list = recommend_names.to_dict(orient="records")
        return jsonify({
            "student_id": student_id,
            "auto_year_detected": student_year,
            "start_year": start_year,
            "recommendations": recommend_list,
            "taken": taken_names
        })

    # Nếu có model, dự đoán score cho mỗi course chưa học
    available_ids = filtered[filtered["status"].astype(str).str.lower() != "completed"]["course_id"].astype(str).unique()
    predictions = []
    for cid in available_ids:
        if cid not in taken_courses:
            try:
                pred = algo.predict(student_id, cid)
                est = float(pred.est) if hasattr(pred, "est") else None
                if est is not None:
                    predictions.append((cid, course_map.get(cid, cid), est))
            except Exception:
                # nếu model không thể dự đoán (vd student unseen), bỏ qua
                continue

    # sắp xếp theo estimate desc
    predictions.sort(key=lambda x: x[2], reverse=True)

    # trả tối đa 10 kết quả (frontend chỉ lấy 5 hoặc render tùy)
    recommend_objs = []
    for cid, cname, score in predictions[:10]:
        recommend_objs.append({"course_id": cid, "course_name": cname, "score": round(score, 3)})

    return jsonify({
        "student_id": student_id,
        "auto_year_detected": student_year,
        "start_year": start_year,
        "recommendations": recommend_objs,
        "taken": taken_names
    })


# =======================================
# 🧾 API: CẬP NHẬT TRẠNG THÁI MÔN HỌC
# =======================================
@app.route("/update_status/<student_id>/<course_id>/<new_status>", methods=["POST"])
def update_status(student_id, course_id, new_status):
    global df, algo

    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Không tìm thấy file dataset.csv"}), 404

    # Đọc file mới nhất
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Chuẩn hóa cột
    df["student_id"] = df["student_id"].astype(str)
    df["faculty_id"] = df["faculty_id"].astype(str)
    df["course_id"] = df["course_id"].astype(str)
    if "status" not in df.columns:
        df["status"] = "not_started"
    df["status"] = df["status"].fillna("not_started").astype(str)

    # Chuẩn hóa input
    student_id_u = str(student_id).upper().strip()
    course_id_u = str(course_id).upper().strip()
    new_status_clean = str(new_status).lower().strip()

    # Tìm mask và cập nhật
    mask = (
        (df["student_id"].astype(str).str.upper() == student_id_u) &
        (df["course_id"].astype(str).str.upper() == course_id_u)
    )

    if mask.sum() == 0:
        return jsonify({"error": "Không tìm thấy môn học của sinh viên này"}), 404

    df.loc[mask, "status"] = new_status_clean

    # Lưu lại file và retrain
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    df, algo = load_and_train()

    # Trả về dữ liệu mới nhất của sinh viên để frontend render ngay
    student_data = df[df["student_id"].astype(str).str.upper() == student_id_u].to_dict(orient="records")

    return jsonify({
        "message": f"✅ Đã cập nhật {course_id_u} của {student_id_u} thành '{new_status_clean}' và retrain model.",
        "updated_courses": student_data
    })


# =======================================
# 🚀 CHẠY SERVER 
# =======================================
if __name__ == "__main__":
    # Khi Flask bật debug, nó sẽ tự restart -> ta chỉ muốn train ở lần chính
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("🚀 Flask đang chạy chính thức, model đã sẵn sàng.")
    else:
        print("🧩 Flask khởi động reloader, bỏ qua train model.")
    
    app.run(debug=True, use_reloader=True)
