from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
from surprise import Dataset, Reader, SVD
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Cho phép tất cả origins để tránh lỗi CORS

CSV_PATH = "dataset.csv"

# Default courses for faculty "IT" or "CNTT" (Công nghệ Thông tin) if CSV is empty
# Based on sample program from UET Vietnam, simplified for 4 years, 2 semesters each
# Format: list of dicts with course_id, course_name, year, semester
DEFAULT_COURSES = [
    # Year 1, Semester 1
    {"course_id": "PHI1006", "course_name": "Triết học Mác – Lênin", "year": 1, "semester": 1},
    {"course_id": "MAT1093", "course_name": "Đại số", "year": 1, "semester": 1},
    {"course_id": "MAT1041", "course_name": "Giải tích 1", "year": 1, "semester": 1},
    {"course_id": "INT1007", "course_name": "Giới thiệu về Công nghệ thông tin", "year": 1, "semester": 1},
    {"course_id": "INT1008", "course_name": "Nhập môn lập trình", "year": 1, "semester": 1},
    
    # Year 1, Semester 2
    {"course_id": "PEC1008", "course_name": "Kinh tế chính trị Mác – Lênin", "year": 1, "semester": 2},
    {"course_id": "MAT1042", "course_name": "Giải tích 2", "year": 1, "semester": 2},
    {"course_id": "EPN1095", "course_name": "Vật lý đại cương 1", "year": 1, "semester": 2},
    {"course_id": "INT1050", "course_name": "Toán học rời rạc", "year": 1, "semester": 2},
    {"course_id": "FLF1107", "course_name": "Tiếng Anh B1", "year": 1, "semester": 2},
    
    # Year 2, Semester 1
    {"course_id": "PHI1002", "course_name": "Chủ nghĩa xã hội khoa học", "year": 2, "semester": 1},
    {"course_id": "EPN1096", "course_name": "Vật lý đại cương 2", "year": 2, "semester": 1},
    {"course_id": "INT2210", "course_name": "Cấu trúc dữ liệu và giải thuật", "year": 2, "semester": 1},
    {"course_id": "INT2215", "course_name": "Lập trình nâng cao", "year": 2, "semester": 1},
    {"course_id": "ELT2035", "course_name": "Tín hiệu và hệ thống", "year": 2, "semester": 1},
    
    # Year 2, Semester 2
    {"course_id": "HIS1001", "course_name": "Lịch sử Đảng Cộng sản Việt Nam", "year": 2, "semester": 2},
    {"course_id": "INT2212", "course_name": "Kiến trúc máy tính", "year": 2, "semester": 2},
    {"course_id": "INT2214", "course_name": "Nguyên lý hệ điều hành", "year": 2, "semester": 2},
    {"course_id": "INT2211", "course_name": "Cơ sở dữ liệu", "year": 2, "semester": 2},
    {"course_id": "INT2204", "course_name": "Lập trình hướng đối tượng", "year": 2, "semester": 2},
    
    # Year 3, Semester 1
    {"course_id": "POL1001", "course_name": "Tư tưởng Hồ Chí Minh", "year": 3, "semester": 1},
    {"course_id": "INT2213", "course_name": "Mạng máy tính", "year": 3, "semester": 1},
    {"course_id": "INT2208", "course_name": "Công nghệ phần mềm", "year": 3, "semester": 1},
    {"course_id": "INT3202", "course_name": "Hệ quản trị cơ sở dữ liệu", "year": 3, "semester": 1},
    {"course_id": "INT3110", "course_name": "Phân tích và thiết kế hướng đối tượng", "year": 3, "semester": 1},
    
    # Year 3, Semester 2
    {"course_id": "INT3306", "course_name": "Phát triển ứng dụng Web", "year": 3, "semester": 2},
    {"course_id": "INT3401", "course_name": "Trí tuệ nhân tạo", "year": 3, "semester": 2},
    {"course_id": "INT3507", "course_name": "Các vấn đề hiện đại Công nghệ thông tin", "year": 3, "semester": 2},
    {"course_id": "INT3117", "course_name": "Kiểm thử và đảm bảo chất lượng phần mềm", "year": 3, "semester": 2},
    {"course_id": "INT3105", "course_name": "Kiến trúc phần mềm", "year": 3, "semester": 2},
    
    # Year 4, Semester 1
    {"course_id": "INT3508", "course_name": "Thực tập chuyên ngành", "year": 4, "semester": 1},
    {"course_id": "INT3206", "course_name": "Cơ sở dữ liệu phân tán", "year": 4, "semester": 1},
    {"course_id": "INT3209", "course_name": "Khai phá dữ liệu", "year": 4, "semester": 1},
    {"course_id": "INT3301", "course_name": "Thực hành hệ điều hành mạng", "year": 4, "semester": 1},
    {"course_id": "INT3303", "course_name": "Mạng không dây", "year": 4, "semester": 1},
    
    # Year 4, Semester 2
    {"course_id": "INT3514", "course_name": "Pháp luật và đạo đức nghề nghiệp trong CNTT", "year": 4, "semester": 2},
    {"course_id": "INT3120", "course_name": "Phát triển ứng dụng di động", "year": 4, "semester": 2},
    {"course_id": "INT3307", "course_name": "An toàn và an ninh mạng", "year": 4, "semester": 2},
    {"course_id": "INT3319", "course_name": "Điện toán đám mây", "year": 4, "semester": 2},
    {"course_id": "INT3404", "course_name": "Xử lý ảnh", "year": 4, "semester": 2},
]

# 🔧 HÀM LOAD VÀ TRAIN MODEL CF

def load_and_train():
    """
    Đọc dataset từ CSV, chuẩn hóa các cột cơ bản và train SVD
    Trả về: (df, algo) - algo=None nếu không đủ dữ liệu để train
    """
    if not os.path.exists(CSV_PATH):
        # Tạo file CSV với header nếu chưa tồn tại
        empty_cols = ["student_id", "year", "semester", "course_id", "course_name", "rating", "faculty_id", "status"]
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        return empty_df, None

    df_local = pd.read_csv(CSV_PATH, encoding="utf-8", dtype=str)  # Đọc tất cả là string để tránh float infer

    # Đảm bảo các cột tồn tại
    required_cols = ["student_id", "course_id", "course_name", "rating", "faculty_id", "year", "status", "semester"]
    for col in required_cols:
        if col not in df_local.columns:
            df_local[col] = '' if col != "semester" else '1'  # Default semester to 1 if missing

    # Chuẩn hóa cột
    df_local["student_id"] = df_local["student_id"].astype(str).str.strip().str.upper().str.replace('.0', '')
    df_local["faculty_id"] = df_local["faculty_id"].astype(str).str.strip().str.upper()
    df_local["course_id"] = df_local["course_id"].astype(str).str.strip().str.upper().str.replace('.0', '')
    df_local["course_name"] = df_local["course_name"].astype(str).str.strip()
    df_local["status"] = df_local["status"].fillna("not_started").astype(str).str.strip().str.lower()
    df_local["rating"] = pd.to_numeric(df_local["rating"], errors="coerce").fillna(0.0)
    # year và semester ép về int
    df_local["year"] = pd.to_numeric(df_local["year"], errors="coerce").fillna(0).astype(int)
    df_local["semester"] = pd.to_numeric(df_local["semester"], errors="coerce").fillna(1).astype(int)

    # Lưu lại CSV với chuẩn hóa (để tránh lỗi lần sau)
    df_local.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # Train chỉ trên các bản ghi marked 'completed' và rating > 0 (tránh rating=0 làm nhiễu)
    train_df = df_local[(df_local["status"] == "completed") & (df_local["rating"] > 0)]

    if train_df.empty or len(train_df) < 2:
        # Nếu chưa có đủ dữ liệu để train thì trả dataset chuẩn hóa và algo None
        print("⚠️ Chưa có dữ liệu completed đủ để train model CF.")
        return df_local, None

    try:
        reader = Reader(rating_scale=(1, 5))  # Thay rating_scale từ 1-5 để tránh 0
        data = Dataset.load_from_df(train_df[["student_id", "course_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)
        print("✅ Đã train model CF thành công.")
        return df_local, algo
    except Exception as e:
        print("⚠️ Lỗi khi train CF:", e)
        return df_local, None



# 🚀 KHỞI TẠO BIẾN TOÀN CỤC (ngoài hàm)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    df, algo = load_and_train()
else:
    df, algo = pd.DataFrame(), None




# 🏠 HOME ROUTE

@app.route("/")
def home():
    return "✅ Flask API - Gợi ý môn học (Realtime Update)"



# 🎯 HELPER: xác định start_year & học năm hiện tại từ MSSV

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



# 🎯 API: GỢI Ý MÔN HỌC

@app.route("/recommend/<student_id>/<faculty_id>", defaults={"year": None})
@app.route("/recommend/<student_id>/<faculty_id>/<year>")
def recommend(student_id, faculty_id, year):
    global df, algo

    # Đọc lại CSV để luôn đồng bộ với thay đổi mới nhất
    df, algo = load_and_train()  # Reload và retrain để đảm bảo mới nhất

    # chuẩn input
    student_id = str(student_id).strip().upper().replace('.0', '')
    faculty_id = str(faculty_id).strip().upper()

    # xác định start_year và student_year từ MSSV
    start_year = parse_start_year_from_mssv(student_id)
    student_year = compute_student_year_from_start(start_year)

    # Xác định xem sinh viên có phải là mới (năm 1 hoặc chưa có completed courses)
    student_data = df[(df["student_id"] == student_id) & 
                      (df["faculty_id"] == faculty_id)]
    has_completed = not student_data[student_data["status"] == "completed"].empty
    is_new_student = (student_year < 2) or not has_completed

    # Lấy tất cả courses unique theo faculty (để tránh duplicate)
    all_courses = df[df["faculty_id"] == faculty_id].drop_duplicates(subset=["course_id", "year", "semester"])

    # Nếu không có courses nào cho faculty (CSV trống hoặc mới), sử dụng default courses (giả sử faculty "IT" hoặc "CNTT")
    if all_courses.empty and faculty_id in ["IT", "CNTT"]:  # Áp dụng cho cả "IT" và "CNTT"
        default_df = pd.DataFrame(DEFAULT_COURSES)
        default_df["faculty_id"] = faculty_id
        default_df["course_id"] = default_df["course_id"].str.upper()
        all_courses = default_df
        # Thêm default courses vào CSV nếu chưa có
        existing_courses = df[df["faculty_id"] == faculty_id]
        if existing_courses.empty:
            df = pd.concat([df, all_courses], ignore_index=True)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")
            df, algo = load_and_train()  # Reload sau khi thêm default

    # map id -> name, year, semester
    course_details = all_courses.set_index("course_id")[["course_name", "year", "semester"]].to_dict(orient="index")

    # Nếu student chưa tồn tại, thêm tất cả courses với status="not_started"
    if student_data.empty:
        if not course_details:  # Nếu vẫn không có courses
            return jsonify({"error": "Không có môn học nào cho khoa này. Vui lòng thêm dữ liệu mẫu vào CSV."}), 400
        new_rows = []
        for cid, details in course_details.items():
            new_rows.append({
                "student_id": student_id,
                "year": details["year"],
                "semester": details["semester"],
                "course_id": cid,
                "course_name": details["course_name"],
                "rating": 0.0,
                "faculty_id": faculty_id,
                "status": "not_started"
            })
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")
            df, algo = load_and_train()  # Retrain sau khi thêm
        is_new_student = True  # Vừa thêm, coi là mới

    # Lấy taken_courses (completed) của student trong faculty
    taken_mask = (
        (df["student_id"] == student_id) &
        (df["faculty_id"] == faculty_id) &
        (df["status"] == "completed")
    )
    taken_courses = df.loc[taken_mask, "course_id"].astype(str).unique().tolist()
    taken_details = [{"course_id": cid, "course_name": course_details.get(cid, {"course_name": cid})["course_name"]} for cid in taken_courses]

    # Lấy untaken_courses (status != completed)
    untaken_courses = [cid for cid in course_details.keys() if cid not in taken_courses]

    # Nếu year được chỉ định, filter untaken theo year
    if year:
        try:
            year_int = int(year)
            untaken_courses = [cid for cid in untaken_courses if course_details[cid]["year"] == year_int]
        except:
            pass

    # Logic gợi ý theo yêu cầu
    recommendations = []

    if is_new_student:
        # Sinh viên mới: Gợi ý TẤT CẢ courses trong 4 năm, sắp xếp theo year, semester, avg rating (nếu có)
        avg_ratings = df[df["status"] == "completed"].groupby("course_id")["rating"].mean().to_dict()
        for cid in course_details:
            details = course_details[cid]
            avg_rating = avg_ratings.get(cid, 0.0)
            recommendations.append({
                "course_id": cid,
                "course_name": details["course_name"],
                "year": details["year"],
                "semester": details["semester"],
                "predicted_rating": round(avg_rating, 3)
            })
        # Sắp xếp theo year asc, semester asc, predicted_rating desc
        recommendations.sort(key=lambda x: (x["year"], x["semester"], -x["predicted_rating"]))

    else:
        # Sinh viên năm 2+: Gợi ý TẤT CẢ untaken, sorted by predicted rating nếu có model, else by avg rating
        if algo is None:
            # Fallback: dùng avg rating
            avg_ratings = df[df["status"] == "completed"].groupby("course_id")["rating"].mean().to_dict()
            for cid in untaken_courses:
                details = course_details[cid]
                est = avg_ratings.get(cid, 0.0)
                recommendations.append({
                    "course_id": cid,
                    "course_name": details["course_name"],
                    "year": details["year"],
                    "semester": details["semester"],
                    "predicted_rating": round(est, 3)
                })
            recommendations.sort(key=lambda x: -x["predicted_rating"])
        else:
            # Dùng model predict
            for cid in untaken_courses:
                try:
                    pred = algo.predict(student_id, cid)
                    est = float(pred.est)
                except:
                    est = 0.0
                details = course_details[cid]
                recommendations.append({
                    "course_id": cid,
                    "course_name": details["course_name"],
                    "year": details["year"],
                    "semester": details["semester"],
                    "predicted_rating": round(est, 3)
                })
            recommendations.sort(key=lambda x: -x["predicted_rating"])

    return jsonify({
        "student_id": student_id,
        "auto_year_detected": student_year,
        "start_year": start_year,
        "recommendations": recommendations,  # Trả tất cả, không limit 10
        "taken": taken_details
    })



# 🧾 API: CẬP NHẬT TRẠNG THÁI MÔN HỌC

@app.route("/update_status/<student_id>/<course_id>/<faculty_id>/<new_status>", methods=["POST"])
def update_status(student_id, course_id, faculty_id, new_status):
    print(f"Received POST request to /update_status/{student_id}/{course_id}/{faculty_id}/{new_status}")  # Log để debug
    global df, algo

    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Không tìm thấy file dataset.csv"}), 404

    # Đọc file mới nhất với chuẩn hóa
    df, algo = load_and_train()

    # Chuẩn hóa input
    student_id_u = str(student_id).strip().upper().replace('.0', '')
    course_id_u = str(course_id).strip().upper().replace('.0', '')
    faculty_id_u = str(faculty_id).strip().upper()
    new_status_clean = str(new_status).lower().strip()

    # Nếu update thành "completed", yêu cầu rating từ request (nếu có)
    new_rating = request.json.get("rating", 0.0) if request.is_json else 0.0
    new_rating = float(new_rating) if new_status_clean == "completed" else 0.0

    # Tìm mask và cập nhật
    mask = (
        (df["student_id"] == student_id_u) &
        (df["course_id"] == course_id_u) &
        (df["faculty_id"] == faculty_id_u)
    )

    if mask.sum() == 0:
        print("No matching rows. Printing relevant data for debug:")
        print(df[(df["student_id"] == student_id_u) & (df["faculty_id"] == faculty_id_u)])
        return jsonify({"error": "Không tìm thấy môn học của sinh viên này"}), 404

    df.loc[mask, "status"] = new_status_clean
    df.loc[mask, "rating"] = new_rating  # Cập nhật rating nếu completed

    # Lưu lại file và retrain
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    df, algo = load_and_train()

    # Trả về dữ liệu mới nhất của sinh viên để frontend render ngay
    student_data = df[df["student_id"] == student_id_u].to_dict(orient="records")

    return jsonify({
        "message": f"✅ Đã cập nhật {course_id_u} của {student_id_u} thành '{new_status_clean}' và retrain model.",
        "updated_courses": student_data
    })



# 🚀 CHẠY SERVER 

if __name__ == "__main__":
    # Khi Flask bật debug, nó sẽ tự restart -> ta chỉ muốn train ở lần chính
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("🚀 Flask đang chạy chính thức, model đã sẵn sàng.")
    else:
        print("🧩 Flask khởi động reloader, bỏ qua train model.")
    
    app.run(host='0.0.0.0', debug=True, use_reloader=True)  # Chạy trên 0.0.0.0 để accessible từ network/browser