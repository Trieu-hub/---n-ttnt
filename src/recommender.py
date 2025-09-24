from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# ====== Dataset giả lập ======
# Giả sử mỗi sinh viên có các môn đã học, ta sẽ gợi ý thêm môn mới
data = {
    "sv1": {
        "taken": ["Toán cao cấp", "Cấu trúc dữ liệu", "Lập trình C"],
        "recommend": ["Trí tuệ nhân tạo", "Lập trình Java", "Hệ điều hành"]
    },
    "sv2": {
        "taken": ["Kinh tế chính trị", "Nguyên lý kế toán"],
        "recommend": ["Quản trị tài chính", "Marketing căn bản"]
    },
    "sv3": {
        "taken": ["Lập trình Java", "Cơ sở dữ liệu"],
        "recommend": ["Web nâng cao", "Phát triển ứng dụng di động"]
    }
}

# Chuyển qua DataFrame cho dễ xử lý (nếu cần)
df = pd.DataFrame(data).T


# ====== API Routes ======

@app.route("/")
def home():
    return "✅ API gợi ý môn học đang chạy..."


@app.route("/recommend/<student_id>")
def recommend(student_id):
    if student_id in data:
        return jsonify({
            "student_id": student_id,
            "taken": data[student_id]["taken"],
            "recommendations": data[student_id]["recommend"]
        })
    else:
        return jsonify({
            "error": "Không tìm thấy sinh viên trong dataset",
            "recommendations": []
        }), 404


# ====== Run Flask ======
if __name__ == "__main__":
    app.run(debug=True)
