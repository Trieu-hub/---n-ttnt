from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # bật CORS

# ====== Dataset từ CSV ======
df = pd.read_csv("dataset.csv")

@app.route("/")
def home():
    return "✅ API gợi ý môn học đang chạy..."

@app.route("/recommend/<student_id>")
def recommend(student_id):
    if student_id not in df["student_id"].values:
        return jsonify({"error": "Không tìm thấy sinh viên"}), 404
    
    courses = df[df["student_id"] == student_id]["course_name"].tolist()
    
    return jsonify({
        "student_id": student_id,
        "recommendations": courses   # ✅ Trả ra hết môn trong CSV
    })


if __name__ == "__main__":
    app.run(debug=True)
