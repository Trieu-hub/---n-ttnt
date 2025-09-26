from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS
from surprise import Dataset, Reader, SVD

app = Flask(__name__)
CORS(app)

# ====== Dataset từ CSV ======
df = pd.read_csv("dataset.csv")

required_cols = {'student_id', 'course_id', 'course_name', 'rating', 'faculty_id', 'year'}
if required_cols.issubset(df.columns):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['student_id', 'course_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
else:
    algo = None

@app.route("/")
def home():
    return "✅ API gợi ý môn học đang chạy..."

@app.route("/recommend/<student_id>/<faculty_id>", defaults={"year": None})
@app.route("/recommend/<student_id>/<faculty_id>/<year>")
def recommend(student_id, faculty_id, year):
    if student_id.upper() not in [str(s).upper() for s in df["student_id"].values]:
        return jsonify({"error": "Không tìm thấy sinh viên"}), 404

    # Lọc các môn thuộc khoa và năm học (nếu có)
    if year is not None:
        try:
            year_int = int(year)
            faculty_year_df = df[(df["faculty_id"] == faculty_id) & (df["year"] == year_int)]
        except:
            faculty_year_df = df[df["faculty_id"] == faculty_id]
    else:
        faculty_year_df = df[df["faculty_id"] == faculty_id]

    course_map = dict(zip(faculty_year_df["course_id"], faculty_year_df["course_name"]))
    taken_courses = faculty_year_df[faculty_year_df["student_id"] == student_id]["course_id"].tolist()
    taken_names = [course_map[cid] for cid in taken_courses if cid in course_map]

    if algo is None:
        return jsonify({
            "student_id": student_id,
            "recommendations": [],
            "taken": taken_names
        })

    all_courses = faculty_year_df["course_id"].unique()
    predictions = []
    for course_id in all_courses:
        if course_id not in taken_courses:
            pred = algo.predict(student_id, course_id)
            predictions.append((course_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_courses = [course_map[cid] for cid, _ in predictions[:5]]

    return jsonify({
        "student_id": student_id,
        "recommendations": recommended_courses,
        "taken": taken_names
    })

if __name__ == "__main__":
    app.run(debug=True)
