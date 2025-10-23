from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

CSV_PATH = "dataset.csv"

# Default courses for faculty "IT" or "CNTT" (Công nghệ Thông tin) if CSV is empty
# Based on sample program from UET Vietnam, simplified for 4 years, 2 semesters each
# Format: list of dicts with course_id, course_name, year, semester
DEFAULT_COURSES = [
    # Year 1, Semester 1
    {"course_id": "ENC120", "course_name": "Anh ngữ 1", "year": 1, "semester": 1},
    {"course_id": "CMP363", "course_name": "Công tác kỹ ngành cntt", "year": 1, "semester": 1},
    {"course_id": "CMP1074", "course_name": "Cơ sở lập trình", "year": 1, "semester": 1},
    {"course_id": "CMP3075", "course_name": "Thực hành cơ sở lập trình", "year": 1, "semester": 1},
    
    # Year 1, Semester 2
    {"course_id": "ENC121", "course_name": "Anh Ngữ 2", "year": 1, "semester": 2},
    {"course_id": "MAT118", "course_name": "Giải tích 1", "year": 1, "semester": 2},
    {"course_id": "MAT101", "course_name": "Đại số tuyến tính", "year": 1, "semester": 2},
    {"course_id": "CMP164", "course_name": "Kỹ thuật lập trình", "year": 1, "semester": 2},
    {"course_id": "CMP365", "course_name": "Thực hành kỹ thuật lập trình", "year": 1, "semester": 2},
    {"course_id": "COS137", "course_name": "Nhập môn kiến trúc máy tính", "year": 1, "semester": 2},
    {"course_id": "COS319", "course_name": "Thực hành kiến trúc máy tính", "year": 1, "semester": 2},
    
    # Year 1, Semester 3
    {"course_id": "SKL115", "course_name": "Tư duy thiết kế dự án", "year": 1, "semester": 3},
    {"course_id": "PSY167", "course_name": "Tâm lý học ứng dụng", "year": 1, "semester": 3},
    
    
    # Year 2, Semester 1
    {"course_id": "ENC122", "course_name": "Anh Ngữ 3", "year": 2, "semester": 1},
    {"course_id": "POS103", "course_name": "Tư tưởng Hồ Chí Minh", "year": 2, "semester": 1},
    {"course_id": "COS120", "course_name": "Cấu trúc dữ liệu và giải thuật", "year": 2, "semester": 1},
    {"course_id": "COS321", "course_name": "Thực hành cấu trúc dữ liệu và giải thuật", "year": 2, "semester": 1},
    {"course_id": "COS138", "course_name": "Nhập môn hệ điều hành", "year": 2, "semester": 1},
    {"course_id": "COS318", "course_name": "Thực hành hệ điều hành", "year": 2, "semester": 1},
    {"course_id": "MAT105", "course_name": "Xác xuất thống kê", "year": 2, "semester": 1},

    # Year 2, Semester 2
    {"course_id": "ENC123", "course_name": "Anh Ngữ 4", "year": 2, "semester": 2},
    {"course_id": "MAT104", "course_name": "Toán rời rạc", "year": 2, "semester": 2},
    {"course_id": "CMP3014", "course_name": "Thực hành lý thuyết đồ thị", "year": 2, "semester": 2},
    {"course_id": "COS136", "course_name": "Nhập môn cơ sở dữ liệu", "year": 2, "semester": 2},
    {"course_id": "COS323", "course_name": "Thực hành cơ sở dữ liệu", "year": 2, "semester": 2},
    {"course_id": "CMP172", "course_name": "Mạng máy tính", "year": 2, "semester": 2},
    {"course_id": "CMP373", "course_name": "Thực hành mạng máy tính", "year": 2, "semester": 2},
    {"course_id": "CMP167", "course_name": "Lập trình hướng đối tượng", "year": 2, "semester": 2},
    {"course_id": "CMP368", "course_name": "Thực hành lập trình hướng đối tượng", "year": 2, "semester": 2},

    # Year 2, Semester 3
    {"course_id": "SKL116", "course_name": "Đổi mới sáng tạo và tư duy khởi nghiệp", "year": 2, "semester": 3},
    {"course_id": "ENS109", "course_name": "Môi trường", "year": 2, "semester": 3},
    {"course_id": "POS107", "course_name": "Lịch sử đảng cộng sản việt nam", "year": 2, "semester": 3},
    {"course_id": "CMP383", "course_name": "Thực hành an toàn trên máy chủ Windows", "year": 2, "semester": 3},

    # Year 3, Semester 1
    {"course_id": "CMP169", "course_name": "Trí tuệ nhân tạo", "year": 3, "semester": 1},
    {"course_id": "COS136", "course_name": "Phân tích và quản trị cơ sở dữ liệu", "year": 3, "semester": 1},
    {"course_id": "CMP385", "course_name": "Thực hành phân tích thiết kế hệ thống", "year": 3, "semester": 1},
    {"course_id": "COS324", "course_name": "Thực hành quản trị cơ sở dữ liệu", "year": 3, "semester": 1},
    {"course_id": "CMP174", "course_name": "Bảo mật thông tin", "year": 3, "semester": 1},
    {"course_id": "CMP382", "course_name": "Thực hành bảo mật thông tin", "year": 3, "semester": 1},
    {"course_id": "CMP170", "course_name": "Lập trình môi trường windows", "year": 3, "semester": 1},
    {"course_id": "CMP371", "course_name": "Thực hành lập trình môi trường windows", "year": 3, "semester": 1},
    
    # Year 3, Semester 2
    {"course_id": "COS101", "course_name": "Cơ sở dữ liệu nâng cao", "year": 3, "semester": 2},
    {"course_id": "CMP184", "course_name": "Phân tích thiết kế hệ thống", "year": 3, "semester": 2},
    {"course_id": "CMP3019", "course_name": "Thực hành phân tích thiết kế hệ thống theo hướng đối tượng", "year": 3, "semester": 2},
    {"course_id": "CMP301", "course_name": "Công nghê phần mềm", "year": 3, "semester": 2},
    {"course_id": "COS340", "course_name": "Thực hành phát triển phần mềm mã nguồn mở", "year": 3, "semester": 2},
    {"course_id": "COS339", "course_name": "Thực hành bảo mật thông tin nâng cao", "year": 3, "semester": 2},
    {"course_id": "CMP175", "course_name": "Lập trình web", "year": 3, "semester": 2},
    {"course_id": "CMP376", "course_name": "Thực hành lập trình web", "year": 3, "semester": 2},
    
    # Year 3, Semester 3
    {"course_id": "CMP437", "course_name": "Đồ án cơ sở Công nghệ thông tin", "year": 3, "semester": 3},
    {"course_id": "LAW106", "course_name": "Pháp luật đại cương", "year": 3, "semester": 3},
    
    # Year 4, Semester 1
    {"course_id": "MAN104", "course_name": "Quản lý dự án công nghệ thông tin", "year": 4, "semester": 1},
    {"course_id": "POS105", "course_name": "Kinh tế chính trị Mác-Lênin", "year": 4, "semester": 1},
    {"course_id": "CMP180", "course_name": "Lập trình mạng máy tính", "year": 4, "semester": 1},
    {"course_id": "CMP381", "course_name": "Thực hành lập trình mạng máy tính", "year": 4, "semester": 1},
    {"course_id": "CMP436", "course_name": "Đồ án chuyên ngành công nghệ thông tin", "year": 4, "semester": 1},
    {"course_id": "CMP173", "course_name": "Lập trình trên thiết bị di động", "year": 4, "semester": 1},
    
    # Year 4, Semester 2
    {"course_id": "CMP596", "course_name": "Thực tập tốt nghiệp ngành Công nghệ thông tin", "year": 4, "semester": 2},
    #nhóm 1
    {"course_id": "CMP386", "course_name": "Công cụ và môi trường phát triển phần mềm", "year": 4, "semester": 2},
    {"course_id": "COS141", "course_name": "Phát triển ứng dụng J2SE", "year": 4, "semester": 2},
    {"course_id": "CMP179", "course_name": "Kiểm thử và đảm bảo chất lượng phần mềm", "year": 4, "semester": 2},
    {"course_id": "CAP126", "course_name": "Ngôn ngữ phát triển ứng dụng mới", "year": 4, "semester": 2},
    #nhóm 2
    {"course_id": "COS125", "course_name": "Cơ sở dữ liệu phân tán", "year": 4, "semester": 2},
    {"course_id": "COS126", "course_name": "Hệ quản trị cơ sở dữ liệu Oracle", "year": 4, "semester": 2},
    {"course_id": "COS127", "course_name": "Kho dữ liệu và khai thác dữ liệu", "year": 4, "semester": 2},
    {"course_id": "CMP189", "course_name": "Phân tích dữ liệu trên điện toán đám mây", "year": 4, "semester": 2},
    #nhóm 3
    {"course_id": "CMP191", "course_name": "Quản trị mạng", "year": 4, "semester": 2},
    {"course_id": "CMP192", "course_name": "Mạng máy tính nâng cao", "year": 4, "semester": 2},
    {"course_id": "COS128", "course_name": "Hệ điều hành Linux", "year": 4, "semester": 2},
    {"course_id": "COS129", "course_name": "Điện toán đám mây", "year": 4, "semester": 2},
    #nhóm 4
    {"course_id": "CMP1020", "course_name": "Học sâu", "year": 4, "semester": 2},
    {"course_id": "CMP1021", "course_name": "Thị giác máy tính", "year": 4, "semester": 2},
    {"course_id": "CMP1022", "course_name": "Trí tuệ nhân tạo cho internet vạn vật", "year": 4, "semester": 2},
    {"course_id": "CMP1023", "course_name": "Công nghệ ứng dụng robot", "year": 4, "semester": 2},
    #nhóm 5
    {"course_id": "COS130", "course_name": "An toàn hệ điều hành và ngôn ngữ lập trình", "year": 4, "semester": 2},
    {"course_id": "CMP195", "course_name": "An toàn hệ thống mạng máy tính", "year": 4, "semester": 2},
    {"course_id": "CMP194", "course_name": "An toàn thông tin cho ứng dụng web", "year": 4, "semester": 2},
    {"course_id": "CMP193", "course_name": "Phân tích và đánh giá an toàn thông tin", "year": 4, "semester": 2},
    #nhóm 6
    {"course_id": "CMP497", "course_name": "Đồ án tốt nghiệp ngành Công nghệ thông tin", "year": 4, "semester": 2},
    
    # Year 4, Semester 3
    {"course_id": "POS104", "course_name": "Triết học Mác-Lênin", "year": 4, "semester": 3},
    {"course_id": "POS106", "course_name": "Chủ nghĩa xã hội khoa học", "year": 4, "semester": 3},
]
# Mở rộng: Map categories courses để suy đoán career paths
CAREER_MAP = {
    #  Web Development & Mobile
    'web_dev': [
        'CMP175',  # Lập trình web
        'CMP376',  # Thực hành lập trình web
        'CMP173',  # Lập trình trên thiết bị di động
        'CMP371',  # Thực hành lập trình môi trường windows
        'CMP170',  # Lập trình môi trường windows
        'COS141',  # Phát triển ứng dụng J2SE
    ],

    #  Data Science / AI / Machine Learning
    'data_science': [
        'MAT105',   # Xác suất thống kê
        'COS127',   # Kho dữ liệu và khai thác dữ liệu
        'CMP189',   # Phân tích dữ liệu trên điện toán đám mây
        'CMP1020',  # Học sâu
        'CMP1021',  # Thị giác máy tính
        'CMP1022',  # Trí tuệ nhân tạo cho Internet vạn vật
        'CMP169',   # Trí tuệ nhân tạo
    ],

    #  Network & Security
    'network_security': [
        'CMP172',   # Mạng máy tính
        'CMP373',   # Thực hành mạng máy tính
        'CMP191',   # Quản trị mạng
        'CMP192',   # Mạng máy tính nâng cao
        'CMP180',   # Lập trình mạng máy tính
        'CMP381',   # Thực hành lập trình mạng máy tính
        'CMP174',   # Bảo mật thông tin
        'CMP382',   # Thực hành bảo mật thông tin
        'COS130',   # An toàn hệ điều hành và ngôn ngữ lập trình
        'CMP195',   # An toàn hệ thống mạng máy tính
        'CMP194',   # An toàn thông tin cho ứng dụng web
        'CMP193',   # Phân tích và đánh giá an toàn thông tin
        'COS128',   # Hệ điều hành Linux
    ],

    #  Database & Data Engineering
    'database': [
        'COS136',   # Nhập môn cơ sở dữ liệu
        'COS323',   # Thực hành cơ sở dữ liệu
        'COS101',   # Cơ sở dữ liệu nâng cao
        'COS324',   # Thực hành quản trị cơ sở dữ liệu
        'COS125',   # Cơ sở dữ liệu phân tán
        'COS126',   # Hệ quản trị cơ sở dữ liệu Oracle
        'COS136',   # Phân tích và quản trị cơ sở dữ liệu
    ],

    #  Software Engineering
    'software_eng': [
        'CMP301',   # Công nghệ phần mềm
        'CMP184',   # Phân tích thiết kế hệ thống
        'CMP3019',  # Thực hành phân tích thiết kế hệ thống theo hướng đối tượng
        'CMP385',   # Thực hành phân tích thiết kế hệ thống
        'CMP386',   # Công cụ và môi trường phát triển phần mềm
        'MAN104',   # Quản lý dự án công nghệ thông tin
        'CMP179',   # Kiểm thử và đảm bảo chất lượng phần mềm
        'COS340',   # Thực hành phát triển phần mềm mã nguồn mở
        'CMP437',   # Đồ án cơ sở CNTT
        'CMP436',   # Đồ án chuyên ngành CNTT
        'CMP497',   # Đồ án tốt nghiệp CNTT
    ],
}


# Map career paths và resources dựa trên categories
CAREER_SUGGESTIONS = {
    # 🌐 WEB DEVELOPMENT & MOBILE
    'web_dev': {
        'paths': [
            'Frontend Developer',
            'Backend Developer',
            'Full-Stack Developer',
            'Mobile App Developer'
        ],
        'resources': [
            {'name': 'The Web Developer Bootcamp 2024 – Udemy', 
                'url': 'https://www.udemy.com/course/the-web-developer-bootcamp/'},
            {'name': 'React.js Course – freeCodeCamp', 
                'url': 'https://www.freecodecamp.org/learn/front-end-development-libraries/'},
            {'name': 'Full Stack Open – University of Helsinki', 
                'url': 'https://fullstackopen.com/en/'},
            {'name': 'Google Android Developer Certification', 
                'url': 'https://developers.google.com/certification/android-developer'}
        ]
    },

    # 🧠 DATA SCIENCE / AI / MACHINE LEARNING
    'data_science': {
        'paths': [
            'Data Scientist',
            'Machine Learning Engineer',
            'AI Engineer',
            'Computer Vision Engineer',
            'Data Analyst'
        ],
        'resources': [
            {'name': 'Machine Learning by Andrew Ng – Coursera', 
                'url': 'https://www.coursera.org/learn/machine-learning'},
            {'name': 'Google Advanced Data Analytics Certificate', 
                'url': 'https://www.coursera.org/professional-certificates/google-advanced-data-analytics'},
            {'name': 'Deep Learning Specialization – Coursera', 
                'url': 'https://www.coursera.org/specializations/deep-learning'},
            {'name': 'Python for Data Science – freeCodeCamp', 
                'url': 'https://www.freecodecamp.org/learn/data-analysis-with-python/'}
        ]
    },

    # 🛡️ NETWORK & CYBERSECURITY
    'network_security': {
        'paths': [
            'Network Engineer',
            'Cybersecurity Analyst',
            'Security Operations (SOC) Specialist',
            'Penetration Tester',
            'DevSecOps Engineer'
        ],
        'resources': [
            {'name': 'Cisco CCNA Networking Basics – Udemy', 
                'url': 'https://www.udemy.com/course/ccna-complete/'},
            {'name': 'CompTIA Security+ Certification', 
                'url': 'https://www.comptia.org/certifications/security'},
            {'name': 'Google Cybersecurity Certificate – Coursera', 
                'url': 'https://www.coursera.org/professional-certificates/google-cybersecurity'},
            {'name': 'Ethical Hacking Essentials – Coursera', 
                'url': 'https://www.coursera.org/learn/ethical-hacking-essentials-ehe'}
        ]
    },

    # 🗃️ DATABASE & DATA ENGINEERING
    'database': {
        'paths': [
            'Database Administrator (DBA)',
            'Data Engineer',
            'SQL Developer',
            'ETL Specialist',
            'Cloud Database Engineer'
        ],
        'resources': [
            {'name': 'SQL for Data Science – Coursera', 
                'url': 'https://www.coursera.org/learn/sql-for-data-science'},
            {'name': 'Oracle Database SQL Certified Associate', 
                'url': 'https://education.oracle.com/oracle-database-sql-certified-associate/trackp_820'},
            {'name': 'Data Engineering on Google Cloud – Coursera', 
                'url': 'https://www.coursera.org/professional-certificates/gcp-data-engineering'},
            {'name': 'MongoDB – The Complete Developer’s Guide – Udemy', 
                'url': 'https://www.udemy.com/course/mongodb-the-complete-developers-guide/'}
        ]
    },

    # 💻 SOFTWARE ENGINEERING
    'software_eng': {
        'paths': [
            'Software Engineer',
            'Backend Engineer',
            'QA/Test Engineer',
            'DevOps Engineer',
            'Project Manager'
        ],
        'resources': [
            {'name': 'Software Engineering Basics for Everyone – edX', 
                'url': 'https://www.edx.org/course/software-engineering-basics-for-everyone'},
            {'name': 'ISTQB Certified Tester Foundation Level', 
                'url': 'https://www.istqb.org/certifications.html'},
            {'name': 'Google Project Management Certificate', 
                'url': 'https://www.coursera.org/professional-certificates/google-project-management'},
            {'name': 'DevOps on AWS – Coursera', 
                'url': 'https://www.coursera.org/learn/devops-on-aws'}
        ]
    },
}


# NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential()
        input_dim = embedding_dim * 2
        for i, dim in enumerate(layers):
            self.mlp.add_module(f'linear{i}', nn.Linear(input_dim, dim))
            self.mlp.add_module(f'relu{i}', nn.ReLU())
            input_dim = dim
        self.output = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        concat = torch.cat([user_embed, item_embed], dim=-1)
        mlp_out = self.mlp(concat)
        pred = self.sigmoid(self.output(mlp_out))
        return pred * 5  # Scale to 1-5

class RatingDataset(TorchDataset):
    def __init__(self, df):
        # users/items phải là Long (index cho embedding)
        self.users = torch.tensor(df['student_id_int'].values, dtype=torch.long)
        self.items = torch.tensor(df['course_id_int'].values, dtype=torch.long)
        # ratings phải là float32 và giữ scale 0..5 (không chia /5)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# 🔧 HÀM LOAD VÀ TRAIN MODEL NCF với Hybrid chuẩn bị
def load_and_train():
    """
    Đọc dataset từ CSV, chuẩn hóa, train NCF, và chuẩn bị TF-IDF cho content-based
    Trả về: (df, model, user_map, item_map, vectorizer, course_index_map, course_vectors)
    model=None nếu không đủ dữ liệu
    """
    if not os.path.exists(CSV_PATH):
        # Tạo file CSV với header nếu chưa tồn tại
        empty_cols = ["student_id", "year", "semester", "course_id", "course_name", "rating", "faculty_id", "status"]
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        return empty_df, None, {}, {}, None, {}, None

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

    # Chuẩn bị TF-IDF cho tất cả courses unique
    all_courses = df_local.drop_duplicates(subset=['course_id'])
    course_names = all_courses['course_name'].values
    course_ids = all_courses['course_id'].values

    if len(all_courses) > 0:
        vectorizer = TfidfVectorizer()
        course_vectors = vectorizer.fit_transform(course_names)
        course_index_map = {cid: idx for idx, cid in enumerate(course_ids)}
    else:
        vectorizer = None
        course_vectors = None
        course_index_map = {}

    # Train chỉ trên các bản ghi marked 'completed' và rating > 0
    train_df = df_local[(df_local["status"] == "completed") & (df_local["rating"] > 0)].copy()

    if len(train_df) < 10:
        print("⚠️ Chưa có đủ dữ liệu completed để train model NCF.")
        return df_local, None, {}, {}, vectorizer, course_index_map, course_vectors

    try:
        # Map string IDs sang int cho embeddings
        user_map = {uid: i for i, uid in enumerate(train_df['student_id'].unique())}
        item_map = {iid: i for i, iid in enumerate(train_df['course_id'].unique())}
        train_df['student_id_int'] = train_df['student_id'].map(user_map)
        train_df['course_id_int'] = train_df['course_id'].map(item_map)

        num_users = len(user_map)
        num_items = len(item_map)

        dataset = RatingDataset(train_df)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        model = NCF(num_users, num_items)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):  # Số epochs có thể điều chỉnh
            total_loss = 0
            for users, items, ratings in dataloader:
                # đảm bảo dtype đúng
                users = users.long()
                items = items.long()
                ratings = ratings.float()
                preds = model(users, items).squeeze()
                loss = criterion(preds, ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}: Avg Loss {total_loss / len(dataloader)}")

        print("✅ Đã train model NCF thành công.")
        return df_local, model, user_map, item_map, vectorizer, course_index_map, course_vectors
    except Exception as e:
        print("⚠️ Lỗi khi train NCF:", e)
        return df_local, None, {}, {}, vectorizer, course_index_map, course_vectors

# 🚀 KHỞI TẠO BIẾN TOÀN CỤC
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()
else:
    df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = pd.DataFrame(), None, {}, {}, None, {}, None

# 🏠 HOME ROUTE
@app.route("/")
def home():
    return "✅ Flask API - Gợi ý môn học với NCF Hybrid (Realtime Update)"

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

# Helper cho content score trong hybrid
def get_content_score(taken_courses, cat_courses, course_index_map, course_vectors):
    if course_vectors is None:
        return 0.0

    taken_indices = [course_index_map.get(c) for c in taken_courses if c in course_index_map]
    cat_indices = [course_index_map.get(c) for c in cat_courses if c in course_index_map]

    if not taken_indices or not cat_indices:
        return 0.0

    # Slice matrix và mean (hỗ trợ sparse)
    taken_vecs = course_vectors[taken_indices]
    cat_vecs = course_vectors[cat_indices]

    # ✅ Chuyển sang ndarray để tránh lỗi np.matrix
    taken_vec = np.asarray(taken_vecs.mean(axis=0)).ravel()
    cat_vec = np.asarray(cat_vecs.mean(axis=0)).ravel()

    # ✅ Reshape về dạng (1, -1) để dùng cosine_similarity
    sim = cosine_similarity(taken_vec.reshape(1, -1), cat_vec.reshape(1, -1))[0][0]

    return sim * 5  # Scale to 1-5


# 🎯 API: GỢI Ý MÔN HỌC
@app.route("/recommend/<student_id>/<faculty_id>", defaults={"year": None})
@app.route("/recommend/<student_id>/<faculty_id>/<year>")
def recommend(student_id, faculty_id, year):
    global df, model, user_map, item_map, vectorizer, course_index_map, course_vectors

    # Reload và retrain để đảm bảo mới nhất
    df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()

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
            df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()  # Reload sau khi thêm

    # map id -> name, year, semester
    # Loại bỏ trùng course_id, ưu tiên giữ bản ghi đầu tiên
    all_courses = all_courses.drop_duplicates(subset=["course_id"], keep="first")

    # Chuyển thành dict an toàn
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
            df = df.drop_duplicates(subset=["student_id", "course_id", "faculty_id"])  # Loại duplicate nếu có
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")
            df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()  # Retrain sau khi thêm
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
    avg_ratings = df[df["status"] == "completed"].groupby("course_id")["rating"].mean().to_dict()

    if is_new_student:
        # Sinh viên mới: Gợi ý TẤT CẢ courses trong 4 năm, sắp xếp theo year, semester, avg rating (nếu có)
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
        if model is None:
            # Fallback: dùng avg rating
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
            # Dùng NCF predict
            for cid in untaken_courses:
                if student_id in user_map and cid in item_map:
                    user_tensor = torch.tensor([user_map[student_id]], dtype=torch.long)
                    item_tensor = torch.tensor([item_map[cid]], dtype=torch.long)
                    model.eval()
                    with torch.no_grad():
                        est = model(user_tensor, item_tensor).item()
                else:
                    est = avg_ratings.get(cid, 3.0)  # Fallback trung bình
                details = course_details[cid]
                recommendations.append({
                    "course_id": cid,
                    "course_name": details["course_name"],
                    "year": details["year"],
                    "semester": details["semester"],
                    "predicted_rating": round(est, 3)
                })
            recommendations.sort(key=lambda x: -x["predicted_rating"])

    # Mở rộng: Suy đoán career paths và resources dựa trên hybrid NCF + Content-based
    category_scores = {}
    for cat, courses in CAREER_MAP.items():
        cat_rating = 0.0
        count = 0
        for c in courses:
            if c in taken_courses:
                # Dùng rating thực từ completed
                rating = df[(df["course_id"] == c) & (df["student_id"] == student_id) & (df["status"] == "completed")]["rating"].mean()
                if not pd.isna(rating) and rating > 0:
                    cat_rating += rating
                    count += 1
            elif model is not None and student_id in user_map and c in item_map:
                user_tensor = torch.tensor([user_map[student_id]], dtype=torch.long)
                item_tensor = torch.tensor([item_map[c]], dtype=torch.long)
                model.eval()
                with torch.no_grad():
                    pred_rating = model(user_tensor, item_tensor).item()
                cat_rating += pred_rating
                count += 1

        content_score = get_content_score(taken_courses, courses, course_index_map, course_vectors)
        if count > 0:
            cf_avg = cat_rating / count
        else:
            cf_avg = 0.0
        hybrid_score = 0.7 * cf_avg + 0.3 * content_score
        category_scores[cat] = hybrid_score

    # Sắp xếp categories theo score giảm dần
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

    career_paths = []
    recommended_resources = []
    # Lấy top 2 categories cao nhất, với threshold 3.5 để tránh bias nếu data ít
    for cat, score in sorted_categories[:2]:
        if score > 3.5:
            career_paths.extend(CAREER_SUGGESTIONS.get(cat, {}).get('paths', []))
            recommended_resources.extend(CAREER_SUGGESTIONS.get(cat, {}).get('resources', []))

    # Loại duplicate
    career_paths = list(set(career_paths))
    recommended_resources = [dict(t) for t in {tuple(d.items()) for d in recommended_resources}]

    # --- Ensure we always provide a usable list for frontend and compatibility ---
    # If recommendations empty (e.g. no ratings / model None), fallback to untaken ordered by year/semester
    if not recommendations:
        for cid in untaken_courses:
            details = course_details[cid]
            recommendations.append({
                "course_id": cid,
                "course_name": details["course_name"],
                "year": details["year"],
                "semester": details["semester"],
                "predicted_rating": round(avg_ratings.get(cid, 0.0), 3)
            })
        # sort by year/semester then by avg rating
        recommendations.sort(key=lambda x: (x["year"], x["semester"], -x["predicted_rating"]))

    # Build a simple list of names for backward-compatible frontends
    recommendation_names = [r["course_name"] for r in recommendations]

    # taken_details already prepared earlier (list of dicts). Build taken_names for compatibility.
    taken_names = [d["course_name"] for d in taken_details] if taken_details else []
    # If no completed taken courses found, show any existing rows for the student (including not_started) to display something
    if not taken_names:
        alt_taken = df[(df["student_id"] == student_id) & (df["faculty_id"] == faculty_id)]
        if not alt_taken.empty:
            # preserve order and uniqueness
            seen = set()
            for _, row in alt_taken.iterrows():
                name = row.get("course_name") or row.get("course_id")
                if pd.notna(name) and name not in seen:
                    seen.add(name)
            taken_names = list(seen)
            # also ensure taken_details contains minimal objects
            if not taken_details:
                taken_details = []
                for cid in alt_taken["course_id"].astype(str).unique().tolist():
                    taken_details.append({"course_id": cid, "course_name": course_details.get(cid, {"course_name": cid})["course_name"]})

    print(f"[DEBUG] student={student_id} faculty={faculty_id} year={year} taken_count={len(taken_names)} rec_count={len(recommendations)}")

    return jsonify({
        "student_id": student_id,
        "auto_year_detected": student_year,
        "start_year": start_year,
        "recommendations": recommendations,            # list of objects with metadata
        "recommendation_names": recommendation_names,  # backward-compatible list of strings
        "taken": taken_details,
        "taken_names": taken_names,                    # backward-compatible list of strings
        "career_paths": career_paths,
        "recommended_resources": recommended_resources
    })

# 🧾 API: CẬP NHẬT TRẠNG THÁI MÔN HỌC
@app.route("/update_status/<student_id>/<course_id>/<faculty_id>/<new_status>", methods=["POST"])
def update_status(student_id, course_id, faculty_id, new_status):
    print(f"Received POST request to /update_status/{student_id}/{course_id}/{faculty_id}/{new_status}")  # Log để debug
    global df, model, user_map, item_map, vectorizer, course_index_map, course_vectors

    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Không tìm thấy file dataset.csv"}), 404

    # Đọc file mới nhất với chuẩn hóa
    df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()

    # Chuẩn hóa input
    student_id_u = str(student_id).strip().upper().replace('.0', '')
    course_id_u = str(course_id).strip().upper().replace('.0', '')
    faculty_id_u = str(faculty_id).strip().upper()
    new_status_clean = str(new_status).lower().strip()

    # Nếu update thành "completed", yêu cầu rating từ request (nếu có)
    json_data = request.get_json(silent=True)
    new_rating = 0.0
    if json_data:
        new_rating = json_data.get("rating", 0.0)
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
    df, model, user_map, item_map, vectorizer, course_index_map, course_vectors = load_and_train()

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
    
    app.run(host='127.0.0.1', port=5000, debug=True)