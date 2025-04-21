from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from recommendation import recommend_listings  # 매물 추천 함수
import pandas as pd

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정

# 데이터베이스 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///real_estate.db'  # SQLite 예시 (PostgreSQL도 사용 가능)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 매물 데이터 모델 (SQLAlchemy)
class RealEstate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    apartment_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    area = db.Column(db.Float, nullable=False)
    floor = db.Column(db.Integer, nullable=False)
    construction_year = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<RealEstate {self.apartment_name}>"

# 기본 페이지
@app.route('/')
def home():
    return "AI Real Estate Platform API"

# 매물 추천 API
@app.route('/recommend', methods=['GET'])
def get_recommendations():
    budget = request.args.get('budget', type=int)
    location = request.args.get('location', type=str)
    min_area = request.args.get('min_area', type=float)

    # 데이터베이스에서 매물 불러오기
    listings = RealEstate.query.all()
    df = pd.DataFrame([(listing.apartment_name, listing.location, listing.area,
                        listing.floor, listing.construction_year, listing.price) 
                       for listing in listings],
                      columns=['아파트', '법정동', '전용면적', '층', '건축년도', '거래금액'])
    
    # 추천 리스트 생성
    recommendations = recommend_listings(df, budget, location, min_area)

    if recommendations.empty:
        return jsonify({"message": "조건에 맞는 매물이 없습니다."}), 404

    # 추천 결과 반환
    return jsonify(recommendations.to_dict(orient="records"))

# 부동산 시세 예측 API (기초 랜덤포레스트 모델)
@app.route('/predict', methods=['GET'])
def predict_price():
    # 간단한 예시 데이터 (추후 실제 모델로 교체)
    area = request.args.get('area', type=float)
    floor = request.args.get('floor', type=int)
    construction_year = request.args.get('construction_year', type=int)

    # 예측 모델 (랜덤포레스트 모델 필요)
    # 예: model.predict([area, floor, construction_year])
    predicted_price = area * 1.5 + floor * 200000 + (2025 - construction_year) * 100000

    return jsonify({"predicted_price": predicted_price})

# 데이터베이스 초기화
@app.route('/init_db', methods=['GET'])
def init_db():
    db.create_all()
    return jsonify({"message": "Database initialized!"})

if __name__ == '__main__':
    app.run(debug=True)
