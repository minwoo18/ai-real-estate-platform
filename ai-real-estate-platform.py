"""
real_estate_price_predictor.py

부동산 실거래가 데이터를 기반으로 시세를 예측하는 랜덤 포레스트 모델입니다.
데이터는 국토교통부 실거래가 공개 API에서 수집하며, 예측 모델은 scikit-learn을 사용합니다.

작성자: [김민우]
작성일: 2025-03-26
"""

import requests
import pandas as pd
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# === 설정 ===
SERVICE_KEY = "여기에_본인_API_인증키_입력"  # https://www.data.go.kr 에서 발급
REGION_CODE = "11110"   # 서울 종로구 (법정동 코드)
DEAL_YMD = "202403"     # 거래연월: 2024년 3월

def fetch_data(service_key, region_code, deal_ymd):
    """국토부 API에서 아파트 실거래가 데이터 수집"""
    url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    params = {
        "serviceKey": service_key,
        "LAWD_CD": region_code,
        "DEAL_YMD": deal_ymd,
        "numOfRows": 1000,
        "pageNo": 1
    }

    response = requests.get(url, params=params)
    tree = ET.fromstring(response.content)
    items = tree.findall(".//item")

    rows = []
    for item in items:
        try:
            row = {
                "거래금액": int(item.findtext("거래금액").strip().replace(",", "")),
                "건축년도": int(item.findtext("건축년도")),
                "전용면적": float(item.findtext("전용면적")),
                "층": int(item.findtext("층")),
                "법정동": item.findtext("법정동"),
                "아파트": item.findtext("아파트")
            }
            rows.append(row)
        except:
            continue  # 오류나는 항목은 건너뜀

    return pd.DataFrame(rows)

def preprocess_data(df):
    """데이터 전처리 및 인코딩"""
    le_dong = LabelEncoder()
    le_apt = LabelEncoder()

    df["법정동"] = le_dong.fit_transform(df["법정동"])
    df["아파트"] = le_apt.fit_transform(df["아파트"])

    X = df[["건축년도", "전용면적", "층", "법정동", "아파트"]]
    y = df["거래금액"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """랜덤포레스트 회귀 모델 학습"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """모델 성능 평가 및 시각화"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"평균 절대 오차 (MAE): {mae:,.0f} 원")

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("실제 거래금액")
    plt.ylabel("예측 거래금액")
    plt.title("실제 vs 예측 거래금액")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = fetch_data(SERVICE_KEY, REGION_CODE, DEAL_YMD)

    if df.empty:
        print("데이터를 불러오지 못했습니다. API 키나 지역/날짜 설정을 확인하세요.")
        return

    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
