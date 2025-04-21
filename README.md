1. 프로젝트 개요

프로젝트명: AI 기반 부동산 시세 예측 및 매물 추천 플랫폼

목표: 빅데이터 및 AI를 활용하여 부동산 가격을 예측하고, 사용자 맞춤형 매물 추천 기능을 제공하는 웹/모바일 플랫폼 개발

주요 기능:
 
 부동산 시세 예측: 지역, 면적, 학군, 교통 접근성 등 다양한 데이터를 분석하여 시세 예측
 
 매물 추천: 사용자의 관심 지역 및 예산을 기반으로 최적의 매물 추천
 
 실시간 시장 동향 분석: 지역별 가격 변동률 및 트렌드 제공
 
 커뮤니티 기능: 사용자 간 거래 후기 공유 및 부동산 전문가 Q&A

2. 기술 스택

프론트엔드: React (Next.js), Tailwind CSS

백엔드: FastAPI, PostgreSQL, Firebase

데이터 분석 & AI: Python (Pandas, Scikit-learn, TensorFlow)

API 활용: 공공데이터포털(부동산 실거래가 API), 지도 API (Kakao, Google Maps)

CI/CD: GitHub Actions, Vercel

3. 개발 일정
   
## 개발 일정

| 주차  | 주요 작업                                  |
|------|--------------------------------|
| 1주차 | 프로젝트 기획 및 개발계획서 작성 , 기술 스택 확정 |
| 2주차 | UI/UX 디자인 및 와이어프레임 제작       |
| 3주차 | 데이터 수집 및 부동산 시세 예측 모델 개발  |
| 4주차 | 매물 추천 알고리즘 개발                |
| 5주차 | 백엔드 API 개발 및 DB 구축            |
| 6주차 | 프론트엔드 개발 및 연동               |
| 7주차 | 테스트 및 버그 수정                   |
| 8주차 | 최종 배포 및 발표 준비                |


4. 기대 효과

AI 기반 부동산 가격 예측을 통해 사용자에게 유용한 투자 정보 제공

맞춤형 매물 추천으로 사용자의 검색 시간을 단축하고 효율적인 거래 유도

실무에서 데이터 분석 및 웹 서비스 개발 경험을 쌓을 수 있음


5. 기타 사항

프로젝트 진행 중 일정 조정 가능

개발 과정에서 추가적인 요구사항이 생길 경우 반영


---

## 2주차 UI/UX 디자인 및 와이어프레임

### 2주차 작업 목표

- 사용자 경험(UX) 및 사용자 인터페이스(UI) 설계
- 와이어프레임(Wireframe) 제작
- 화면 기획 및 프로토타입 디자인
- 사용성 테스트 및 피드백 반영

### 요구사항 분석

- 타겟 사용자 분석 (예: 부동산 구매자, 임대자, 중개업체 등)
- 사용자 페르소나(Persona) 정의
- 주요 기능 및 페이지 구성 정리
- 사용자 흐름(UX Flow) 설계

### 2주차 목표 일정

| 날짜    | 작업 내용               |
| ----- | ------------------- |
| 1일차   | UI/UX 리서치 및 벤치마킹    |
| 2일차   | 주요 기능 정의 및 정보 구조 설계 |
| 3일차   | 와이어프레임 초안 제작        |
| 4일차   | 프로토타입 제작 (Figma 등)  |
| 5일차   | 팀원 피드백 반영 및 수정      |
| 6-7일차 | 최종 UI 디자인 완료 및 문서화  |

## AI Real Estate Platform - UI/UX 디자인 및 와이어프레임

### 1. 프로젝트 개요
AI 기반 부동산 시세 예측 및 매물 추천 플랫폼의 UI/UX 디자인을 진행합니다. 

주요 목표는 사용자가 쉽게 부동산 정보를 탐색하고 AI 기반 추천을 받을 수 있도록 직관적인 인터페이스를 구축하는 것입니다.

---

### 2. 주요 화면 설계

#### **홈 화면 (Home Screen)**
- 상단 검색 바 (지역, 가격대, 방 개수 등 필터 적용 가능)
- 추천 매물 섹션 (AI 추천 알고리즘 기반 맞춤형 매물 표시)
- 인기 지역 및 최신 등록 매물 리스트

#### **부동산 상세 페이지 (Property Details Page)**
- 매물 이미지 슬라이더
- 부동산 정보 (가격, 면적, 위치 등)
- AI 기반 시세 예측 및 비교 그래프
- 중개사 연락 버튼

#### **사용자 대시보드 (User Dashboard)**
- 저장한 매물 목록
- 개인 맞춤형 추천 매물 리스트
- 최근 검색 기록

#### **지도 검색 페이지 (Map Search Page)**
- 부동산 매물 지도 뷰
- 필터 옵션 (가격, 면적, 유형 등)
- AI 분석 기반 인기 지역 표시

---

## 3주차 부동산 시세 예측 모델 개발

### 부동산 시세 예측 모델 (기초 모델: 랜덤포레스트) 

"""
ai_real_estate_platform.py

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
