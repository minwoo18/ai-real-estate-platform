import pandas as pd

def recommend_listings(df, budget, location=None, min_area=None, top_k=5):
    """
    사용자의 조건에 맞는 매물 추천
    df: DataFrame (매물 데이터)
    budget: 최대 예산 (만원)
    location: 선호 지역 (법정동 이름)
    min_area: 최소 전용면적 (㎡)
    top_k: 추천할 매물 수
    """

    # === 1. 기본 필터링 ===
    filtered = df[df["거래금액"] <= budget]

    if location:
        filtered = filtered[filtered["법정동"].str.contains(location)]

    if min_area:
        filtered = filtered[filtered["전용면적"] >= min_area]

    if filtered.empty:
        print("❌ 조건에 맞는 매물이 없습니다.")
        return pd.DataFrame()

    # === 2. 단순 선호 기준 기반 정렬 ===
    # 점수: (면적 클수록 +, 층 높을수록 +, 건축년도 최근일수록 +)
    filtered = filtered.copy()
    filtered["점수"] = (
        (filtered["전용면적"] / filtered["전용면적"].max()) * 0.4 +
        (filtered["층"] / filtered["층"].max()) * 0.3 +
        (filtered["건축년도"] / filtered["건축년도"].max()) * 0.3
    )

    recommendations = filtered.sort_values(by="점수", ascending=False).head(top_k)

    print(f"🔍 추천 매물 {len(recommendations)}건:")
    return recommendations[["아파트", "법정동", "전용면적", "층", "건축년도", "거래금액"]]
