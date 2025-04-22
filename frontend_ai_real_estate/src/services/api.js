export async function getRecommendations(budget, location, minArea) {
  const params = new URLSearchParams({
    budget,
    location,
    min_area: minArea,
  });

  try {
    const res = await fetch(`http://localhost:5000/recommend?${params.toString()}`);
    if (!res.ok) throw new Error('추천 실패');
    return await res.json();
  } catch (err) {
    console.error(err);
    return [];
  }
}