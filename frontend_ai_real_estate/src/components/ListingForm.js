import React, { useState } from 'react';
import { getRecommendations } from '../services/api';

function ListingForm({ onResults }) {
  const [budget, setBudget] = useState('');
  const [location, setLocation] = useState('');
  const [minArea, setMinArea] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = await getRecommendations(budget, location, minArea);
    onResults(data);
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '1rem' }}>
      <input type="number" placeholder="예산 (만원)" value={budget} onChange={e => setBudget(e.target.value)} required />
      <input type="text" placeholder="지역 (예: 창신동)" value={location} onChange={e => setLocation(e.target.value)} />
      <input type="number" placeholder="최소 면적 (㎡)" value={minArea} onChange={e => setMinArea(e.target.value)} />
      <button type="submit">추천받기</button>
    </form>
  );
}

export default ListingForm;