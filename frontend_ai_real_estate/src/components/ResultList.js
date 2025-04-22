import React from 'react';

function ResultList({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2>📋 추천 매물</h2>
      <table border="1" cellPadding="10">
        <thead>
          <tr>
            <th>아파트</th>
            <th>법정동</th>
            <th>전용면적</th>
            <th>층</th>
            <th>건축년도</th>
            <th>거래금액 (만원)</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, idx) => (
            <tr key={idx}>
              <td>{item.아파트}</td>
              <td>{item.법정동}</td>
              <td>{item.전용면적}</td>
              <td>{item.층}</td>
              <td>{item.건축년도}</td>
              <td>{item.거래금액}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ResultList;