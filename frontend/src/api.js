import axios from 'axios';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const client = axios.create({
  baseURL: BASE,
  timeout: 300_000,   // 5 min — GEE + inference can be slow
  headers: { 'Content-Type': 'application/json' },
});

export const api = {
  health:  ()                    => client.get('/health'),
  regions: ()                    => client.get('/regions'),
  analyze: (region, year1, year2) =>
    client.post('/analyze', { region, year1, year2 }),
  report:  (filename)            => `${BASE}/report/${filename}`,
};
