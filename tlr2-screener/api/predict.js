/**
 * /api/predict - proxy to Railway ML server.
 *
 * Set ML_ENDPOINT env var in Vercel to your Railway URL, e.g.:
 *   https://tlr2-ml.up.railway.app
 *
 * If ML_ENDPOINT is not set, returns 503 so the frontend falls back to LLM.
 */
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const endpoint = process.env.ML_ENDPOINT;
  if (!endpoint) {
    return res.status(503).json({
      error: 'ML endpoint not configured. Set ML_ENDPOINT in Vercel env vars.',
      unavailable: true,
    });
  }

  const { smiles, name } = req.body;
  if (!smiles) {
    return res.status(400).json({ error: 'SMILES required' });
  }

  try {
    const upstream = await fetch(`${endpoint.replace(/\/$/, '')}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, name: name || 'Query' }),
      signal: AbortSignal.timeout(15000),
    });

    const data = await upstream.json();

    if (!upstream.ok) {
      return res.status(502).json({ error: data.detail || 'ML server error', unavailable: false });
    }

    return res.status(200).json(data);
  } catch (err) {
    return res.status(503).json({ error: 'ML server unreachable: ' + err.message, unavailable: true });
  }
}
