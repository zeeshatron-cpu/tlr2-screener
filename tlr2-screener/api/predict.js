import { checkQuota, incrementUsage } from './usage.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { smiles, name } = req.body;
  if (!smiles) return res.status(400).json({ error: 'SMILES required' });

  // Quota check — shared counter with analyze.js
  let quota;
  try {
    quota = await checkQuota(req);
  } catch (_) {
    quota = { allowed: true, kv_unavailable: true };
  }
  if (!quota.allowed) {
    return res.status(429).json({
      error: quota.error || `Monthly limit reached (${quota.limit} queries). Add an API key to continue.`,
      quota_exceeded: true,
      used: quota.used,
      limit: quota.limit,
      tier: quota.tier,
    });
  }

  const externalEndpoint = process.env.ML_ENDPOINT;

  // If an external ML server is configured, proxy to it
  if (externalEndpoint) {
    try {
      const upstream = await fetch(`${externalEndpoint.replace(/\/$/, '')}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles, name: name || 'Query' }),
        signal: AbortSignal.timeout(15000),
      });
      const data = await upstream.json();
      if (!upstream.ok) return res.status(502).json({ error: data.detail || 'ML server error' });
      try { await incrementUsage(req); } catch (_) {}
      return res.status(200).json(data);
    } catch (err) {
      return res.status(503).json({ error: 'ML server unreachable: ' + err.message, unavailable: true });
    }
  }

  // Otherwise call the built-in Python serverless function
  const proto = req.headers['x-forwarded-proto'] || 'https';
  const host = req.headers['x-forwarded-host'] || req.headers.host;
  const localUrl = `${proto}://${host}/api/predict_tlr2`;

  try {
    const upstream = await fetch(localUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, name: name || 'Query' }),
      signal: AbortSignal.timeout(30000),
    });
    const data = await upstream.json();
    if (!upstream.ok) return res.status(upstream.status).json(data);
    try { await incrementUsage(req); } catch (_) {}
    return res.status(200).json(data);
  } catch (err) {
    return res.status(503).json({ error: 'Prediction unavailable: ' + err.message, unavailable: true });
  }
}
