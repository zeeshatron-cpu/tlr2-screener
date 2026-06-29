import { checkQuota, incrementUsage } from './usage.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const endpoint = process.env.ML_ENDPOINT;
  if (!endpoint) return res.status(503).json({ error: 'ML endpoint not configured.', unavailable: true });

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

  try {
    const upstream = await fetch(`${endpoint.replace(/\/$/, '')}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, name: name || 'Query' }),
      signal: AbortSignal.timeout(15000),
    });
    const data = await upstream.json();
    if (!upstream.ok) return res.status(502).json({ error: data.detail || 'ML server error', unavailable: false });

    try { await incrementUsage(req); } catch (_) {}

    return res.status(200).json(data);
  } catch (err) {
    return res.status(503).json({ error: 'ML server unreachable: ' + err.message, unavailable: true });
  }
}
