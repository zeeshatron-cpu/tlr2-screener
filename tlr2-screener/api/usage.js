import { kv } from '@vercel/kv';

const FREE_LIMIT = 20;

function getMonthKey(id) {
  const d = new Date();
  return `usage:${id}:${d.getUTCFullYear()}-${d.getUTCMonth()}`;
}

// Returns { allowed, used, limit, tier }
export async function checkQuota(req) {
  const apiKey = req.headers['x-api-key'] || req.body?.api_key;

  if (apiKey) {
    const meta = await kv.get(`key:${apiKey}`);
    if (!meta) return { allowed: false, used: 0, limit: 0, tier: 'invalid', error: 'Invalid API key' };
    const monthKey = getMonthKey(`key:${apiKey}`);
    const used = (await kv.get(monthKey)) || 0;
    const limit = meta.limit ?? 500;
    return { allowed: used < limit, used, limit, tier: meta.tier || 'lab' };
  }

  // IP-based free tier
  const ip = req.headers['x-forwarded-for']?.split(',')[0].trim() || req.socket?.remoteAddress || 'unknown';
  const monthKey = getMonthKey(`ip:${ip}`);
  const used = (await kv.get(monthKey)) || 0;
  return { allowed: used < FREE_LIMIT, used, limit: FREE_LIMIT, tier: 'free' };
}

export async function incrementUsage(req) {
  const apiKey = req.headers['x-api-key'] || req.body?.api_key;
  const id = apiKey ? `key:${apiKey}` : (() => {
    const ip = req.headers['x-forwarded-for']?.split(',')[0].trim() || req.socket?.remoteAddress || 'unknown';
    return `ip:${ip}`;
  })();
  const monthKey = getMonthKey(id);
  await kv.incr(monthKey);
  await kv.expire(monthKey, 60 * 60 * 24 * 40); // 40 days TTL
}

export default async function handler(req, res) {
  if (req.method !== 'GET' && req.method !== 'POST') return res.status(405).end();
  try {
    const quota = await checkQuota(req);
    return res.status(200).json({ used: quota.used, limit: quota.limit, tier: quota.tier });
  } catch (e) {
    // KV not configured — return unlimited so the app still works
    return res.status(200).json({ used: 0, limit: null, tier: 'free', kv_unavailable: true });
  }
}
