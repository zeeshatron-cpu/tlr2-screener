import { checkQuota, incrementUsage } from './usage.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { smiles, name, mode } = req.body;
  if (!smiles) return res.status(400).json({ error: 'SMILES string required' });

  // Quota check
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

  const systemPrompt = mode === 'batch'
    ? `You are a computational chemistry expert specializing in TLR2 lipopeptide drug design. Analyze the SMILES and respond ONLY with a valid JSON object, no prose, no markdown fences. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string}. Base scores on real SAR: triacylated lipopeptides score highest (TLR1-TLR2), diacylated are TLR2/6, non-lipopeptides score low.`
    : `You are a computational chemistry expert specializing in TLR2 lipopeptide drug design. Given a SMILES string, analyze it as a potential TLR2 agonist based on the TLR1-TLR2 crystal structure (PDB: 2Z7X) and known lipopeptide SAR. Respond ONLY with a valid JSON object, no prose, no markdown fences. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string,"asn294_contact":boolean,"phe349_contact":boolean,"has_lipid_anchor":boolean,"flags":[{"type":"pass"|"warn"|"fail","text":string}],"analysis":string}. Base scores on real SAR: triacylated lipopeptides highest, diacylated mid, non-lipopeptides lowest. Be scientifically accurate.`;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-6',
        max_tokens: mode === 'batch' ? 400 : 1000,
        system: systemPrompt,
        messages: [{ role: 'user', content: `Name: ${name || 'Query molecule'}. SMILES: ${smiles}` }]
      })
    });

    const data = await response.json();
    if (!response.ok) return res.status(500).json({ error: data.error?.message || 'API error' });

    const raw = data.content?.[0]?.text || '{}';
    const result = JSON.parse(raw.replace(/```json|```/g, '').trim());

    // Count usage only on success
    try { await incrementUsage(req); } catch (_) {}

    return res.status(200).json(result);
  } catch (err) {
    return res.status(500).json({ error: 'Analysis failed: ' + err.message });
  }
}
