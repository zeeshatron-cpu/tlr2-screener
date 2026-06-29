import { checkQuota, incrementUsage } from './usage.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { smiles, name, mode, target } = req.body;
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

  const targetCtx = {
    'TLR1/2':   'TLR1-TLR2 heterodimer binding (PDB: 2Z7X). Key pharmacophore: three acyl chains, N-terminal cysteine lipopeptide. Reference agonist: Pam3CSK4.',
    'TLR2/6':   'TLR2-TLR6 heterodimer binding. Key pharmacophore: two acyl chains, MALP-2/FSL-1 scaffold. Reference agonist: FSL-1.',
    'TLR4':     'TLR4-MD2 complex binding (PDB: 3FXI). Key pharmacophore: lipid A bisphosphorylated diglucosamine with 4-6 acyl chains. Reference agonist: lipid A.',
    'general':  'general pattern recognition receptor (PRR) binding. Assess overall drug-likeness, lipophilicity, and any immune-activating structural features.',
  }[target || 'TLR1/2'];

  const systemPrompt = mode === 'batch'
    ? `You are a computational chemistry expert. Analyze this SMILES for ${targetCtx} Respond ONLY with valid JSON, no prose, no markdown. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string}.`
    : `You are a computational chemistry expert. Analyze this SMILES for ${targetCtx} Respond ONLY with valid JSON, no prose, no markdown. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string,"asn294_contact":boolean,"phe349_contact":boolean,"has_lipid_anchor":boolean,"flags":[{"type":"pass"|"warn"|"fail","text":string}],"analysis":string}. Be scientifically accurate. If the molecule is irrelevant to the selected target, say so clearly in analysis and give low scores.`;

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
        max_tokens: mode === 'batch' ? 400 : 1500,
        system: systemPrompt,
        messages: [{ role: 'user', content: `Name: ${name || 'Query molecule'}. SMILES: ${smiles}` }]
      })
    });

    const data = await response.json();
    if (!response.ok) return res.status(500).json({ error: data.error?.message || 'API error' });

    const raw = (data.content?.[0]?.text || '{}').replace(/```json|```/g, '').trim();
    let result;
    try {
      result = JSON.parse(raw);
    } catch (_) {
      // Truncated JSON — extract what we can with a forgiving parse
      const safe = raw.replace(/,\s*"[^"]*"\s*:\s*"[^"]*$/, '').replace(/,\s*"[^"]*":\s*$/, '') + '}';
      try { result = JSON.parse(safe); } catch(__) { result = JSON.parse(raw.slice(0, raw.lastIndexOf(',')) + '}'); }
    }

    // Count usage only on success
    try { await incrementUsage(req); } catch (_) {}

    return res.status(200).json(result);
  } catch (err) {
    return res.status(500).json({ error: 'Analysis failed: ' + err.message });
  }
}
