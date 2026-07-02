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
    'TLR1/2':  'TLR1-TLR2 heterodimer binding (PDB: 2Z7X). Key pharmacophore: three acyl chains, N-terminal cysteine lipopeptide. Reference agonist: Pam3CSK4.',
    'TLR2/6':  'TLR2-TLR6 heterodimer binding. Key pharmacophore: two acyl chains, MALP-2/FSL-1 scaffold. Reference agonist: FSL-1.',
    'TLR3':    'TLR3 ectodomain binding (PDB: 3CIY). TLR3 recognizes dsRNA; small-molecule agonists mimic poly I:C. Key features: phosphate backbone mimicry, hydrogen-bond donors. Reference agonist: poly I:C.',
    'TLR4':    'TLR4-MD2 complex binding (PDB: 3FXI). Key pharmacophore: lipid A bisphosphorylated diglucosamine with 4-6 acyl chains. Reference agonist: lipid A.',
    'TLR5':    'TLR5 binding (PDB: 3J0A). Recognizes flagellin protein fragments. Key features: amphipathic helix, hydrophobic residue presentation. Reference agonist: CBLB502.',
    'TLR6':    'TLR6 in TLR2-TLR6 heterodimer context. Diacyl lipopeptides only (no third acyl chain). Reference agonist: Pam2CSK4.',
    'TLR7':    'TLR7 binding (PDB: 5GMH). Recognizes ssRNA and imidazoquinolines. Key features: H-bond acceptor at N1, hydrophobic C2/C4 substituents. Reference agonists: imiquimod, resiquimod.',
    'TLR8':    'TLR8 binding (PDB: 3W3J). Similar to TLR7 with distinct selectivity. Key features: uridine-like scaffold, Phe346/Tyr348 contacts. Reference agonist: VTX-2337.',
    'TLR9':    'TLR9 binding (PDB: 3WPB). Recognizes CpG DNA. Key pharmacophore: phosphodiester backbone, cytosine/guanine recognition. Reference agonist: CpG-ODN 2006.',
    'TLR10':   'TLR10 — poorly characterised ligand; forms heterodimer with TLR2. Assess similarity to diacyl/triacyl lipopeptide TLR2 agonist scaffolds.',
    'STING':   'STING CDN-binding domain (PDB: 4LOH). Recognizes cyclic dinucleotides (cGAMP, c-di-GMP). Key pharmacophore: two purine bases in hydrophobic cleft, phosphodiester H-bonds with Thr267/Arg238/Ser162. Reference agonists: cGAMP, DMXAA, diABZI.',
    'cGAS':    'cGAS catalytic pocket (PDB: 4KM5). Inhibitors target nucleotide-binding active site; activators are dsDNA. Key features: Mg2+ chelation, GTP/ATP substrate binding. Reference inhibitor: RU.521.',
    'RIG-I':   'RIG-I helicase/CTD (PDB: 4A36). Recognizes 5′-triphosphate dsRNA. Key features: 5′-triphosphate binding by Lys888/Lys858, dsRNA backbone contacts. Reference agonist: 5′ppp-dsRNA, KIN1148.',
    'MDA5':    'MDA5 helicase domain (PDB: 4GL2). Recognizes long dsRNA; forms cooperative filaments. Key features: helicase motif ATP-binding, dsRNA backbone contacts. Reference agonist: poly I:C (long form).',
    'general': 'general innate immune pattern recognition receptor (PRR) binding. Assess drug-likeness, lipophilicity, structural similarity to known PRR agonists, and any immune-activating features.',
  }[target || 'TLR1/2'];

  const scoreLabel = ['TLR1/2','TLR2/6','TLR4'].includes(target) ? 'tlr2_binding_score' : 'binding_score';

  const systemPrompt = mode === 'batch'
    ? `You are a computational chemistry expert. Analyze this SMILES for ${targetCtx} Respond ONLY with valid JSON, no prose, no markdown. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string}.`
    : `You are a computational chemistry expert. Analyze this SMILES for ${targetCtx} Respond ONLY with valid JSON, no prose, no markdown. Schema: {"molecule_name":string,"tlr2_binding_score":number(0-100),"pharmacophore_match":number(0-100),"mw_estimate":number,"logp_estimate":number,"lipopeptide_class":string,"verdict":"high"|"medium"|"low","verdict_text":string,"asn294_contact":boolean,"phe349_contact":boolean,"has_lipid_anchor":boolean,"flags":[{"type":"pass"|"warn"|"fail","text":string}],"analysis":string}. Be scientifically accurate. Use binding_score and pharmacophore_match to reflect the selected target receptor, not TLR2 specifically. If the molecule is irrelevant to the selected target, say so in analysis and give low scores.`;

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
