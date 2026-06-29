export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method !== 'GET') return res.status(405).end();

  const { name, debug } = req.query;
  if (!name) return res.status(400).json({ error: 'name required' });
  const errors = [];

  const urls = [
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/IsomericSMILES/JSON`,
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/CanonicalSMILES/JSON`,
  ];

  for (const url of urls) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': 'tlr2-screener/1.0' }, signal: AbortSignal.timeout(8000) });
      const body = await r.text();
      if (!r.ok) { errors.push(`${r.status} ${url}`); continue; }
      const d = JSON.parse(body);
      const props = d.PropertyTable?.Properties?.[0];
      const smi = props?.IsomericSMILES || props?.CanonicalSMILES;
      if (smi) return res.status(200).json({ smiles: smi });
      errors.push(`no SMILES in response: ${body.slice(0,200)}`);
    } catch (e) {
      errors.push(`fetch error: ${e.message}`);
    }
  }

  // Fallback: CID lookup then SMILES fetch
  try {
    const cidRes = await fetch(
      `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/cids/JSON`,
      { headers: { 'User-Agent': 'tlr2-screener/1.0' }, signal: AbortSignal.timeout(8000) }
    );
    const cidBody = await cidRes.text();
    if (cidRes.ok) {
      const cidData = JSON.parse(cidBody);
      const cid = cidData.IdentifierList?.CID?.[0];
      if (cid) {
        const smiRes = await fetch(
          `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/property/IsomericSMILES/JSON`,
          { headers: { 'User-Agent': 'tlr2-screener/1.0' }, signal: AbortSignal.timeout(8000) }
        );
        if (smiRes.ok) {
          const smiData = await smiRes.json();
          const smi = smiData.PropertyTable?.Properties?.[0]?.IsomericSMILES;
          if (smi) return res.status(200).json({ smiles: smi, cid });
        }
      }
    } else {
      errors.push(`CID lookup ${cidRes.status}: ${cidBody.slice(0,200)}`);
    }
  } catch (e) {
    errors.push(`CID fallback error: ${e.message}`);
  }

  return res.status(404).json({ error: 'Not found on PubChem', debug: errors });
}
