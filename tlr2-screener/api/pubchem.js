export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method !== 'GET') return res.status(405).end();

  const { name } = req.query;
  if (!name) return res.status(400).json({ error: 'name required' });

  const urls = [
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/IsomericSMILES/JSON`,
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/CanonicalSMILES/JSON`,
  ];

  for (const url of urls) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': 'tlr2-screener/1.0' } });
      if (!r.ok) continue;
      const d = await r.json();
      const props = d.PropertyTable?.Properties?.[0];
      const smi = props?.IsomericSMILES || props?.CanonicalSMILES;
      if (smi) return res.status(200).json({ smiles: smi });
    } catch (e) {
      console.error('PubChem fetch error:', e.message);
    }
  }

  // Fallback: try PubChem CID lookup then SMILES fetch
  try {
    const cidRes = await fetch(
      `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/cids/JSON`,
      { headers: { 'User-Agent': 'tlr2-screener/1.0' } }
    );
    if (cidRes.ok) {
      const cidData = await cidRes.json();
      const cid = cidData.IdentifierList?.CID?.[0];
      if (cid) {
        const smiRes = await fetch(
          `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/property/IsomericSMILES/JSON`,
          { headers: { 'User-Agent': 'tlr2-screener/1.0' } }
        );
        if (smiRes.ok) {
          const smiData = await smiRes.json();
          const smi = smiData.PropertyTable?.Properties?.[0]?.IsomericSMILES;
          if (smi) return res.status(200).json({ smiles: smi, cid });
        }
      }
    }
  } catch (e) {
    console.error('PubChem CID fallback error:', e.message);
  }

  return res.status(404).json({ error: 'Not found on PubChem' });
}
