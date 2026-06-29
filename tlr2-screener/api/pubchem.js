export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method !== 'GET') return res.status(405).end();

  const { name } = req.query;
  if (!name) return res.status(400).json({ error: 'name required' });

  const attempts = [
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/IsomericSMILES/JSON`,
    `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name.replace(/[^a-zA-Z0-9]/g, ''))}/property/IsomericSMILES/JSON`,
  ];

  for (const url of attempts) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      const d = await r.json();
      const smi = d.PropertyTable?.Properties?.[0]?.IsomericSMILES;
      if (smi) return res.status(200).json({ smiles: smi });
    } catch (_) {}
  }

  return res.status(404).json({ error: 'Not found' });
}
