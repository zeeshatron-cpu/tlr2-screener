# TLR2 Screener

AI-powered TLR2 lipopeptide binding prediction grounded in PDB:2Z7X crystal structure data.

## What it does
- Single compound analysis: binding score, pharmacophore match, Asn294/Phe349 contacts, drug-likeness flags
- Batch CSV upload: screen up to 20 compounds, download results as CSV
- Pricing page with free/lab/biotech tiers

## Deploy to Vercel (takes ~10 minutes)

### 1. Get an Anthropic API key
- Go to console.anthropic.com
- Create an API key
- Copy it

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tlr2-screener.git
git push -u origin main
```

### 3. Deploy on Vercel
- Go to vercel.com and sign in with GitHub
- Click "Add New Project"
- Import your tlr2-screener repo
- Under "Environment Variables", add:
  - Key: `ANTHROPIC_API_KEY`
  - Value: your API key from step 1
- Click Deploy

That's it. Vercel gives you a live URL instantly.

### 4. Custom domain (optional)
- Buy `tlr2screen.com` or similar on Namecheap (~$10/year)
- In Vercel project settings → Domains → add your domain
- Follow the DNS instructions

## Monetization
- Update the contact email in the pricing page (`your@email.com`)
- Add your bioRxiv paper DOI to the paper link on the landing page
- For payment processing: add Stripe when you're ready

## File structure
```
tlr2-screener/
  index.html        ← full frontend (single file)
  api/
    analyze.js      ← serverless function (API key lives here)
  vercel.json       ← routing config
  README.md
```

## Built by
Ahmed Zeeshan · Interlake High School · Bellevue, WA
Scientific basis: Pam3Cys-SNFKK paper (add DOI once published)
