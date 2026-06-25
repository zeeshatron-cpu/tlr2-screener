"""
/api/predict stub for Vercel deployment.

RDKit (~300MB uncompressed) exceeds Vercel's serverless function size limit.
The full ML pipeline runs locally via ml/run_pipeline.sh.

To deploy the ML endpoint: host predict_server.py on Railway, Render,
or Hugging Face Spaces (all support rdkit), then point the frontend here.
"""

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({
            "error": "ML endpoint not available on Vercel (rdkit size limit). Using LLM fallback.",
            "source": "ml_model",
            "unavailable": True,
        }).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"status": "unavailable", "reason": "rdkit too large for Vercel serverless"}).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
