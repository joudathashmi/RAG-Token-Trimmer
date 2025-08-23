#!/usr/bin/env python3
# single_app.py — RAG + prompt budgeting in one file

import os, json, re, argparse
import numpy as np
import faiss
import requests
from typing import List, Tuple
from flask import Flask, render_template, request, jsonify
import threading
import webbrowser
import time

# Load environment variables from .env file
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass  # .env file doesn't exist, continue without it

# Load environment variables
load_env()

# Global variables for storing prompts and outputs
prompt_history = []
output_history = []

# -----------------------------
# 0) Sample policy texts (you can replace with your own)
# -----------------------------
PDPL_SAMPLE = """PDPL §4 – Cross-Border Personal Data Transfers
Personal data shall not be transferred outside the Kingdom unless permitted by applicable regulations. Controllers must ensure adequate protection, obtain approvals where required, and minimize data shared to what is strictly necessary for the stated purpose.

PDPL §9 – Data Minimization
Controllers shall only process personal data that is necessary, relevant, and limited to the purposes for which it is collected.
"""

VISION2030_SAMPLE = """Vision2030 §12 – Localization
Industrial projects should target local value creation. A recommended threshold is at least 40% of total manufacturing value within KSA, alongside workforce development and supplier enablement.

Vision2030 §18 – Innovation and Quality
Projects should demonstrate adoption of modern technologies and quality systems that enhance competitiveness and export readiness.
"""

# -----------------------------
# 1) Chunking (token-ish by chars)
# -----------------------------
def char_budget_from_tokens(tokens: int) -> int:
    # Rough conversion: ≈ 1 token ~ 4 chars
    return int(tokens * 4)

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def split_into_chunks(text: str, size_tokens: int = 400, overlap_tokens: int = 60) -> List[Tuple[int, str]]:
    max_chars = char_budget_from_tokens(size_tokens)
    overlap_chars = char_budget_from_tokens(overlap_tokens)
    t = clean_text(text)
    out, i, section = [], 0, 0
    while i < len(t):
        chunk = t[i:i+max_chars]
        if not chunk:
            break
        out.append((section, chunk))
        i += max_chars - overlap_chars
        section += 1
    return out

# -----------------------------
# 2) Simple store + vector index
# -----------------------------
class DocumentStore:
    def __init__(self):
        self.chunks = []  # list of (doc_id, section, text)

    def add_document(self, doc_id: str, text: str, size_tokens=400, overlap_tokens=60):
        for sec, chunk in split_into_chunks(text, size_tokens, overlap_tokens):
            self.chunks.append((doc_id, sec, chunk))

class VectorIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # downloads a small sentence-transformer the first time
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []  # [(doc_id, section)]
        self.emb = None

    def build(self, chunks: List[Tuple[str,int,str]]):
        texts = [c[2] for c in chunks]
        self.emb = np.array(self.model.encode(texts, normalize_embeddings=True), dtype="float32")
        dim = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.emb)
        self.meta = [(c[0], c[1]) for c in chunks]

    def search(self, query: str, k: int = 8) -> List[int]:
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k)
        return I[0].tolist()

# -----------------------------
# 3) Retrieval (top-8 -> keep 2–3)
# -----------------------------
def retrieve_minimal(query: str, store: DocumentStore, vindex: VectorIndex, k=8, keep=3, cap_tokens=380):
    idxs = vindex.search(query, k=k)
    # No reranker here to keep deps light; keep first 'keep'
    selected = []
    for i in idxs[:keep]:
        if i < 0 or i >= len(store.chunks): 
            continue
        doc_id, section, text = store.chunks[i]
        # cap each chunk to ~cap_tokens (by chars)
        max_chars = char_budget_from_tokens(cap_tokens)
        selected.append((doc_id, section, text[:max_chars]))
    return selected

# -----------------------------
# 4) Prompt discipline + schema
# -----------------------------
SYSTEM_RULES = (
    "You are an evaluator for Saudi government opportunities. "
    "Return only valid JSON with keys: ok, criteria, gaps, reason. "
    "No prose outside JSON. Keep reason ≤ 25 words. Use only provided context."
)

def build_user_prompt(inquiry: str, ctx_chunks: List[Tuple[str,int,str]]) -> str:
    # Context lines like "- PDPL §4: ..."
    ctx_lines = []
    for doc_id, section, text in ctx_chunks:
        ctx_lines.append(f"- {doc_id} §{section}: {text}")
    context = "\n".join(ctx_lines)
    return (
        "Task: Evaluate the inquiry.\n"
        f"Inquiry: {inquiry}\n"
        "Context:\n"
        f"{context}\n"
        "Return JSON only: {\"ok\":bool,\"criteria\":[],\"gaps\":[],\"reason\":\"≤25w\"}"
    )

def validate_json_output(s: str):
    try:
        data = json.loads(s)
        for k in ("ok","criteria","gaps","reason"):
            if k not in data:
                return False, {"error": f"missing key {k}"}
        if not isinstance(data["ok"], bool): return False, {"error":"ok must be bool"}
        if not isinstance(data["criteria"], list): return False, {"error":"criteria must be list"}
        if not isinstance(data["gaps"], list): return False, {"error":"gaps must be list"}
        if not isinstance(data["reason"], str): return False, {"error":"reason must be string"}
        words = data["reason"].split()
        if len(words) > 25:
            data["reason"] = " ".join(words[:25])
        return True, data
    except Exception as e:
        return False, {"error": str(e)}

# -----------------------------
# 5) LLM call (OpenAI or on-prem)
# -----------------------------
def llm_call(prompt: str, system: str = SYSTEM_RULES, model="gpt-4o-mini",
             base_url="https://api.openai.com/v1", api_key=None, max_tokens=200):
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    
    # Store prompt for web interface
    prompt_data = {
        "system": system,
        "user": prompt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    prompt_history.append(prompt_data)
    
    # Demo mode if no API key is provided
    if not api_key or api_key == "your-actual-api-key-here":
        print("🔧 DEMO MODE: Using mock response (no API key provided)")
        print(f"📝 System prompt: {system[:100]}...")
        print(f"📝 User prompt: {prompt[:200]}...")
        
        # Mock response based on the inquiry
        if "medical device" in prompt.lower() and "export data" in prompt.lower():
            response = '''{"ok": false, "criteria": ["data_minimization", "cross_border_transfer"], "gaps": ["insufficient_local_workforce", "data_export_concerns"], "reason": "Fails PDPL data transfer and workforce localization requirements"}'''
        else:
            response = '''{"ok": true, "criteria": ["localization", "innovation"], "gaps": [], "reason": "Meets Vision 2030 requirements for local value creation"}'''
        
        # Store output for web interface
        output_history.append({
            "response": response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "demo"
        })
        return response
    
    # Real API call
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    response = r.json()["choices"][0]["message"]["content"].strip()
    
    # Store output for web interface
    output_history.append({
        "response": response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "api"
    })
    
    return response

# -----------------------------
# 6) End-to-end evaluate
# -----------------------------
def evaluate(inquiry: str, store: DocumentStore, vindex: VectorIndex,
             model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key=None):
    ctx = retrieve_minimal(inquiry, store, vindex, k=8, keep=3, cap_tokens=360)
    prompt = build_user_prompt(inquiry, ctx)
    raw = llm_call(prompt, model=model, base_url=base_url, api_key=api_key, max_tokens=200)
    ok, data = validate_json_output(raw)
    if not ok:
        # one retry with extra reminder
        raw = llm_call(prompt + "\nReturn JSON only.", model=model, base_url=base_url, api_key=api_key, max_tokens=200)
        ok, data = validate_json_output(raw)
        if not ok:
            return {"ok": False, "criteria": [], "gaps": ["invalid_output"], "reason": "Model did not return valid JSON", "raw": raw}
    # add traceability of which chunks were used
    data["context_refs"] = [{"doc_id": d, "section": s} for d, s, _ in ctx]
    return data

# -----------------------------
# 7) CLI
# -----------------------------
def build_default_store() -> Tuple[DocumentStore, VectorIndex]:
    store = DocumentStore()
    store.add_document("PDPL", PDPL_SAMPLE)
    store.add_document("Vision2030", VISION2030_SAMPLE)
    vindex = VectorIndex()
    vindex.build(store.chunks)
    return store, vindex

def main():
    ap = argparse.ArgumentParser(description="RAG + Prompt Budget demo (single file)")
    ap.add_argument("--inquiry", default="New AI-enabled medical device factory in Riyadh with 35% local workforce and plans to export data abroad.")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--web", action="store_true", help="Start web interface")
    args = ap.parse_args()

    if args.web:
        start_web_interface()
    else:
        store, vindex = build_default_store()
        result = evaluate(args.inquiry, store, vindex, model=args.model, base_url=args.base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))

# -----------------------------
# 8) Web Interface
# -----------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_web():
    data = request.get_json()
    inquiry = data.get('inquiry', '')
    
    store, vindex = build_default_store()
    result = evaluate(inquiry, store, vindex)
    
    return jsonify({
        'result': result,
        'prompts': prompt_history[-1] if prompt_history else None,
        'outputs': output_history[-1] if output_history else None
    })

@app.route('/history')
def get_history():
    return jsonify({
        'prompts': prompt_history,
        'outputs': output_history
    })

def start_web_interface():
    # Create templates directory and HTML file
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Saudi Government Evaluator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .input-section {
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 30px;
        }
        .result-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }
        .prompt-box {
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
        .output-box {
            background: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .status.info { background: #d1ecf1; color: #0c5460; }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇸🇦 Saudi Government Opportunity Evaluator</h1>
        
        <div class="input-section">
            <h3>Enter your inquiry:</h3>
            <textarea id="inquiry" placeholder="Describe the government opportunity or project you want to evaluate...">New AI-enabled medical device factory in Riyadh with 35% local workforce and plans to export data abroad.</textarea>
            <button onclick="evaluateInquiry()">Evaluate</button>
        </div>

        <div class="loading" id="loading">
            <p>🔍 Analyzing your inquiry...</p>
        </div>

        <div class="results" id="results" style="display: none;">
            <h3>📊 Evaluation Results</h3>
            <div id="result-content"></div>
            
            <h3>📝 System Prompt</h3>
            <div id="prompt-content"></div>
            
            <h3>🤖 AI Response</h3>
            <div id="output-content"></div>
        </div>
    </div>

    <script>
        async function evaluateInquiry() {
            const inquiry = document.getElementById('inquiry').value;
            if (!inquiry.trim()) {
                alert('Please enter an inquiry to evaluate.');
                return;
            }

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            try {
                const response = await fetch('/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ inquiry: inquiry })
                });

                const data = await response.json();
                
                // Display results
                document.getElementById('result-content').innerHTML = `
                    <div class="result-box">
                        <h4>Status: ${data.result.ok ? '✅ Approved' : '❌ Rejected'}</h4>
                        <p><strong>Reason:</strong> ${data.result.reason}</p>
                        <p><strong>Criteria Met:</strong> ${data.result.criteria.join(', ') || 'None'}</p>
                        <p><strong>Gaps Identified:</strong> ${data.result.gaps.join(', ') || 'None'}</p>
                        <pre>${JSON.stringify(data.result, null, 2)}</pre>
                    </div>
                `;

                // Display prompt
                if (data.prompts) {
                    document.getElementById('prompt-content').innerHTML = `
                        <div class="prompt-box">
                            <p><strong>System Prompt:</strong></p>
                            <pre>${data.prompts.system}</pre>
                            <p><strong>User Prompt:</strong></p>
                            <pre>${data.prompts.user}</pre>
                            <p><strong>Timestamp:</strong> ${data.prompts.timestamp}</p>
                        </div>
                    `;
                }

                // Display output
                if (data.outputs) {
                    document.getElementById('output-content').innerHTML = `
                        <div class="output-box">
                            <p><strong>Mode:</strong> ${data.outputs.mode}</p>
                            <p><strong>Response:</strong></p>
                            <pre>${data.outputs.response}</pre>
                            <p><strong>Timestamp:</strong> ${data.outputs.timestamp}</p>
                        </div>
                    `;
                }

                document.getElementById('results').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('Error evaluating inquiry. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        // Auto-evaluate on page load
        window.onload = function() {
            evaluateInquiry();
        };
    </script>
</body>
</html>
    '''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("🌐 Starting web interface...")
    print("📱 Open your browser and go to: http://localhost:8080")
    print("🔧 Press Ctrl+C to stop the server")
    
    # Open browser automatically
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:8080')
    
    threading.Thread(target=open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=8080)

if __name__ == "__main__":
    main()
