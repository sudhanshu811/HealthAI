# HealthAI — Clinical Intelligence Chatbot

A production-grade healthcare AI platform powered by **Mistral LLM** and **LlamaIndex**, unifying 6 specialized pipelines behind one chatbot interface.

---

## Architecture

```
User (Browser)
    │
    ▼
Frontend (index.html) — static, served by FastAPI
    │
    ▼  POST /chat  or  POST /chat/image
FastAPI Orchestrator  ←──── auto-classifies query mode
    │
    ├─── basic_rag        →  Basic RAG over 10 WHO health docs
    ├─── multi_doc_agent  →  Multi-doc ReAct agent (5 diseases)
    ├─── react_agent      →  Clinical calculators + RAG
    ├─── router           →  RouterQueryEngine (guidelines)
    ├─── subquestion      →  SubQuestion hospital comparator
    └─── multimodal       →  Pixtral vision (drug labels)
         │
         ▼
    Mistral API  (mistral-small-latest / pixtral-12b)
         │
         ▼
    In-memory LRU cache (reduces API calls)
```

---

## Quick Start (local)

```bash
# 1. Clone / unzip this folder
cd healthai

# 2. Create virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Set your Mistral API key
export MISTRAL_API_KEY="your_key_here"
# Windows: set MISTRAL_API_KEY=your_key_here

# 5. Run the server
uvicorn backend.main:app --reload --port 8000

# 6. Open browser
open http://localhost:8000
```

---

## Free Deployment on Render.com

Render offers a **free tier** for web services (750 hrs/month, sleeps after 15 min inactivity).

### Step-by-step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "HealthAI initial commit"
   # create a repo on github.com, then:
   git remote add origin https://github.com/YOUR_USERNAME/healthai.git
   git push -u origin main
   ```

2. **Create account** at [render.com](https://render.com) (free)

3. **New Web Service**
   - Connect your GitHub repo
   - Runtime: **Python 3**
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

4. **Set environment variable**
   - Key: `MISTRAL_API_KEY`
   - Value: your key from [console.mistral.ai](https://console.mistral.ai)

5. **Deploy** — Render builds and deploys automatically. Your URL:
   `https://healthai-xxxx.onrender.com`

> ⚠️ **Important**: The HuggingFace embedding models (~440 MB) are downloaded on first build. Free Render instances have 512 MB RAM — if you hit memory limits, switch to `BAAI/bge-small-en-v1.5` in `main.py`.

---

## Alternative: Deploy on Railway.app

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Add env var `MISTRAL_API_KEY`
3. Railway auto-detects the `Dockerfile` and deploys

---

## Alternative: Deploy on Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. SDK: **Docker**
3. Upload all files
4. Add secret `MISTRAL_API_KEY` in Space Settings

---

## Adding Your Data Files

Drop your `.txt` files from the notebooks into the correct folders:

| Folder | Pipeline | Files from |
|--------|----------|-----------|
| `data/health/` | Basic RAG | Notebook 1 health docs |
| `data/disease_outbreaks/` | Multi-doc Agent | Notebook 2 disease .txt files |
| `data/hospital_reports/` | SubQuestion | Notebook 6 hospital reports |
| `data/clinical_guidelines/` | Router + ReAct | Notebook 5 guidelines |
| `data/drug_labels/` | Multimodal | Notebook 3 images (optional) |

> The app ships with sample data so it works immediately. Replace with your own richer data.

---

## Pipelines Reference

| Pipeline | Auto-triggered by | Data source |
|----------|-------------------|-------------|
| `basic_rag` | General health questions | `data/health/` |
| `multi_doc_agent` | Disease names, outbreaks, epidemic | `data/disease_outbreaks/` |
| `react_agent` | BMI, BP, dosage, eGFR, calculate | `data/clinical_guidelines/` + built-in tools |
| `router` | Guidelines, protocol, management | `data/clinical_guidelines/` |
| `subquestion` | Hospital comparison, readmission, mortality | `data/hospital_reports/` |
| `multimodal` | Image upload (drug labels) | Pixtral vision API |

---

## API Endpoints

```
POST /chat
Body: { "query": "...", "mode": "auto" }
Response: { "answer": "...", "mode": "...", "latency_ms": 1200, "cached": false }

POST /chat/image
Form: file=<image>, prompt="Analyze..."
Response: { "answer": "...", "mode": "multimodal", ... }

GET /health
Response: { "status": "ok", "loaded_pipelines": [...] }
```

---

## Memory & Performance Tips

- First query to each pipeline takes 30–60 seconds (index building + model download)
- Subsequent queries: 2–8 seconds
- Caching: identical queries return instantly
- On Render free tier, the server sleeps after 15 min — first request after sleep takes ~30 seconds to wake up
