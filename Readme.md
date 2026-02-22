# 🧠 MIA-Spector: Membership Inference Analysis Platform

**MIA-Spector** is a unified platform designed for privacy risk evaluation on **Large Language Models (LLMs)**, providing Membership Inference Attack (MIA) analysis and visualization.

The platform contains two major components:

1. 🧩 **MIA-Inspector API (Backend)** — FastAPI-powered inference & decision service  
2. 💡 **MIA-Portal (Frontend)** — React + Tailwind interactive visualization console

---

## 🌟 Features Overview

| Module | Function | Tech Highlights |
| ------ | -------- | --------------- |
| **Backend — MIA-Inspector API** | Unified LLM loading (Pythia, LLaMA, etc.) with MIA metric computation and decision | ✔ Async FastAPI<br>✔ Auto model registry + caching<br>✔ Prometheus performance metrics |
| **Frontend — MIA-Portal** | Interactive dashboard for selecting model/config and performing single-sample MIA decision | ✔ Modern dark UI<br>✔ Responsive layout<br>✔ Pretty JSON rendering |
| **Security Layer (Auth + Rate Limit)** | API key authentication with Token-Bucket rate control | ✔ Dynamic `.env` loading<br>✔ Configurable QPS limit<br>✔ Per-client counters |
| **Metric Engine** | Supports Min-K%, Min-K++, PPL, Renyi entropy, etc. | ✔ YAML-based threshold loading<br>✔ Multiple inference modes (Youden J / FPR@α) |
| **System Monitoring** | `/metrics` endpoint export | ✔ Traffic stats, latency histograms, GPU monitoring |

---

## 🧩 Project Structure

```text
MIA-Spector/
│
├── service/
│   ├── app/              ← Backend API (FastAPI)
│   │   ├── main.py       # Entrypoint
│   │   ├── deps.py       # Model loading & caching
│   │   ├── middlewares/  # Auth + rate limit
│   │   ├── routers/      # Health, Meta, Decision
│   │   └── config.py     # Global config (MODELS, CFGS)
│   │
│   ├── portal/           ← Frontend (React + Vite + Tailwind)
│   │   ├── src/
│   │   │   ├── pages/Console.jsx
│   │   │   ├── components/SectionCard.jsx
│   │   │   └── index.css
│   │   └── vite.config.js
│   │
│   └── uvicorn.run.sh    # One-click launch script
│
├── attacks/               # Core MIA metric algorithms
├── src/                   # Utility modules
├── configs/               # YAML thresholds
├── models/                # Local HF model weights
└── README.md
```

---

## ⚙️ Installation

### 🖥️ Windows & Docker Support

For detailed instructions on running MIA-Spector on Windows 11 (via Docker or Native), please refer to [WINDOWS.md](WINDOWS.md).

### Backend Environment (Linux/Mac)

```bash
conda create -n mia-inspector python=3.11
conda activate mia-inspector
pip install -r requirements.txt
```

### Frontend
```bash
cd service/portal
npm install
```

````markdown
## 🚀 Launch & Usage

### 🔹 Start Backend

```bash
cd service/app
bash ../uvicorn.run.sh
````

or manually:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### `.env` Example

```bash
API_KEYS=abc123,def456
REQUIRE_AUTH=True
```

#### Authentication Test

```bash
curl -H "Authorization: Bearer abc123" http://localhost:8080/healthz
```

---

### 🔹 Start Frontend

```bash
cd service/portal
npm run dev
```

Visit: [http://localhost:5173](http://localhost:5173)

#### Initial Setup on UI:

1. Set API Base URL: `http://localhost:8080`
2. Enter API Key (e.g., `abc123`)
3. Click **Load Models / Configs**
4. Input text → Run decision

---

## 🎯 Backend API Endpoints

| Path          | Method | Description                   |
| ------------- | ------ | ----------------------------- |
| `/healthz`    | GET    | Health check                  |
| `/v1/models`  | GET    | Available model registry      |
| `/v1/configs` | GET    | Threshold configs             |
| `/v1/decide`  | POST   | Membership inference decision |
| `/metrics`    | GET    | Prometheus monitoring export  |

Example request:

```json
{
  "text": "The mitochondrion is the powerhouse of the cell.",
  "family": "pythia",
  "model": "pythia-410m",
  "cfg": "WikiMIA_length128",
  "metric_group": "mink++",
  "subkey": "0.3",
  "mode": "bestJ"
}
```

---

## 💡 Frontend Highlight Features

| Module          | Function                            | Tech                                     |
| --------------- | ----------------------------------- | ---------------------------------------- |
| Config Panel    | Bind API base, key, clientId        | React Hooks + LocalStorage               |
| Model Browser   | Fetch `/v1/models`                  | Axios + JSON viewer                      |
| Config Browser  | Fetch `/v1/configs`                 | Responsive grid, dark-mode tuned         |
| Inference Panel | Text input + model/metric selection | Tailwind form components                 |
| Result Viewer   | Pretty JSON rendering               | font-mono + shadow cards                 |
| Theme           | Dark-mode optimized                 | `dark:bg-slate-900` + custom gray scales |

---

## 🧠 Background

**Membership Inference Attack (MIA)** evaluates whether a model leaks training samples by observing its output distribution (e.g., PPL, Min-K%, confidence gaps).

MIA-Spector provides:

* Single-sample decision + batch analysis
* Multi-metric fusion & directional decision
* ROC / AUC visualization
* Cross-model threshold sharing via YAML configs

---

## 📊 Example Output

```json
{
  "decision": "Uncertain",
  "confidence": 0.67,
  "score": -0.76,
  "threshold": -0.91,
  "metric_group": "mink++",
  "subkey": "0.3",
  "direction": "+",
  "mode": "bestJ"
}
```

---

## 🧾 Citation

```bibtex
@misc{MIA-Spector2025,
  title  = {MIA-Spector: Unified Platform for Text and Image Membership Inference Analysis},
  author = {Liu, Jiajun and Collaborators},
  year   = {2025},
  url    = {https://github.com/JiajunLiu/MIA-Spector}
}
```

> ⚠️ This project is intended only for privacy & security research and defensive analysis. Any misuse for unauthorized attacks or data leakage is strictly prohibited.

