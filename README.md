# NERVE NOC — Network Emergency & Risk Visualization Engine
### AI-Powered Telecom Network Diagnostic Command Center

---

## 📌 Project Overview
**NERVE NOC** is an AI-powered Telecom Network Diagnostic Command Center designed for proactive fault prediction, root-cause explanation (XAI), counterfactual scenario simulation, and evidence-grounded standard operating procedure (SOP) troubleshooting.

Built using the Telstra Network Disruptions telemetry dataset, NERVE NOC bridges machine learning intelligence with network operations center (NOC) workflows.

---

## 🚀 Core Capabilities

### 1. 📡 Fleet Overview (Command Center Hub)
- **High-Level KPIs**: Real-time fleet health aggregation ($59\text{ Healthy} + 25\text{ Warning} + 16\text{ Critical} = 100\text{ Devices}$).
- **Fleet Risk Distribution**: Interactive health gauge ($73/100$) and multi-class severity breakdown donut.
- **Risk by Location**: Global geographic risk density visualization (*synthetic coordinates representing anonymized location IDs*).
- **Network Topology Graph**: Graph visualization of device interconnections and regional clusters.
- **Monitored Infrastructure Table**: Filter by severity, search by Device ID/Location, and execute batch PDF report export.

### 2. ⚡ Operations Workspace
- **Active Incidents Stream**: Prioritized incident feed with left-accented severity status and one-click diagnostic deep dive.
- **Live Telemetry Inference Form**: Real-time XGBoost inference sandbox allowing operators to adjust location, severity tier, event types, and log feature volumes to instantly evaluate fault severity probabilities.

### 3. 🔍 Device Detail Deep Dive
- **Composite Health Score & Risk Vitals**: Real-time CPU, memory, latency, and packet loss metrics alongside failure blast radius calculation.
- **Explainable AI (SHAP TreeExplainer)**: Attribution weights ranking top anomaly triggers with plain-English feature translations.
- **What-If Counterfactual Simulator**: Interactive anomaly surge reduction slider simulating the impact of traffic rerouting and signal damping on projected device health.
- **SOP Verification & Citations**: Cross-referencing against 16 indexed standard operating procedure playbooks with match confidence scoring.
- **Automated Incident Reporting**: One-click technical PDF incident report generation.

### 4. 🤖 Tilly — NOC Assistant
- **Hybrid Intent Router**: Automatically classifies incoming operator queries into numerical telemetry queries, technical SOP lookups, device-specific diagnostics, or off-topic questions.
- **Numerical Telemetry Aggregation**: Direct DataFrame execution for aggregations, counts, and location hotspot queries.
- **FAISS-Powered SOP Grounding**: Nearest-neighbor retrieval over technical playbooks and similar historical device incidents.
- **Domain Guardrails**: Strict refusal of non-network/off-topic questions.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **User Interface** | Streamlit, HTML5/CSS3 Custom Theme, Plotly Express & Graph Objects |
| **Machine Learning** | XGBoost (Multi-class Classifier), SHAP (TreeExplainer), Scikit-Learn |
| **Information Retrieval (RAG)** | FAISS (CPU), Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Large Language Models** | OpenRouter API / OpenAI Client (with offline local fallback routing) |
| **Data Processing & Storage** | Pandas, NumPy, Joblib, JSON Playbook Store |
| **Reporting & Export** | FPDF2, Python ZipFile |
| **Security & Auth** | Google OAuth 2.0, Session State Management |

---

## 📂 Project Architecture

```text
TelecomFaultPrediction/
├── app.py                      # Main Streamlit Command Center Application
├── requirements.txt            # Python Dependencies
├── .env.template               # Environment Variable Template
├── data/
│   ├── raw/                    # Telstra Network Disruptions Datasets
│   └── playbooks/
│       └── telecom_sops.json   # 16 Indexed Technical SOP Playbooks
├── models/
│   ├── model.pkl               # Trained XGBoost Multi-Class Model
│   ├── location_freq_map.pkl   # Location Frequency Encoding
│   ├── faiss_playbooks.index   # FAISS Vector Index (SOP Playbooks)
│   ├── playbook_summaries.pkl  # Playbook Document Store
│   ├── faiss_devices.index     # FAISS Vector Index (Historical Devices)
│   └── device_summaries.pkl    # Device Incident Document Store
└── src/
    ├── auth.py                 # Google OAuth & Authentication Gateway
    ├── features.py             # Feature Engineering & Preprocessing
    ├── llm_service.py          # Hybrid Intent Router & LLM Client
    ├── predictor.py            # XGBoost Inference & SHAP Explainer
    ├── preprocessing.py        # Dataset Ingestion & Merging
    ├── rag_service.py          # FAISS Multi-Index Retrieval Engine
    ├── recommender.py          # Rule Engine & SOP Verification
    ├── topology.py             # Graph & Geographic Visualization
    └── what_if_engine.py       # Counterfactual Scenario Simulator
```

---

## ⚡ Setup & Local Execution

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git

### 2. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/NERVE-NOC.git
cd NERVE-NOC

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.template` to `.env` and fill in your keys:
```bash
cp .env.template .env
```
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
DEV_AUTH=true
```
*(Note: If `DEV_AUTH=true`, one-click demo login is enabled for local offline testing).*

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔍 Implemented vs. Roadmap

| Capability | Status | Implementation Detail |
| :--- | :--- | :--- |
| **Multi-Class Fault Prediction** | ✅ Implemented | XGBoost trained on severity classes 0, 1, and 2 |
| **SHAP Root Cause Attribution** | ✅ Implemented | Local TreeExplainer attribution translated to plain-English labels |
| **What-If Scenario Simulation** | ✅ Implemented | Counterfactual volume reduction projection |
| **Hybrid Intent Routing RAG** | ✅ Implemented | Numerical DataFrame routing + FAISS SOP grounding |
| **16 Indexed SOP Playbooks** | ✅ Implemented | Full technical playbooks covering resource types, events & log signals |
| **PDF Incident Reporting** | ✅ Implemented | Single-device PDF and batch multi-device ZIP downloads |
| **Network Topology & Geo Risk** | ✅ Implemented | Deterministic graph and risk distribution (*synthetic positioning disclaimer*) |
| *Real-Time Kafka Streaming* | 🗺️ Roadmap | Planned for high-throughput live telemetry ingestion |
| *Dynamic GIS / Real GPS Tracking*| 🗺️ Roadmap | Planned for live cellular tower GPS coordinate mapping |
| *Automated Self-Healing Remediation*| 🗺️ Roadmap | Planned for closed-loop automated configuration deployment |

---

## 👥 Project Team
- **Sivasakthi E** (Team Leader)
- **Shiju S**
- **Geethanjali V N**
- **Shanmuga Sundaram S N**
- **Kishore S**

