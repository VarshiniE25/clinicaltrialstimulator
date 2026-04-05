# 🧬 GenAI Drug Discovery & Clinical Trial Simulation

An end-to-end AI pipeline that:
1. Generates drug-like molecules via **Ollama llama3**
2. Computes RDKit descriptors + Lipinski filter
3. Runs heuristic docking & ADMET toxicity scoring
4. Ranks candidates, runs synthetic clinical trials
5. Generates an AI scientific explanation — all live in **Streamlit**

---

## Project Structure

```
drug_discovery/
├── app.py                               ← Streamlit UI  (run this)
├── pipeline.py                          ← 8-step orchestrator
├── init_db.py                           ← Creates MySQL DB + all tables
├── config.py                            ← DB credentials + Ollama settings
├── requirements.txt
│
├── database/
│   └── db_connection.py                 ← MySQL connection helper
│
└── agents/
    ├── genai_molecule_generator.py      ← Step 1 : Ollama → SMILES + RDKit
    ├── lipinski_filter_agent.py         ← Step 2 : Rule of Five
    ├── docking_agent.py                 ← Step 3 : Heuristic docking score
    ├── admet_agent.py                   ← Step 4 : Heuristic toxicity score
    ├── drug_ranking_agent.py            ← Step 5 : Normalised ranking
    ├── synthetic_patient_generator.py   ← Step 6 : Ollama → patient cohort
    ├── clinical_trial_agent.py          ← Step 7 : Drug × patient simulation
    └── drug_explanation_agent.py        ← Step 8 : Ollama → explanation
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 + | Runtime |
| MySQL | 8.0 + | Database |
| Ollama | latest | LLM backend |
| llama3 model | — | Molecule & patient generation |

---

## Step-by-Step Setup & Run Guide

### Step 1 — Install MySQL

**macOS**
```bash
brew install mysql
brew services start mysql
# Set a root password
mysql_secure_installation
```

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install mysql-server -y
sudo systemctl start mysql
sudo mysql_secure_installation
```

**Windows**
Download the MySQL installer from https://dev.mysql.com/downloads/installer/
Choose "Server only", follow the wizard, set a root password.

---

### Step 2 — Install Ollama

**macOS / Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**
Download the installer from https://ollama.com/download

---

### Step 3 — Pull the llama3 model

```bash
ollama pull llama3
```

This downloads ~4.7 GB. Wait for it to finish.

---

### Step 4 — Start Ollama

```bash
ollama serve
```

Leave this terminal open. Ollama listens on http://localhost:11434.

---

### Step 5 — Clone / copy the project

```
drug_discovery/          ← this folder
```

Make sure your folder has exactly the structure shown above.

---

### Step 6 — Edit config.py

Open `config.py` and set your MySQL root password:

```python
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = "YOUR_PASSWORD_HERE"   # ← change this
DB_NAME     = "ai_drug_discovery"

OLLAMA_MODEL = "llama3"
OLLAMA_HOST  = "http://localhost:11434"
```

---

### Step 7 — Create a Python virtual environment

```bash
cd drug_discovery
python -m venv venv

# Activate
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

---

### Step 8 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **RDKit note**: if `pip install rdkit` fails, install via conda instead:
> ```bash
> conda install -c conda-forge rdkit
> ```

---

### Step 9 — Initialise the MySQL database

```bash
python init_db.py
```

Expected output:
```
✓ Database 'ai_drug_discovery' ready.

Creating tables…
  ✓ molecules
  ✓ drug_candidates
  ✓ patients
  ✓ clinical_trial_results
  ✓ trial_patient_responses

✅ All tables ready.
```

---

### Step 10 — Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Using the App

1. Type a disease name in the sidebar — e.g. `Diabetes`, `Cancer`, `Alzheimer's`
2. Adjust **Molecules to generate** (5–30) and **Synthetic patients** (10–50)
3. Click **▶ Run Full Drug Discovery Pipeline**
4. Watch all 8 steps complete with live log output
5. Explore results:
   - §3  Drug screening table (docking + toxicity)
   - §4  Lipinski validation table
   - §6  Patient cohort + aggregated success rate
   - §7  Top candidate properties, radar chart, AI explanation
   - §8  Analytics charts (docking distribution, scatter, trial bar chart)
   - §9  Full ranked molecule table

---

## MySQL Tables

| Table | Description |
|-------|-------------|
| `molecules` | SMILES + RDKit descriptors + docking/toxicity scores |
| `drug_candidates` | Ranked molecules with composite score and position |
| `patients` | Synthetic patient cohort |
| `clinical_trial_results` | Aggregated per-drug success & side-effect rates |
| `trial_patient_responses` | Per-patient, per-drug effectiveness & side-effect risk |

To inspect the data after a run:
```sql
USE ai_drug_discovery;
SELECT smiles, docking_score, toxicity_score FROM molecules;
SELECT rank_position, ranking_score FROM drug_candidates ORDER BY rank_position;
SELECT * FROM clinical_trial_results;
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Cannot reach Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3` |
| `Access denied for user 'root'` | Check DB_PASSWORD in config.py |
| `Unknown database 'ai_drug_discovery'` | Run `python init_db.py` |
| `No module named 'rdkit'` | Install via conda: `conda install -c conda-forge rdkit` |
| `No module named 'mysql.connector'` | Run `pip install mysql-connector-python` |

---

## Scoring Formulas

**Docking Score** (range −15 to 0, lower = stronger binding):
```
base = −8.0
MW  200–450 Da  → up to −2.0 bonus
LogP 1–3        → −1.5 bonus
HBD  (capped 3) → × −0.6 each
HBA  (capped 5) → × −0.4 each
RotBonds > 3    → +0.3 penalty each
```

**Toxicity Score** (range 0 to 1, lower = safer):
```
base = 0.2
MW > 500        → + (MW−500)/1000 × 0.3
LogP > 4        → + (LogP−4) × 0.08
HBD > 3         → + (HBD−3) × 0.05
HBA > 7         → + (HBA−7) × 0.04
RotBonds > 8    → + (rot−8) × 0.03
```

**Ranking Score** (lower = better candidate):
```
norm_docking  = −docking / 15
ranking_score = −(0.6 × norm_docking − 0.4 × toxicity)
```
