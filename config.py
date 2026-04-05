"""
config.py
=========
Central configuration for the Drug Discovery pipeline.

Edit the values below to match your local MySQL setup,
then run:  streamlit run app.py
"""

# ── MySQL ──────────────────────────────────────────────────────────────────
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = "Varshini@25"        # ← your MySQL root password
DB_NAME     = "ai_drug_discovery"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3"          # must be pulled: ollama pull llama3
OLLAMA_HOST  = "http://localhost:11434"
