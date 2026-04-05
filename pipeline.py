"""
pipeline.py
===========
Full 8-step pipeline orchestrator.

  log_fn  : callable(str) — Streamlit uses this to stream live progress.
  Returns : dict with all results consumed by app.py.
"""

import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from init_db import init_db, reset_run_data
from database.db_connection import get_connection
from agents.genai_molecule_generator    import GenAIMoleculeGenerator
from agents.lipinski_filter_agent       import LipinskiFilterAgent
from agents.docking_agent               import DockingAgent
from agents.admet_agent                 import ADMETAgent
from agents.drug_ranking_agent          import DrugRankingAgent
from agents.synthetic_patient_generator import SyntheticPatientGenerator
from agents.clinical_trial_agent        import ClinicalTrialAgent
from agents.drug_explanation_agent      import DrugExplanationAgent


def run_pipeline(
    disease:       str = "diabetes",
    num_molecules: int = 10,
    num_patients:  int = 20,
    log_fn=None,
) -> dict:
    log = log_fn or print
    t0  = time.time()

    # ── DB initialisation ─────────────────────────────────────────────────
    init_db()
    conn = get_connection()
    reset_run_data(conn)

    results = {}

    # ── Step 1 ────────────────────────────────────────────────────────────
    log("🧬 **Step 1/8 — GenAI Molecule Generation**")
    gen = GenAIMoleculeGenerator(disease=disease, log_fn=log)
    results["molecules"] = gen.run(conn, n=num_molecules)

    # ── Step 2 ────────────────────────────────────────────────────────────
    log("⚗  **Step 2/8 — Lipinski Drug-Likeness Filter**")
    results["lipinski"] = LipinskiFilterAgent(log_fn=log).run_filter(conn)

    # ── Step 3 ────────────────────────────────────────────────────────────
    log("⚓ **Step 3/8 — Heuristic Docking Simulation**")
    results["docking"] = DockingAgent(log_fn=log).run_docking(conn)

    # ── Step 4 ────────────────────────────────────────────────────────────
    log("☣  **Step 4/8 — ADMET Toxicity Prediction**")
    results["admet"] = ADMETAgent(log_fn=log).run_toxicity_prediction(conn)

    # ── Step 5 ────────────────────────────────────────────────────────────
    log("🏅 **Step 5/8 — Drug Ranking**")
    results["ranked"] = DrugRankingAgent(log_fn=log).run_ranking(conn)

    # ── Step 6 ────────────────────────────────────────────────────────────
    log("👥 **Step 6/8 — Synthetic Patient Generation**")
    results["patients"] = SyntheticPatientGenerator(
        disease=disease, log_fn=log
    ).run(conn, n=num_patients)

    # ── Step 7 ────────────────────────────────────────────────────────────
    log("🏥 **Step 7/8 — Clinical Trial Simulation**")
    results["trial"] = ClinicalTrialAgent(log_fn=log).run_trial(conn)

    # ── Step 8 ────────────────────────────────────────────────────────────
    log("💡 **Step 8/8 — AI Drug Explanation**")
    out = DrugExplanationAgent(log_fn=log).run(conn, disease=disease)
    if out:
        results["explanation"], results["top_candidate"] = out
    else:
        results["explanation"]  = "No explanation available."
        results["top_candidate"] = None

    conn.close()
    results["elapsed"] = round(time.time() - t0, 1)
    log(f"🎉 **Pipeline complete in {results['elapsed']}s**")
    return results
