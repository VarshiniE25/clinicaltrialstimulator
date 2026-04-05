"""
agents/clinical_trial_agent.py
================================
Simulates drug × patient clinical trial responses.

Stores two result sets in MySQL:
  1. trial_patient_responses  — per-patient, per-drug effectiveness & side-effect risk
  2. clinical_trial_results   — aggregated success_rate & side_effect_rate per drug
"""

import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

EFFICACY_MARKERS = {"TCF7L2", "PPARG", "BRCA1"}


class ClinicalTrialAgent:

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    def _simulate(
        self,
        docking:  float,
        toxicity: float,
        liver:    float,
        marker:   str,
    ) -> tuple[float, float]:
        """
        Returns (effectiveness, side_effect_risk) both in [0, 1].

        effectiveness:
          - stronger binding (more negative docking) → higher
          - better liver function → better drug processing
          - efficacy-linked genetic markers → +0.1 boost

        side_effect_risk:
          - higher toxicity → more risk
          - lower liver function → reduced clearance → more risk
          - small random noise
        """
        nd  = max(0.0, min(1.0, -docking / 15.0))
        gb  = 0.1 if marker in EFFICACY_MARKERS else 0.0

        eff  = nd * 0.5 + liver * 0.3 + gb + random.uniform(-0.05, 0.05)
        side = toxicity * 0.6 + (1.0 - liver) * 0.3 + random.uniform(0.0, 0.1)

        return (
            round(max(0.0, min(1.0, eff)),  4),
            round(max(0.0, min(1.0, side)), 4),
        )

    def run_trial(self, conn) -> dict:
        cur = conn.cursor()

        # ── Fetch top 5 ranked drug candidates ───────────────────────────
        cur.execute(
            "SELECT dc.molecule_id, m.smiles, m.docking_score, m.toxicity_score "
            "FROM drug_candidates dc "
            "JOIN molecules m ON m.id = dc.molecule_id "
            "ORDER BY dc.rank_position ASC "
            "LIMIT 5"
        )
        drugs = cur.fetchall()

        # ── Fetch patient cohort ──────────────────────────────────────────
        cur.execute(
            "SELECT id, liver_function_score, genetic_marker FROM patients"
        )
        patients = cur.fetchall()

        if not drugs:
            raise RuntimeError("No drug candidates found. Run ranking first.")
        if not patients:
            raise RuntimeError("No patients found. Run patient generator first.")

        # ── Clear previous trial data ─────────────────────────────────────
        cur.execute("DELETE FROM clinical_trial_results")
        cur.execute("DELETE FROM trial_patient_responses")

        summary = []
        patient_sql = (
            "INSERT INTO trial_patient_responses "
            "(drug_id, patient_id, effectiveness, side_effect_risk) "
            "VALUES (%s, %s, %s, %s)"
        )
        aggregate_sql = (
            "INSERT INTO clinical_trial_results "
            "(drug_id, success_rate, side_effect_rate) "
            "VALUES (%s, %s, %s)"
        )

        for mol_id, smiles, docking, toxicity in drugs:
            effs, sides = [], []

            for pid, liver, marker in patients:
                eff, side = self._simulate(docking, toxicity, liver, marker)
                effs.append(eff)
                sides.append(side)
                # Granular row
                cur.execute(patient_sql, (mol_id, pid, eff, side))

            sr  = round(sum(effs)  / len(effs),  4)
            ser = round(sum(sides) / len(sides), 4)

            # Aggregate row
            cur.execute(aggregate_sql, (mol_id, sr, ser))

            summary.append({
                "smiles":           smiles,
                "docking":          docking,
                "toxicity":         toxicity,
                "success_rate":     sr,
                "side_effect_rate": ser,
            })

        conn.commit()
        self.log(
            f"✅ Clinical trial: {len(drugs)} drugs × {len(patients)} patients. "
            f"Results saved to MySQL."
        )
        return {"drugs": summary, "n_patients": len(patients)}
