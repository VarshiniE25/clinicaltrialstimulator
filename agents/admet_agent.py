"""
agents/admet_agent.py
======================
Heuristic toxicity score in [0, 1].  Lower = safer.

Formula
-------
  base = 0.2  (baseline for a typical small molecule)
  MW > 500    → harder to metabolise / excrete
  LogP > 4    → bioaccumulation risk
  LogP < 0    → poor ADME (extreme hydrophilicity)
  HBD > 3     → potential membrane disruption
  HBA > 7     → off-target binding risk
  RotBonds > 8 → high conformational flexibility → non-selectivity
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class ADMETAgent:

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    def _score(self, mw: float, logp: float,
                hbd: int, hba: int, rot: int) -> float:
        s = 0.2

        if mw > 500:
            s += (mw - 500) / 1000 * 0.3
        elif mw < 100:
            s += 0.1

        if logp > 4:
            s += (logp - 4) * 0.08
        elif logp < 0:
            s += abs(logp) * 0.05

        if hbd > 3:
            s += (hbd - 3) * 0.05

        if hba > 7:
            s += (hba - 7) * 0.04

        if rot > 8:
            s += (rot - 8) * 0.03

        return round(max(0.0, min(1.0, s)), 4)

    def run_toxicity_prediction(self, conn) -> list[dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, smiles, molecular_weight, logp, "
            "h_bond_donors, h_bond_acceptors, rotatable_bonds "
            "FROM molecules "
            "WHERE lipinski_pass = 1 AND toxicity_score IS NULL"
        )
        rows    = cur.fetchall()
        results = []

        for row in rows:
            mid, smi, mw, logp, hbd, hba, rot = row
            tox = self._score(mw, logp, hbd, hba, rot)
            cur.execute(
                "UPDATE molecules SET toxicity_score = %s WHERE id = %s",
                (tox, mid),
            )
            results.append({"id": mid, "smiles": smi, "toxicity_score": tox})

        conn.commit()
        self.log(f"✅ Toxicity prediction complete: {len(rows)} molecules scored.")
        return results
