"""
agents/docking_agent.py
========================
Heuristic pseudo-docking score in [-15, 0].  More negative = stronger binding.

Formula
-------
  base = -8.0  (anchor for a typical drug-like molecule)
  MW:       200–450 Da optimal → reward; outside range → penalty
  LogP:     1–3 optimal → -1.5 bonus; extremes → penalty
  HBD:      each donor adds polar contacts  (benefit capped at 3 donors)
  HBA:      each acceptor contributes       (benefit capped at 5 acceptors)
  RotBonds: flexibility costs entropy       (penalty above 3)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class DockingAgent:

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    def _score(self, mw: float, logp: float,
                hbd: int, hba: int, rot: int) -> float:
        s = -8.0

        # Molecular weight
        if mw < 150:
            s += 3.0
        elif mw <= 450:
            s -= (450 - mw) / 450 * 2.0
        else:
            s += (mw - 450) / 100 * 0.8

        # LogP
        if 1.0 <= logp <= 3.0:
            s -= 1.5
        elif logp < 0:
            s += abs(logp) * 0.4
        elif logp > 5:
            s += (logp - 5) * 0.5

        # H-bond donors
        s -= min(hbd, 3) * 0.6

        # H-bond acceptors
        s -= min(hba, 5) * 0.4

        # Rotatable bonds
        if rot > 3:
            s += (rot - 3) * 0.3

        return round(max(-15.0, min(0.0, s)), 4)

    def run_docking(self, conn) -> list[dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, smiles, molecular_weight, logp, "
            "h_bond_donors, h_bond_acceptors, rotatable_bonds "
            "FROM molecules "
            "WHERE lipinski_pass = 1 AND docking_score IS NULL"
        )
        rows    = cur.fetchall()
        results = []

        for row in rows:
            mid, smi, mw, logp, hbd, hba, rot = row
            score = self._score(mw, logp, hbd, hba, rot)
            cur.execute(
                "UPDATE molecules SET docking_score = %s WHERE id = %s",
                (score, mid),
            )
            results.append({"id": mid, "smiles": smi, "docking_score": score})

        conn.commit()
        self.log(f"✅ Docking complete: {len(rows)} molecules scored.")
        return results
