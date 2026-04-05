"""
agents/lipinski_filter_agent.py
================================
Applies Lipinski's Rule of Five to every molecule in the DB
and sets lipinski_pass = 1 (pass) or 0 (fail).

Rule of Five:
  MW  ≤ 500 Da
  LogP ≤ 5
  H-bond donors    ≤ 5
  H-bond acceptors ≤ 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class LipinskiFilterAgent:

    MW_MAX   = 500
    LOGP_MAX = 5
    HBD_MAX  = 5
    HBA_MAX  = 10

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    def _passes(self, mw: float, logp: float, hbd: int, hba: int) -> bool:
        return (
            mw   <= self.MW_MAX
            and logp <= self.LOGP_MAX
            and hbd  <= self.HBD_MAX
            and hba  <= self.HBA_MAX
        )

    def run_filter(self, conn) -> list[dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, smiles, molecular_weight, logp, "
            "h_bond_donors, h_bond_acceptors FROM molecules"
        )
        rows    = cur.fetchall()
        results = []
        passed  = failed = 0

        for row in rows:
            mol_id, smiles, mw, logp, hbd, hba = row
            ok = self._passes(mw, logp, hbd, hba)
            cur.execute(
                "UPDATE molecules SET lipinski_pass = %s WHERE id = %s",
                (1 if ok else 0, mol_id),
            )
            if ok: passed += 1
            else:  failed += 1
            results.append({
                "id": mol_id, "smiles": smiles,
                "mw": mw, "logp": logp, "hbd": hbd, "hba": hba,
                "pass": ok,
            })

        conn.commit()
        self.log(f"✅ Lipinski filter: {passed} passed, {failed} failed.")
        return results
