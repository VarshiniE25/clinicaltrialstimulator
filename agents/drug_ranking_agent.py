"""
agents/drug_ranking_agent.py
=============================
Normalised composite ranking.

Formula
-------
  norm_docking  = -docking_score / 15      (flip: higher = better binding)
  ranking_score = -(0.6 × norm_docking − 0.4 × toxicity_score)

  Sort ascending → lower ranking_score = better drug candidate.
  Best possible: -0.6  (perfect binding, zero toxicity)
  Worst possible: +0.4  (no binding, maximum toxicity)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class DrugRankingAgent:

    W_BINDING  = 0.6
    W_TOXICITY = 0.4

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    def _ranking_score(self, docking: float, toxicity: float) -> float:
        nd = max(0.0, min(1.0, -docking / 15.0))
        nt = max(0.0, min(1.0, toxicity))
        return round(-(self.W_BINDING * nd - self.W_TOXICITY * nt), 6)

    def run_ranking(self, conn) -> list[dict]:
        cur = conn.cursor()

        cur.execute(
            "SELECT id, smiles, docking_score, toxicity_score "
            "FROM molecules "
            "WHERE lipinski_pass = 1 "
            "  AND docking_score  IS NOT NULL "
            "  AND toxicity_score IS NOT NULL"
        )
        rows = cur.fetchall()

        ranked = sorted(
            [
                {
                    "id":       r[0],
                    "smiles":   r[1],
                    "docking":  r[2],
                    "toxicity": r[3],
                    "score":    self._ranking_score(r[2], r[3]),
                }
                for r in rows
            ],
            key=lambda x: x["score"],
        )

        # Clear previous ranking and re-insert
        cur.execute("DELETE FROM drug_candidates")
        for pos, m in enumerate(ranked, 1):
            cur.execute(
                "INSERT INTO drug_candidates "
                "(molecule_id, ranking_score, rank_position) "
                "VALUES (%s, %s, %s)",
                (m["id"], m["score"], pos),
            )
            m["rank_position"] = pos

        conn.commit()
        best = ranked[0]["smiles"] if ranked else "N/A"
        self.log(f"✅ Ranked {len(ranked)} molecules. Best: {best}")
        return ranked
