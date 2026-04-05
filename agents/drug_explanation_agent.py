"""
agents/drug_explanation_agent.py
=================================
Uses Ollama (llama3) to generate a scientific explanation of the top
drug candidate fetched from MySQL.
Raises RuntimeError if Ollama is unreachable — no silent fallback.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import ollama


class DrugExplanationAgent:

    def __init__(self, log_fn=None):
        self.log = log_fn or print

    # ── Fetch top candidate from MySQL ────────────────────────────────────
    def _fetch_top_candidate(self, conn) -> dict | None:
        cur = conn.cursor()
        cur.execute(
            "SELECT dc.molecule_id, m.smiles, m.docking_score, "
            "       m.toxicity_score, dc.ranking_score, dc.rank_position "
            "FROM drug_candidates dc "
            "JOIN molecules m ON m.id = dc.molecule_id "
            "ORDER BY dc.rank_position ASC "
            "LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "molecule_id":   row[0],
            "smiles":        row[1],
            "docking_score": row[2],
            "toxicity_score":row[3],
            "ranking_score": row[4],
            "rank_position": row[5],
        }

    # ── Call Ollama ───────────────────────────────────────────────────────
    def _call_ollama(self, candidate: dict, disease: str) -> str:
        prompt = (
            "You are a computational drug discovery expert. "
            f"In 3-4 sentences explain why the following molecule is a "
            f"promising drug candidate for {disease}. "
            "Discuss binding affinity, safety profile, and drug-likeness.\n\n"
            f"SMILES        : {candidate['smiles']}\n"
            f"Docking Score : {candidate['docking_score']}\n"
            f"Toxicity Score: {candidate['toxicity_score']}\n"
            f"Ranking Score : {candidate['ranking_score']}\n"
        )

        try:
            client   = ollama.Client(host=config.OLLAMA_HOST)
            response = client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.6, "num_predict": 350},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST}.\n"
                f"Make sure Ollama is running:  ollama serve\n"
                f"Original error: {e}"
            )

    # ── Public entry ──────────────────────────────────────────────────────
    def run(self, conn, disease: str = "the target disease") -> tuple[str, dict] | None:
        candidate = self._fetch_top_candidate(conn)
        if not candidate:
            self.log("⚠  No ranked candidates found in MySQL.")
            return None

        self.log(
            f"💡 Generating explanation for #{candidate['rank_position']} "
            f"— {candidate['smiles']}…"
        )
        explanation = self._call_ollama(candidate, disease)
        self.log("✅ Drug explanation generated.")
        return explanation, candidate
