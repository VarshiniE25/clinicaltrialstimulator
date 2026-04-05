"""
agents/genai_molecule_generator.py
====================================
Generates drug-like SMILES via Ollama (llama3) and stores them in MySQL.
Raises RuntimeError if Ollama is unreachable — no silent fallback.
"""

import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import ollama
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


class GenAIMoleculeGenerator:

    def __init__(self, disease: str = "diabetes", log_fn=None):
        self.disease = disease.lower()
        self.log     = log_fn or print

    # ── Ollama call ───────────────────────────────────────────────────────
    def fetch_smiles_from_llm(self, n: int = 10) -> list[str]:
        prompt = (
            f"Generate {n} valid SMILES strings for small-molecule drug candidates "
            f"targeting {self.disease}.\n"
            "Rules:\n"
            "- Output ONLY SMILES strings, one per line.\n"
            "- No numbering, no explanation, no extra text.\n"
            "- Every SMILES must be chemically valid.\n"
            "Example output:\nCCO\nc1ccccc1\nCC(=O)Nc1ccc(O)cc1\n"
        )

        self.log(f"🤖 Querying Ollama ({config.OLLAMA_MODEL}) for {n} molecules…")

        try:
            client   = ollama.Client(host=config.OLLAMA_HOST)
            response = client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST}.\n"
                f"Make sure Ollama is running:  ollama serve\n"
                f"And the model is pulled:      ollama pull {config.OLLAMA_MODEL}\n"
                f"Original error: {e}"
            )

        raw    = response["message"]["content"]
        smiles = []
        for line in raw.strip().splitlines():
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            line = re.sub(r"[*_`\-\*]", "", line).strip()
            if line and " " not in line:        # skip lines with spaces (explanatory text)
                smiles.append(line)

        self.log(f"📥 Ollama returned {len(smiles)} raw SMILES strings.")
        return smiles

    # ── RDKit descriptor computation ──────────────────────────────────────
    def compute_descriptors(self, smiles: str) -> dict | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "smiles":            smiles,
            "molecular_weight":  round(Descriptors.MolWt(mol),    4),
            "logp":              round(Descriptors.MolLogP(mol),   4),
            "h_bond_donors":     Lipinski.NumHDonors(mol),
            "h_bond_acceptors":  Lipinski.NumHAcceptors(mol),
            "rotatable_bonds":   Lipinski.NumRotatableBonds(mol),
        }

    # ── MySQL insert ──────────────────────────────────────────────────────
    def insert_molecules(self, conn, molecules: list[dict]) -> int:
        cur = conn.cursor()
        sql = (
            "INSERT INTO molecules "
            "(smiles, molecular_weight, logp, h_bond_donors, h_bond_acceptors, rotatable_bonds) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        for mol in molecules:
            cur.execute(sql, (
                mol["smiles"], mol["molecular_weight"], mol["logp"],
                mol["h_bond_donors"], mol["h_bond_acceptors"], mol["rotatable_bonds"],
            ))
        conn.commit()
        return len(molecules)

    # ── Public entry ──────────────────────────────────────────────────────
    def run(self, conn, n: int = 10) -> list[dict]:
        smiles_list = self.fetch_smiles_from_llm(n)

        valid, skipped = [], 0
        for smi in smiles_list:
            desc = self.compute_descriptors(smi)
            if desc:
                valid.append(desc)
            else:
                self.log(f"  ⚠ Invalid SMILES skipped: {smi}")
                skipped += 1

        self.log(f"✅ Valid molecules: {len(valid)} | Skipped: {skipped}")

        if not valid:
            raise RuntimeError(
                "Ollama returned no valid SMILES. "
                "Try increasing the molecule count or re-running."
            )

        inserted = self.insert_molecules(conn, valid)
        self.log(f"💾 {inserted} molecules saved to MySQL.")
        return valid
