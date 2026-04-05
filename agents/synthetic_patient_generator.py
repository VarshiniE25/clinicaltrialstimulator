"""
agents/synthetic_patient_generator.py
=======================================
Generates synthetic patient profiles via Ollama (llama3) and stores
them in MySQL.  Raises RuntimeError if Ollama is unreachable.
"""

import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
import ollama


class SyntheticPatientGenerator:

    def __init__(self, disease: str = "diabetes", log_fn=None):
        self.disease = disease
        self.log     = log_fn or print

    def _fetch_from_ollama(self, n: int) -> list[dict]:
        prompt = (
            f"Generate exactly {n} synthetic patient profiles for a clinical trial "
            f"targeting {self.disease}.\n"
            "Return ONLY a valid JSON array — no markdown fences, no explanation.\n"
            "Each object must have exactly these keys:\n"
            '  "age"                  : integer (18–80)\n'
            '  "gender"               : "Male" or "Female"\n'
            '  "weight"               : float in kg (45–130)\n'
            '  "genetic_marker"       : one of TCF7L2, PPARG, BRCA1, CYP2D6, '
            'CYP3A4, ALDH2, MTHFR, APOE\n'
            '  "liver_function_score" : float between 0.0 and 1.0\n'
            "Example: "
            '[{"age":45,"gender":"Male","weight":82.5,'
            '"genetic_marker":"TCF7L2","liver_function_score":0.75}]'
        )

        self.log(f"🤖 Querying Ollama ({config.OLLAMA_MODEL}) for {n} patient profiles…")

        try:
            client   = ollama.Client(host=config.OLLAMA_HOST)
            response = client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7},
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST}.\n"
                f"Make sure Ollama is running:  ollama serve\n"
                f"Original error: {e}"
            )

        raw = response["message"]["content"].strip()
        # Strip markdown code fences if the model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            patients = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Ollama returned invalid JSON for patient profiles.\n"
                f"Response was:\n{raw[:400]}\nError: {e}"
            )

        self.log(f"📥 Received {len(patients)} patient profiles from Ollama.")
        return patients

    def run(self, conn, n: int = 20) -> list[dict]:
        patients = self._fetch_from_ollama(n)

        cur = conn.cursor()
        sql = (
            "INSERT INTO patients "
            "(age, gender, weight, genetic_marker, liver_function_score) "
            "VALUES (%s, %s, %s, %s, %s)"
        )
        for p in patients:
            cur.execute(sql, (
                int(p["age"]),
                str(p["gender"]),
                float(p["weight"]),
                str(p["genetic_marker"]),
                float(p["liver_function_score"]),
            ))

        conn.commit()
        self.log(f"✅ {len(patients)} patients saved to MySQL.")
        return patients
