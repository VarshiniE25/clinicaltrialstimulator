"""
init_db.py
==========
Creates the MySQL database (if it doesn't exist) and all pipeline tables.
Also migrates pre-existing tables to add any missing columns.

Run once before starting the app:
    python init_db.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import mysql.connector
import config


# ── DDL — MySQL syntax ────────────────────────────────────────────────────
DDL_TABLES = [

    """CREATE TABLE IF NOT EXISTS molecules (
        id                 INT          AUTO_INCREMENT PRIMARY KEY,
        smiles             VARCHAR(512) NOT NULL,
        molecular_weight   FLOAT,
        logp               FLOAT,
        h_bond_donors      INT,
        h_bond_acceptors   INT,
        rotatable_bonds    INT,
        lipinski_pass      TINYINT(1)   DEFAULT 0,
        docking_score      FLOAT        DEFAULT NULL,
        toxicity_score     FLOAT        DEFAULT NULL,
        created_at         DATETIME     DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB""",

    """CREATE TABLE IF NOT EXISTS drug_candidates (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        molecule_id     INT NOT NULL,
        ranking_score   FLOAT,
        rank_position   INT,
        FOREIGN KEY (molecule_id) REFERENCES molecules(id) ON DELETE CASCADE
    ) ENGINE=InnoDB""",

    """CREATE TABLE IF NOT EXISTS patients (
        id                   INT AUTO_INCREMENT PRIMARY KEY,
        age                  INT,
        gender               VARCHAR(10),
        weight               FLOAT,
        genetic_marker       VARCHAR(100),
        liver_function_score FLOAT
    ) ENGINE=InnoDB""",

    """CREATE TABLE IF NOT EXISTS clinical_trial_results (
        id                INT AUTO_INCREMENT PRIMARY KEY,
        drug_id           INT NOT NULL,
        success_rate      FLOAT,
        side_effect_rate  FLOAT,
        FOREIGN KEY (drug_id) REFERENCES molecules(id) ON DELETE CASCADE
    ) ENGINE=InnoDB""",

    """CREATE TABLE IF NOT EXISTS trial_patient_responses (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        drug_id          INT NOT NULL,
        patient_id       INT NOT NULL,
        effectiveness    FLOAT,
        side_effect_risk FLOAT,
        FOREIGN KEY (drug_id)    REFERENCES molecules(id)  ON DELETE CASCADE,
        FOREIGN KEY (patient_id) REFERENCES patients(id)   ON DELETE CASCADE
    ) ENGINE=InnoDB""",
]

# ── Migration: columns that must exist on pre-existing tables ─────────────
# Format: (table_name, column_name, column_definition)
REQUIRED_COLUMNS = [
    ("molecules", "lipinski_pass",    "TINYINT(1) DEFAULT 0"),
    ("molecules", "docking_score",    "FLOAT DEFAULT NULL"),
    ("molecules", "toxicity_score",   "FLOAT DEFAULT NULL"),
    ("molecules", "molecular_weight", "FLOAT"),
    ("molecules", "logp",             "FLOAT"),
    ("molecules", "h_bond_donors",    "INT"),
    ("molecules", "h_bond_acceptors", "INT"),
    ("molecules", "rotatable_bonds",  "INT"),
    ("molecules", "created_at",       "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("drug_candidates",  "ranking_score",      "FLOAT"),
    ("drug_candidates",  "rank_position",       "INT"),
    ("patients",         "liver_function_score","FLOAT"),
    ("patients",         "genetic_marker",      "VARCHAR(100)"),
    ("patients",         "weight",              "FLOAT"),
    ("clinical_trial_results",    "success_rate",     "FLOAT"),
    ("clinical_trial_results",    "side_effect_rate", "FLOAT"),
    ("trial_patient_responses",   "effectiveness",    "FLOAT"),
    ("trial_patient_responses",   "side_effect_risk", "FLOAT"),
]


def create_database_if_missing():
    """Connect without specifying a database and CREATE DATABASE IF NOT EXISTS."""
    conn = mysql.connector.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    conn.close()
    print(f"✓ Database '{config.DB_NAME}' ready.")


def _get_existing_columns(cur, table: str) -> set:
    """Return the set of column names that already exist in *table*."""
    cur.execute(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
        (config.DB_NAME, table),
    )
    return {row[0].lower() for row in cur.fetchall()}


def migrate_tables(conn):
    """
    Add any missing columns to pre-existing tables.
    Safe to run repeatedly — skips columns that already exist.
    """
    cur = conn.cursor()
    added = 0

    for table, column, definition in REQUIRED_COLUMNS:
        existing = _get_existing_columns(cur, table)
        if column.lower() not in existing:
            sql = f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}"
            try:
                cur.execute(sql)
                print(f"  ➕ Added column {table}.{column}")
                added += 1
            except mysql.connector.Error as e:
                # 1060 = Duplicate column — race condition safety net
                if e.errno != 1060:
                    raise

    conn.commit()
    if added:
        print(f"  ✓ Migration complete: {added} column(s) added.")
    else:
        print("  ✓ Schema already up-to-date — no migration needed.")


def init_db():
    """Create all tables (if missing) then migrate any pre-existing ones."""
    create_database_if_missing()

    from database.db_connection import get_connection
    conn = get_connection()
    cur  = conn.cursor()

    print("\nCreating tables…")
    for stmt in DDL_TABLES:
        name = [l.split()[-1] for l in stmt.splitlines()
                if "CREATE TABLE" in l][0]
        cur.execute(stmt)
        print(f"  ✓ {name}")

    conn.commit()

    print("\nChecking for missing columns…")
    migrate_tables(conn)

    conn.close()
    print("\n✅ All tables ready.\n")


def reset_run_data(conn):
    """
    Delete all pipeline output rows before a new run.
    Respects FK constraints by deleting in dependency order.
    """
    cur = conn.cursor()
    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    for tbl in ["trial_patient_responses", "clinical_trial_results",
                "drug_candidates", "patients", "molecules"]:
        cur.execute(f"DELETE FROM {tbl}")
    cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()


if __name__ == "__main__":
    init_db()