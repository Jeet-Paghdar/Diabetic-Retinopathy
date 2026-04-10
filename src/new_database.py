
"""
new_database.py
===============
Extended MySQL database module for RetinaScan AI — 82% EfficientNetB4 model.

Builds on database.py with the following additions:
  - New 'scans' table schema with gradcam_path, model_version, all_probs columns
  - model_version tracking (records which model made each prediction)
  - GradCAM overlay path stored per scan record
  - Batch statistics by model version (useful for comparing old vs 82% model)
  - JSON storage for per-class probabilities

Table: scans (new, separate from the old 'patients' table)
  id, patient_name, patient_age, eye_side,
  grade, grade_name, confidence, all_probabilities (JSON),
  gradcam_path, model_version, scan_date, notes

Usage:
    from src.new_database import setup_new_database, insert_new_scan, get_all_new_scans
"""

import json
import datetime
import os
import sqlite3
import mysql.connector
from mysql.connector import Error

# ── Database Configuration ─────────────────────────────────────────────────────
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 3306,
    'user'    : 'root',
    'password': 'jeet123',
    'database': 'retinascan_db',
}

SQLITE_PATH = os.path.join(os.getcwd(), 'retinascan_ai.db')
_USE_SQLITE = False

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_VERSION_82PCT = 'EfficientNetB4_v82pct'   # Tag for the new 82% model
MODEL_VERSION_OLD   = 'EfficientNetB3_v1'        # Tag for the old model

GRADE_NAMES = [
    'No DR',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR',
]

RISK_LEVELS = {
    0: 'Low',
    1: 'Low-Medium',
    2: 'Medium',
    3: 'High',
    4: 'Critical',
}


# ── Connection ─────────────────────────────────────────────────────────────────
def get_connection():
    """Return a live MySQL connection or fall back to SQLite if MySQL fails."""
    global _USE_SQLITE
    
    # 1. Try MySQL
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        _USE_SQLITE = False
        return conn
    except Exception:
        # 2. Fall back to SQLite
        if not _USE_SQLITE:
            print("[DB] MySQL unavailable, switching to SQLite fallback.")
        _USE_SQLITE = True
        try:
            return sqlite3.connect(SQLITE_PATH)
        except Exception as e:
            print(f"[DB] CRITICAL: Both MySQL and SQLite failed! {e}")
            return None

def get_query(query_str: str) -> str:
    """Adapt query syntax between MySQL (%s) and SQLite (?)."""
    if _USE_SQLITE:
        return query_str.replace('%s', '?')
    return query_str


# ── Setup ──────────────────────────────────────────────────────────────────────
def setup_new_database():
    """Create database and 'scans' table (MySQL or SQLite)."""
    conn = get_connection()
    if not conn: return
    
    try:
        cursor = conn.cursor()
        
        if not _USE_SQLITE:
            # MySQL Setup
            cursor.execute("CREATE DATABASE IF NOT EXISTS retinascan_db")
            cursor.execute("USE retinascan_db")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id                  INT AUTO_INCREMENT PRIMARY KEY,
                    patient_name        VARCHAR(100) NOT NULL,
                    patient_age         INT,
                    eye_side            VARCHAR(20),
                    grade               INT NOT NULL,
                    grade_name          VARCHAR(50) NOT NULL,
                    confidence          FLOAT NOT NULL,
                    all_probabilities   JSON,
                    gradcam_path        VARCHAR(500),
                    model_version       VARCHAR(100) NOT NULL DEFAULT 'EfficientNetB4_v82pct',
                    risk_level          VARCHAR(30),
                    scan_date           DATETIME NOT NULL,
                    notes               TEXT,
                    INDEX idx_patient   (patient_name),
                    INDEX idx_grade     (grade),
                    INDEX idx_model     (model_version),
                    INDEX idx_scan_date (scan_date)
                )
            """)
        else:
            # SQLite Setup
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name        TEXT NOT NULL,
                    patient_age         INTEGER,
                    eye_side            TEXT,
                    grade               INTEGER NOT NULL,
                    grade_name          TEXT NOT NULL,
                    confidence          REAL NOT NULL,
                    all_probabilities   TEXT, 
                    gradcam_path        TEXT,
                    model_version       TEXT NOT NULL DEFAULT 'EfficientNetB4_v82pct',
                    risk_level          TEXT,
                    scan_date           TIMESTAMP NOT NULL,
                    notes               TEXT
                )
            """)

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] Setup complete ({'SQLite' if _USE_SQLITE else 'MySQL'})")
    except Exception as e:
        print(f"[DB] Setup error: {e}")


# ── CREATE ─────────────────────────────────────────────────────────────────────
def insert_new_scan(
    patient_name: str,
    patient_age: int,
    eye_side: str,
    grade: int,
    confidence: float,
    all_probabilities: list = None,
    gradcam_path: str = None,
    model_version: str = MODEL_VERSION_82PCT,
    notes: str = None,
) -> int:
    """
    Insert a new scan record into the 'scans' table.

    Args:
        patient_name     : Patient's full name
        patient_age      : Age in years
        eye_side         : 'Left Eye', 'Right Eye', or 'Both'
        grade            : DR grade 0–4
        confidence       : Model's confidence for the predicted class (0–1)
        all_probabilities: List of 5 softmax scores (one per class)
        gradcam_path     : Path to the saved Grad-CAM PNG overlay
        model_version    : Identifier for the model used (default: 82% model)
        notes            : Optional clinical notes

    Returns:
        Inserted record ID (int) or None on failure
    """
    conn = get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        probs_json = json.dumps(all_probabilities) if all_probabilities else None

        query = """
            INSERT INTO scans
                (patient_name, patient_age, eye_side,
                 grade, grade_name, confidence, all_probabilities,
                 gradcam_path, model_version, risk_level,
                 scan_date, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            patient_name,
            patient_age,
            eye_side,
            grade,
            GRADE_NAMES[grade],
            round(confidence, 6),
            probs_json,
            gradcam_path,
            model_version,
            RISK_LEVELS[grade],
            datetime.datetime.now(),
            notes,
        )

        cursor.execute(get_query(query), values)
        conn.commit()
        record_id = cursor.lastrowid

        print(
            f"[DB] Inserted scan #{record_id} | "
            f"Patient: {patient_name} | "
            f"Grade: {GRADE_NAMES[grade]} ({confidence*100:.1f}%) | "
            f"Model: {model_version}"
        )
        return record_id

    except Error as e:
        print(f"[DB] Insert error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def insert_scan_from_result(
    patient_name: str,
    patient_age: int,
    eye_side: str,
    prediction_result: dict,
    notes: str = None,
) -> int:
    """
    Convenience wrapper: insert from the dict returned by predict_with_gradcam().

    Args:
        patient_name     : Patient's full name
        patient_age      : Age in years
        eye_side         : 'Left Eye', 'Right Eye', or 'Both'
        prediction_result: Dict from new_model_utils.predict_with_gradcam()
        notes            : Optional clinical notes

    Returns:
        Inserted record ID or None
    """
    return insert_new_scan(
        patient_name=patient_name,
        patient_age=patient_age,
        eye_side=eye_side,
        grade=prediction_result['grade'],
        confidence=prediction_result['confidence'],
        all_probabilities=prediction_result.get('all_probabilities'),
        gradcam_path=prediction_result.get('gradcam_saved_path'),
        model_version=MODEL_VERSION_82PCT,
        notes=notes,
    )


# ── READ ───────────────────────────────────────────────────────────────────────
def get_all_new_scans(model_version: str = None) -> list:
    """
    Fetch all scan records, optionally filtered by model version.

    Args:
        model_version: If given, filter to only this model's records

    Returns:
        List of tuples (all columns in 'scans' table)
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        if model_version:
            cursor.execute(
                get_query("SELECT * FROM scans WHERE model_version = %s ORDER BY scan_date DESC"),
                (model_version,),
            )
        else:
            cursor.execute(get_query("SELECT * FROM scans ORDER BY scan_date DESC"))
        return cursor.fetchall()
    except Error as e:
        print(f"[DB] Read error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_new_scan_by_id(record_id: int) -> tuple:
    """Fetch a single scan record by primary key."""
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(get_query("SELECT * FROM scans WHERE id = %s"), (record_id,))
        return cursor.fetchone()
    except Error as e:
        print(f"[DB] Read error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def search_new_scans(patient_name: str) -> list:
    """Search scan records by patient name (case-insensitive partial match)."""
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(
            get_query("SELECT * FROM scans WHERE patient_name LIKE %s ORDER BY scan_date DESC"),
            (f"%{patient_name}%",),
        )
        return cursor.fetchall()
    except Error as e:
        print(f"[DB] Search error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# Alias so notebook import `get_scans_by_name_new` works
get_scans_by_name_new = search_new_scans


def get_scan_probabilities(record_id: int) -> list:
    """
    Retrieve the stored all_probabilities JSON for a record.

    Returns:
        List of 5 floats (softmax scores) or [] on failure
    """
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(
            get_query("SELECT all_probabilities FROM scans WHERE id = %s"),
            (record_id,),
        )
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []
    except Error as e:
        print(f"[DB] Probabilities fetch error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ── STATISTICS ─────────────────────────────────────────────────────────────────
def get_new_stats(model_version: str = None) -> dict:
    """
    Compute aggregate statistics from the 'scans' table.

    Args:
        model_version: If given, filter to only this model's records

    Returns:
        Dict with: total, dr_detected, severe_or_worse,
                   avg_confidence, grade_distribution, model_breakdown
    """
    conn = get_connection()
    if not conn:
        return {}
    try:
        cursor = conn.cursor()
        where = f"WHERE model_version = '{model_version}'" if model_version else ""

        cursor.execute(get_query(f"SELECT COUNT(*) FROM scans {where}"))
        total = cursor.fetchone()[0]

        cursor.execute(get_query(f"SELECT COUNT(*) FROM scans {where + ' AND' if where else 'WHERE'} grade > 0"))
        dr_detected = cursor.fetchone()[0]

        cursor.execute(get_query(f"SELECT COUNT(*) FROM scans {where + ' AND' if where else 'WHERE'} grade >= 3"))
        severe = cursor.fetchone()[0]

        cursor.execute(get_query(f"SELECT AVG(confidence) FROM scans {where}"))
        avg_conf = cursor.fetchone()[0]

        # Grade distribution
        grade_dist = {}
        for g in range(5):
            cursor.execute(
                get_query(f"SELECT COUNT(*) FROM scans {where + ' AND' if where else 'WHERE'} grade = {g}")
            )
            grade_dist[g] = cursor.fetchone()[0]

        # Model breakdown (when no filter)
        if not model_version:
            cursor.execute(
                get_query("SELECT model_version, COUNT(*) FROM scans GROUP BY model_version")
            )
            model_breakdown = dict(cursor.fetchall())
        else:
            model_breakdown = {model_version: total}

        return {
            'total'           : total,
            'dr_detected'     : dr_detected,
            'severe_or_worse' : severe,
            'avg_confidence'  : round(avg_conf, 6) if avg_conf else 0.0,
            'grade_distribution': grade_dist,
            'model_breakdown' : model_breakdown,
        }

    except Error as e:
        print(f"[DB] Stats error: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()


# ── UPDATE ─────────────────────────────────────────────────────────────────────
def update_new_notes(record_id: int, notes: str) -> bool:
    """Update the notes field for an existing scan record."""
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            get_query("UPDATE scans SET notes = %s WHERE id = %s"),
            (notes, record_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            print(f"[DB] Record #{record_id} notes updated.")
        else:
            print(f"[DB] Record #{record_id} not found.")
        return updated
    except Error as e:
        print(f"[DB] Update error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def update_gradcam_path(record_id: int, gradcam_path: str) -> bool:
    """Update the Grad-CAM overlay path for an existing scan record."""
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            get_query("UPDATE scans SET gradcam_path = %s WHERE id = %s"),
            (gradcam_path, record_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            print(f"[DB] Record #{record_id} GradCAM path updated → {gradcam_path}")
        else:
            print(f"[DB] Record #{record_id} not found.")
        return updated
    except Error as e:
        print(f"[DB] Update error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── DELETE ─────────────────────────────────────────────────────────────────────
def delete_new_scan(record_id: int) -> bool:
    """Delete a single scan record by ID."""
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(get_query("DELETE FROM scans WHERE id = %s"), (record_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            print(f"[DB] Record #{record_id} deleted.")
        else:
            print(f"[DB] Record #{record_id} not found.")
        return deleted
    except Error as e:
        print(f"[DB] Delete error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── DISPLAY ────────────────────────────────────────────────────────────────────
def print_new_scans(model_version: str = None):
    """Pretty-print all scan records (optionally filtered by model version)."""
    rows = get_all_new_scans(model_version=model_version)
    if not rows:
        print("[DB] No records found.")
        return

    print("\n" + "=" * 110)
    print(
        f"{'ID':<5} {'Name':<20} {'Age':<4} {'Eye':<12} {'Grade':<20} "
        f"{'Conf%':<7} {'Risk':<12} {'Model':<28} {'Date'}"
    )
    print("=" * 110)
    for row in rows:
        (rid, name, age, eye, grade, grade_name, conf,
         probs_json, cam_path, model_ver, risk, date, notes) = row
        print(
            f"{rid:<5} {str(name):<20} {str(age):<4} {str(eye):<12} "
            f"{str(grade_name):<20} {conf*100:<7.1f} {str(risk):<12} "
            f"{str(model_ver):<28} {date}"
        )
    print("=" * 110)
    print(f"  Total records: {len(rows)}\n")


def print_new_stats(model_version: str = None):
    """Pretty-print database statistics."""
    stats = get_new_stats(model_version=model_version)
    if not stats:
        print("[DB] Could not retrieve statistics.")
        return

    label = f" ({model_version})" if model_version else " (all models)"
    print("\n" + "=" * 55)
    print(f"  RetinaScan AI — Statistics{label}")
    print("=" * 55)
    print(f"  Total scans         : {stats['total']}")
    print(f"  DR detected         : {stats['dr_detected']}")
    print(f"  Severe / Critical   : {stats['severe_or_worse']}")
    print(f"  Avg confidence      : {stats['avg_confidence']*100:.1f}%")
    print("\n  Grade Distribution:")
    print("-" * 45)
    for g, count in stats['grade_distribution'].items():
        pct = count / max(stats['total'], 1) * 100
        print(f"    Grade {g} ({GRADE_NAMES[g]:<16}): {count:4d}  ({pct:5.1f}%)")
    print("\n  By Model Version:")
    print("-" * 45)
    for mv, cnt in stats['model_breakdown'].items():
        print(f"    {mv:<30}: {cnt:4d}")
    print("=" * 55)


# ── Interactive CLI (for standalone testing) ────────────────────────────────────
if __name__ == '__main__':
    print("\nRetinaScan AI — New Database Module (82% EfficientNetB4)")
    print("=" * 55)
    setup_new_database()

    while True:
        print("\nOptions (new 'scans' table):")
        print("  1. Insert test scan")
        print("  2. View all records")
        print("  3. Search by name")
        print("  4. View statistics")
        print("  5. View by model version")
        print("  6. Update notes")
        print("  7. Delete scan")
        print("  8. Exit")

        choice = input("\nChoice (1-8): ").strip()

        if choice == '1':
            name  = input("Patient name         : ").strip()
            age   = int(input("Patient age          : ").strip())
            eye   = input("Eye side             : ").strip()
            grade = int(input("DR Grade (0-4)       : ").strip())
            conf  = float(input("Confidence (0-1)     : ").strip())
            notes = input("Notes (Enter=skip)   : ").strip()
            # Simulate all_probabilities
            probs = [0.0] * 5
            probs[grade] = conf
            insert_new_scan(name, age, eye, grade, conf, probs, notes=notes or None)

        elif choice == '2':
            print_new_scans()

        elif choice == '3':
            name = input("Search name: ").strip()
            rows = search_new_scans(name)
            print(f"\nFound {len(rows)} record(s).")
            for r in rows:
                print(f"  ID:{r[0]} | {r[1]} | Grade:{r[4]} ({r[5]}) | {r[6]*100:.1f}% | {r[11]}")

        elif choice == '4':
            print_new_stats()

        elif choice == '5':
            mv = input("Model version (e.g. EfficientNetB4_v82pct): ").strip()
            print_new_scans(model_version=mv)

        elif choice == '6':
            rid   = int(input("Record ID  : ").strip())
            notes = input("New notes  : ").strip()
            update_new_notes(rid, notes)

        elif choice == '7':
            rid = int(input("Record ID to delete: ").strip())
            confirm = input("Confirm? (yes/no): ").strip()
            if confirm.lower() == 'yes':
                delete_new_scan(rid)

        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")
