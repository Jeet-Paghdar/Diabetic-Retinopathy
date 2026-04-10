import sqlite3
import mysql.connector
import json
import os
import sys

# Add project root to path so we can import DB_CONFIG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from new_database import DB_CONFIG, SQLITE_PATH

def migrate_mysql_to_sqlite():
    print("=" * 60)
    print("  RetinaScan AI — Data Migration Utility")
    print("=" * 60)
    print(f" Source: MySQL ({DB_CONFIG['database']})")
    print(f" Target: SQLite ({os.path.basename(SQLITE_PATH)})")
    print("-" * 60)

    # 1. Connect to MySQL
    try:
        mysql_conn = mysql.connector.connect(**DB_CONFIG)
        mysql_cursor = mysql_conn.cursor(dictionary=True)
        print("[+] Connected to MySQL.")
    except Exception as e:
        print(f"[!] Error connecting to MySQL: {e}")
        return

    # 2. Connect to SQLite
    try:
        sqlite_conn = sqlite3.connect(SQLITE_PATH)
        sqlite_cursor = sqlite_conn.cursor()
        print("[+] Connected to SQLite.")
    except Exception as e:
        print(f"[!] Error connecting to SQLite: {e}")
        return

    # 3. Create table in SQLite if it doesn't exist
    sqlite_cursor.execute("""
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
            model_version       TEXT NOT NULL,
            risk_level          TEXT,
            scan_date           TIMESTAMP NOT NULL,
            notes               TEXT
        )
    """)

    # 4. Fetch from MySQL
    print("[*] Fetching records from MySQL...")
    mysql_cursor.execute("SELECT * FROM scans")
    rows = mysql_cursor.fetchall()
    
    if not rows:
        print("[!] No records found in MySQL scans table.")
        return

    print(f"[*] Found {len(rows)} records. Migrating...")

    # 5. Insert into SQLite
    # We clear existing data in the target to avoid primary key conflicts 
    # since this is a 'Sync' operation for deployment.
    sqlite_cursor.execute("DELETE FROM scans")
    
    insert_query = """
        INSERT INTO scans 
        (id, patient_name, patient_age, eye_side, grade, grade_name, 
         confidence, all_probabilities, gradcam_path, model_version, 
         risk_level, scan_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    count = 0
    for row in rows:
        # MySQL JSON field might need explicit string conversion for SQLite TEXT
        probs = row['all_probabilities']
        if not isinstance(probs, str) and probs is not None:
            probs = json.dumps(probs)

        vals = (
            row['id'], row['patient_name'], row['patient_age'], row['eye_side'],
            row['grade'], row['grade_name'], row['confidence'],
            probs, row['gradcam_path'], row['model_version'],
            row['risk_level'], row['scan_date'], row['notes']
        )
        sqlite_cursor.execute(insert_query, vals)
        count += 1

    sqlite_conn.commit()
    print(f"[+] Successfully migrated {count} records to SQLite.")
    print("-" * 60)
    print(f" SUCCESS: {os.path.basename(SQLITE_PATH)} is ready for deployment.")
    print("=" * 60)

    mysql_conn.close()
    sqlite_conn.close()

if __name__ == "__main__":
    migrate_mysql_to_sqlite()
