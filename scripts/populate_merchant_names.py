"""
Script to populate merchant_name column in raw_transactions table
Using data from Analyse_specifique/export_postgres_transactions_user_100_2.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from sqlalchemy import text
from db_service.session import get_db

CSV_FILE = "Analyse_specifique/export_postgres_transactions_user_100_2.csv"

def populate_merchant_names():
    """Populate merchant_name from CSV data"""
    db = next(get_db())

    try:
        print(f"Reading CSV file: {CSV_FILE}")

        # Read CSV file (try different encodings)
        merchant_map = {}
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(CSV_FILE, 'r', encoding=encoding) as f:
                    content = f.read()
                    if content.startswith('\ufeff'):
                        content = content[1:]

                    lines = content.strip().split('\n')
                    reader = csv.DictReader(lines, delimiter=';')
                    break
            except UnicodeDecodeError:
                if encoding == encodings[-1]:
                    raise
                continue

        print(f"Successfully read CSV with encoding: {encoding}")

        for row in reader:
            tx_id = row.get('id')
            merchant = row.get('marchand', '').strip()

            if tx_id and merchant:
                merchant_map[int(tx_id)] = merchant

        print(f"Found {len(merchant_map)} transactions with merchant data")

        # Update transactions in batches
        updated_count = 0
        batch_size = 100

        for i in range(0, len(merchant_map), batch_size):
            batch_ids = list(merchant_map.keys())[i:i+batch_size]

            # Build CASE statement for batch update
            cases = []
            for tx_id in batch_ids:
                merchant = merchant_map[tx_id].replace("'", "''")  # Escape quotes
                cases.append(f"WHEN {tx_id} THEN '{merchant}'")

            case_statement = "\n            ".join(cases)

            update_query = text(f"""
                UPDATE raw_transactions
                SET merchant_name = CASE id
                    {case_statement}
                END
                WHERE id IN ({','.join(map(str, batch_ids))})
            """)

            result = db.execute(update_query)
            updated_count += result.rowcount

            if (i + batch_size) % 500 == 0:
                print(f"  Updated {updated_count} transactions...")

        db.commit()

        print(f"\n[OK] Successfully updated {updated_count} transactions with merchant names")

        # Show some examples
        print("\nExamples of updated transactions:")
        examples_query = text("""
            SELECT id, clean_description, merchant_name, amount
            FROM raw_transactions
            WHERE merchant_name IS NOT NULL
              AND merchant_name != 'inconnu'
            LIMIT 10
        """)

        examples = db.execute(examples_query).fetchall()
        for ex in examples:
            print(f"  ID {ex.id}: {ex.clean_description} -> {ex.merchant_name} ({ex.amount} EUR)")

        # Show statistics
        stats_query = text("""
            SELECT
                COUNT(*) as total,
                COUNT(merchant_name) as with_merchant,
                COUNT(CASE WHEN merchant_name = 'inconnu' THEN 1 END) as unknown_merchant,
                COUNT(CASE WHEN merchant_name IS NOT NULL AND merchant_name != 'inconnu' THEN 1 END) as known_merchant
            FROM raw_transactions
            WHERE user_id = 100
        """)

        stats = db.execute(stats_query).fetchone()
        print(f"\nStatistics for user 100:")
        print(f"  Total transactions: {stats.total}")
        print(f"  With merchant_name: {stats.with_merchant}")
        print(f"  Known merchants: {stats.known_merchant}")
        print(f"  Unknown merchants: {stats.unknown_merchant}")

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Error populating merchant names: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    populate_merchant_names()
