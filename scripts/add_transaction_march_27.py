"""
Script to add the missing transaction for March 27, 2025
27.03.2025 Virement Cat Amania VIREMENT-SALAIRE-MARS-25-27710 2,731.69€
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sqlalchemy import text
from db_service.session import get_db

def add_transaction():
    db = next(get_db())

    try:
        # Check if transaction already exists with this description
        check_query = text("""
            SELECT id, clean_description, amount
            FROM raw_transactions
            WHERE user_id = 100
              AND clean_description LIKE '%VIREMENT-SALAIRE-MARS-25-27710%'
        """)
        existing = db.execute(check_query).fetchone()

        if existing:
            print(f"Transaction already exists: ID={existing.id}, Description={existing.clean_description}, Amount={existing.amount}")
            return

        # Find a unique bridge_transaction_id (use a high number that won't conflict)
        max_id_query = text("SELECT MAX(bridge_transaction_id) FROM raw_transactions")
        max_id = db.execute(max_id_query).scalar() or 0
        new_bridge_id = max_id + 1

        print(f"Using bridge_transaction_id: {new_bridge_id}")

        # Insert the transaction
        insert_query = text("""
            INSERT INTO raw_transactions (
                bridge_transaction_id,
                account_id,
                user_id,
                clean_description,
                provider_description,
                amount,
                date,
                transaction_date,
                currency_code,
                category_id,
                operation_type,
                merchant_name,
                deleted,
                future
            ) VALUES (
                :bridge_transaction_id,
                :account_id,
                :user_id,
                :clean_description,
                :provider_description,
                :amount,
                :date,
                :transaction_date,
                :currency_code,
                :category_id,
                :operation_type,
                :merchant_name,
                :deleted,
                :future
            )
            RETURNING id, clean_description, amount
        """)

        result = db.execute(insert_query, {
            "bridge_transaction_id": new_bridge_id,
            "account_id": 262,  # Compte Chèque
            "user_id": 100,
            "clean_description": "Virement Cat Amania",
            "provider_description": "VIREMENT-SALAIRE-MARS-25-27710",
            "amount": 2731.69,
            "date": datetime(2025, 3, 27),
            "transaction_date": datetime(2025, 3, 27),
            "currency_code": "EUR",
            "category_id": 47,  # Salaire
            "operation_type": "Virement",
            "merchant_name": "Catamania",
            "deleted": False,
            "future": False
        })

        db.commit()

        inserted = result.fetchone()
        print(f"[OK] Transaction inserted successfully!")
        print(f"   ID: {inserted.id}")
        print(f"   Description: {inserted.clean_description}")
        print(f"   Amount: {inserted.amount} EUR")

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Error inserting transaction: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    add_transaction()
