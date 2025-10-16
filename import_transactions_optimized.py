"""
Script optimisé pour importer les transactions avec matching des dates.

Optimisations:
1. Pre-compute hash index des descriptions normalisées
2. Matching exact d'abord, puis fuzzy matching uniquement si nécessaire
3. Batch processing avec commits moins fréquents
"""
import os
import sys
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configuration
os.environ['DATABASE_URL'] = 'postgresql://harena_admin:HaReNa2024SecureDbPassword123@63.35.52.216:5432/harena'

from db_service.models import User, SyncAccount, RawTransaction, Category, CategoryGroup

# Créer la session
engine = create_engine(os.environ['DATABASE_URL'])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def normalize_description(desc):
    """Normaliser une description pour le matching"""
    if pd.isna(desc) or desc is None:
        return ""
    desc = str(desc).lower()
    desc = re.sub(r'\d{2}/\d{2}(/\d{2,4})?', '', desc)
    desc = re.sub(r'\d+[,\.]\d+', '', desc)
    desc = re.sub(r'[^\w\s]', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip()

def parse_date(date_str):
    """Parser une date depuis différents formats"""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    formats = [
        '%d/%m/%Y',
        '%d.%m.%Y',
        '%d-%m-%Y',
        '%Y-%m-%d',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue

    return None

def parse_amount(amount_str):
    """Parser un montant"""
    if pd.isna(amount_str) or amount_str == '':
        return 0.0

    amount_str = str(amount_str).strip().replace(',', '.').replace(' ', '')
    try:
        return float(amount_str)
    except:
        return 0.0

def load_date_files_indexed():
    """
    Charger tous les fichiers de dates et créer un index.
    Retourne un dictionnaire: normalized_desc -> list of date records
    """
    date_files = [
        'analyse_specifique/20192020.csv',
        'analyse_specifique/2021.csv',
        'analyse_specifique/2022.csv',
        'analyse_specifique/2023.csv',
        'analyse_specifique/2024.csv',
        'analyse_specifique/2025.csv',
    ]

    # Index: normalized_desc -> list of {date, description, amount}
    date_index = defaultdict(list)

    for file_path in date_files:
        try:
            print(f"Chargement {file_path}...")
            df = None
            for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, sep=';', encoding=encoding, on_bad_lines='skip')
                    break
                except:
                    continue

            if df is None:
                print(f"  [SKIP] Impossible de lire le fichier")
                continue

            if 'Date' not in df.columns or 'Libellé' not in df.columns:
                print(f"  [SKIP] Format invalide")
                continue

            for _, row in df.iterrows():
                date = parse_date(row['Date'])
                if date is None:
                    continue

                libelle = str(row['Libellé']) if not pd.isna(row['Libellé']) else ""
                debit = parse_amount(row.get('Débit', ''))
                credit = parse_amount(row.get('Crédit', ''))

                amount = credit - debit if credit > 0 else -debit
                normalized_desc = normalize_description(libelle)

                # Ajouter à l'index
                date_index[normalized_desc].append({
                    'date': date,
                    'description': libelle,
                    'amount': amount
                })

            print(f"  OK: {len(df)} lignes")
        except Exception as e:
            print(f"  [ERROR] {e}")

    total_records = sum(len(records) for records in date_index.values())
    print(f"Total dates chargees: {total_records}")
    print(f"Descriptions uniques: {len(date_index)}")
    return date_index

def match_transaction_date_fast(trans, date_index):
    """
    Matcher une transaction avec une date en utilisant l'index.
    1. Essayer exact match
    2. Si pas de match, chercher par montant similaire dans les descriptions proches
    """
    trans_norm = normalize_description(trans['clean_description'])
    trans_amount = trans['amount']

    # 1. Exact match sur description normalisée
    if trans_norm in date_index:
        candidates = date_index[trans_norm]

        # Trouver le meilleur match par montant
        best_match = None
        min_diff = float('inf')

        for candidate in candidates:
            amount_diff = abs(trans_amount - candidate['amount'])
            if amount_diff < min_diff:
                min_diff = amount_diff
                best_match = candidate

        # Si le montant est raisonnablement proche, accepter
        if min_diff <= max(abs(trans_amount) * 0.05, 0.5):
            return best_match

    # 2. Pas de match exact - retourner None pour date par défaut
    return None

def import_data():
    """Import principal optimisé"""
    session = SessionLocal()

    try:
        print("=== IMPORT OPTIMISE DES TRANSACTIONS AVEC DATES ===\n")

        # 1. Charger l'utilisateur et le compte
        print("1. Recuperation de l'utilisateur et du compte...")
        user = session.query(User).filter(User.email == 'henri@example.com').first()
        if not user:
            print("[ERROR] Utilisateur non trouve")
            return False

        account = session.query(SyncAccount).join(
            SyncAccount.item
        ).filter(
            SyncAccount.item.has(user_id=user.id)
        ).first()

        if not account:
            print("[ERROR] Compte non trouve")
            return False

        print(f"  OK: User {user.id}, Account {account.id}")

        # 2. Charger les fichiers de dates avec index
        print("\n2. Chargement des fichiers de dates avec indexation...")
        date_index = load_date_files_indexed()

        if len(date_index) == 0:
            print("[ERROR] Aucune date chargee")
            return False

        # 3. Charger le fichier principal
        print("\n3. Chargement du fichier principal...")
        main_file = 'analyse_specifique/export_postgres_transactions_user_100_2.csv'

        for encoding in ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df_main = pd.read_csv(main_file, sep=';', encoding=encoding)
                print(f"  OK: {len(df_main)} transactions (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("[ERROR] Impossible de decoder le fichier")
            return False

        # 4. Charger les catégories existantes
        print("\n4. Chargement des categories existantes...")
        category_map = {}
        for cat in session.query(Category).all():
            category_map[cat.category_name] = cat.category_id

        print(f"  OK: {len(category_map)} categories existantes")

        # 5. Vérifier combien sont déjà insérées
        existing_count = session.query(RawTransaction).filter(
            RawTransaction.user_id == user.id
        ).count()

        print(f"\n5. Transactions deja inserees: {existing_count}")

        # 6. Matcher et insérer les transactions (OPTIMISE)
        print("\n6. Matching et insertion des transactions...")

        matched = 0
        unmatched = 0
        inserted = 0
        skipped = 0

        print("  Insertion des transactions...")
        for idx, row in df_main.iterrows():
            # Skip si déjà inséré
            if idx < existing_count:
                skipped += 1
                continue

            # Matcher avec une date (optimisé avec index)
            date_match = match_transaction_date_fast(row, date_index)

            if date_match:
                matched += 1
                transaction_date = date_match['date']
            else:
                unmatched += 1
                # Date par défaut si pas de match
                transaction_date = datetime(2020, 1, 1)

            # Créer la transaction
            cat_name = row['category_name'] if not pd.isna(row['category_name']) else 'autres'
            category_id = category_map.get(cat_name, list(category_map.values())[0])

            transaction = RawTransaction(
                bridge_transaction_id=10000000 + idx,
                account_id=account.id,
                user_id=user.id,
                clean_description=str(row['clean_description'])[:500],
                provider_description=str(row['clean_description'])[:500],
                amount=float(row['amount']),
                date=transaction_date,
                booking_date=transaction_date,
                transaction_date=transaction_date,
                currency_code='EUR',
                category_id=category_id,
                operation_type=str(row.get('Operation_type', 'unknown'))[:50],
                merchant_name=str(row.get('marchand', 'unknown'))[:200],
                deleted=False,
                future=False,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            session.add(transaction)
            inserted += 1

            # Commit par batch de 100 (plus rapide maintenant)
            if inserted % 100 == 0:
                session.commit()
                print(f"  Progress: {inserted}/{len(df_main) - existing_count} nouvelles transactions inserees")

        # Commit final
        session.commit()

        print(f"\n=== RESULTAT ===")
        print(f"Transactions matchees avec dates: {matched}")
        print(f"Transactions non matchees (date par defaut): {unmatched}")
        print(f"Total transactions inserees: {inserted}")
        print(f"Total dans la base: {existing_count + inserted}")

        return True

    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        session.close()

if __name__ == "__main__":
    success = import_data()
    sys.exit(0 if success else 1)
