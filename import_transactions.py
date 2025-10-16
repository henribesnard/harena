"""
Script pour importer les transactions avec matching des dates.

Fichiers:
- export_postgres_transactions_user_100_2.csv : Données complètes (clean_description, amount, category, marchand, operation_type) SANS dates
- 20192020.csv, 2021.csv, etc. : Dates correctes avec descriptions originales

Stratégie de matching:
1. Normaliser les descriptions (supprimer espaces, ponctuation, minuscules)
2. Matcher par description normalisée + montant proche
3. Gérer les doublons (même description, dates multiples)
"""
import os
import sys
import pandas as pd
import re
from datetime import datetime
from difflib import SequenceMatcher
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
    # Convertir en minuscules
    desc = str(desc).lower()
    # Supprimer les chiffres de dates (jj/mm, jj/mm/aaaa)
    desc = re.sub(r'\d{2}/\d{2}(/\d{2,4})?', '', desc)
    # Supprimer les montants
    desc = re.sub(r'\d+[,\.]\d+', '', desc)
    # Supprimer ponctuation et espaces multiples
    desc = re.sub(r'[^\w\s]', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip()

def similarity_score(str1, str2):
    """Calculer le score de similarité entre deux chaînes"""
    return SequenceMatcher(None, str1, str2).ratio()

def parse_date(date_str):
    """Parser une date depuis différents formats"""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Formats possibles
    formats = [
        '%d/%m/%Y',  # 19/11/2019
        '%d.%m.%Y',  # 05.12.2020
        '%d-%m-%Y',  # 19-11-2019
        '%Y-%m-%d',  # 2019-11-19
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

def load_date_files():
    """Charger tous les fichiers de dates"""
    date_files = [
        'analyse_specifique/20192020.csv',
        'analyse_specifique/2021.csv',
        'analyse_specifique/2022.csv',
        'analyse_specifique/2023.csv',
        'analyse_specifique/2024.csv',
        'analyse_specifique/2025.csv',
    ]

    all_dates = []

    for file_path in date_files:
        try:
            print(f"Chargement {file_path}...")
            # Essayer différents encodages
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

            # Colonnes attendues: Date, Libellé, Débit, Crédit
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

                # Montant = crédit - débit
                amount = credit - debit if credit > 0 else -debit

                all_dates.append({
                    'date': date,
                    'description': libelle,
                    'normalized_desc': normalize_description(libelle),
                    'amount': amount
                })

            print(f"  OK: {len(df)} lignes")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"Total dates chargees: {len(all_dates)}")
    return all_dates

def match_transaction_date(trans, date_records):
    """
    Matcher une transaction avec une date.
    Retourne la meilleure date correspondante.
    """
    trans_norm = normalize_description(trans['clean_description'])
    trans_amount = trans['amount']

    best_match = None
    best_score = 0.0

    for date_rec in date_records:
        # Vérifier similarité de description
        desc_score = similarity_score(trans_norm, date_rec['normalized_desc'])

        # Vérifier similarité de montant (tolérance 5%)
        amount_diff = abs(trans_amount - date_rec['amount'])
        amount_tolerance = abs(trans_amount) * 0.05
        amount_match = amount_diff <= max(amount_tolerance, 0.5)

        # Score combiné: 70% description + 30% montant
        if amount_match:
            score = desc_score * 0.7 + 0.3
        else:
            score = desc_score * 0.7

        # Mise à jour du meilleur match
        if score > best_score and desc_score >= 0.6:  # Seuil minimum de similarité
            best_score = score
            best_match = date_rec

    return best_match

def import_data():
    """Import principal"""
    session = SessionLocal()

    try:
        print("=== IMPORT DES TRANSACTIONS AVEC DATES ===\n")

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

        # 2. Charger les fichiers de dates
        print("\n2. Chargement des fichiers de dates...")
        date_records = load_date_files()

        if len(date_records) == 0:
            print("[ERROR] Aucune date chargee")
            return False

        # 3. Charger le fichier principal
        print("\n3. Chargement du fichier principal...")
        main_file = 'analyse_specifique/export_postgres_transactions_user_100_2.csv'

        # Essayer différents encodages
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

        # 4. Créer les catégories
        print("\n4. Creation des categories...")

        # Extraire toutes les catégories uniques
        categories_data = df_main[['category_name']].drop_duplicates()

        # Créer un mapping de groupes (à personnaliser selon vos besoins)
        category_groups = {
            'achats en ligne': 'Achats',
            'alimentation': 'Alimentation',
            'shopping': 'Achats',
            'restaurants': 'Alimentation',
            'transports': 'Transports',
            'logement': 'Logement',
            'sante': 'Sante',
            'loisirs': 'Loisirs',
            'services': 'Services',
            'autres': 'Autres',
        }

        # Créer les groupes de catégories
        group_map = {}
        for group_name in set(category_groups.values()):
            existing_group = session.query(CategoryGroup).filter(
                CategoryGroup.group_name == group_name
            ).first()

            if not existing_group:
                group = CategoryGroup(
                    group_name=group_name,
                    description=f"Groupe {group_name}",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(group)
                session.flush()
                group_map[group_name] = group.group_id
            else:
                group_map[group_name] = existing_group.group_id

        print(f"  OK: {len(group_map)} groupes crees")

        # Créer les catégories
        category_map = {}
        cat_id = 1

        for _, row in categories_data.iterrows():
            cat_name = row['category_name']
            if pd.isna(cat_name):
                cat_name = 'autres'

            # Déterminer le groupe
            group_name = category_groups.get(cat_name.lower(), 'Autres')
            group_id = group_map[group_name]

            existing_cat = session.query(Category).filter(
                Category.category_name == cat_name
            ).first()

            if not existing_cat:
                category = Category(
                    category_id=cat_id,
                    category_name=cat_name,
                    group_id=group_id,
                    description=f"Categorie {cat_name}",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(category)
                category_map[cat_name] = cat_id
                cat_id += 1
            else:
                category_map[cat_name] = existing_cat.category_id

        session.commit()
        print(f"  OK: {len(category_map)} categories creees")

        # 5. Matcher et insérer les transactions (OPTIMISE)
        print("\n5. Matching et insertion des transactions...")

        # Vérifier combien sont déjà insérées
        existing_count = session.query(RawTransaction).filter(
            RawTransaction.user_id == user.id
        ).count()

        print(f"  Transactions deja inserees: {existing_count}")

        matched = 0
        unmatched = 0
        inserted = 0
        skipped = 0

        # Pre-normaliser toutes les descriptions pour accélérer le matching
        print("  Preprocessing des descriptions...")
        for rec in date_records:
            if 'normalized_desc' not in rec or not rec['normalized_desc']:
                rec['normalized_desc'] = normalize_description(rec['description'])

        print("  Insertion des transactions...")
        for idx, row in df_main.iterrows():
            # Skip si déjà inséré
            if idx < existing_count:
                skipped += 1
                continue

            # Matcher avec une date (optimisé: seuil plus bas pour accélérer)
            date_match = match_transaction_date(row, date_records)

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
                bridge_transaction_id=10000000 + idx,  # ID unique
                account_id=account.id,
                user_id=user.id,
                clean_description=str(row['clean_description'])[:500],  # Limiter la longueur
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

            # Commit par batch de 50 (plus fréquent)
            if inserted % 50 == 0:
                session.commit()
                print(f"  Progress: {inserted}/{len(df_main) - existing_count} nouvelles transactions inserees")

        # Commit final
        session.commit()

        print(f"\n=== RESULTAT ===")
        print(f"Transactions matchees avec dates: {matched}")
        print(f"Transactions non matchees (date par defaut): {unmatched}")
        print(f"Total transactions inserees: {inserted}")
        print(f"Groupes de categories: {len(group_map)}")
        print(f"Categories: {len(category_map)}")

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
