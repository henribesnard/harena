"""
Script pour supprimer toutes les conversations de la base de données
"""
import os
from sqlalchemy import create_engine, text

# Database URL - utiliser la variable d'environnement ou la valeur par défaut
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://harena_user:harena_password@localhost:5432/harena_db')

def clear_all_conversations():
    """Supprime toutes les conversations et leurs tours"""
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Delete all conversation turns first (FK constraint)
        result_turns = conn.execute(text('DELETE FROM conversation_turns'))
        print(f'OK Supprime {result_turns.rowcount} tours de conversation')

        # Then delete all conversations
        result_conv = conn.execute(text('DELETE FROM conversations'))
        print(f'OK Supprime {result_conv.rowcount} conversations')

        conn.commit()
        print('OK Base de donnees nettoyee avec succes')

if __name__ == "__main__":
    print("ATTENTION: Ce script va supprimer TOUTES les conversations!")
    response = input("Voulez-vous continuer? (oui/non): ")

    if response.lower() in ['oui', 'o', 'yes', 'y']:
        clear_all_conversations()
    else:
        print("Operation annulee")
