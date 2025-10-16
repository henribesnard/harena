"""
Script de backup Elasticsearch pour Harena
Sauvegarde complète de l'index transactions avant déploiement AWS
"""
import os
import sys
import json
from datetime import datetime
from elasticsearch import Elasticsearch

# Configuration
ES_HOST = "https://37r8v9zfzn:4o7ydjkcc8@fir-178893546.eu-west-1.bonsaisearch.net:443"
ES_INDEX = "harena_transactions-000001"
BACKUP_DIR = os.path.dirname(__file__)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def backup_elasticsearch():
    """Backup complet Elasticsearch"""
    print("="*80)
    print(f"BACKUP ELASTICSEARCH - {TIMESTAMP}")
    print("="*80)
    print()

    # Connexion
    print(f"Connexion a Elasticsearch: {ES_HOST}")
    es = Elasticsearch([ES_HOST])

    # Vérifier connexion
    if not es.ping():
        print("ERROR: Impossible de se connecter a Elasticsearch")
        return False

    print(f"OK Connexion etablie\n")

    # Obtenir les infos de l'index
    try:
        index_info = es.indices.get(index=ES_INDEX)
        print(f"Index: {ES_INDEX}")
        print(f"Parametres: {json.dumps(index_info[ES_INDEX]['settings'], indent=2)}\n")
    except Exception as e:
        print(f"ERROR recuperation infos index: {str(e)}")
        return False

    # Compter les documents
    try:
        count_result = es.count(index=ES_INDEX)
        doc_count = count_result['count']
        print(f"Documents a sauvegarder: {doc_count}\n")
    except Exception as e:
        print(f"ERROR comptage documents: {str(e)}")
        return False

    # Sauvegarder tous les documents
    print("Sauvegarde des documents...")
    try:
        # Utiliser scroll API pour récupérer tous les documents
        documents = []
        scroll_size = 1000

        # Première requête
        response = es.search(
            index=ES_INDEX,
            scroll='2m',
            size=scroll_size,
            body={"query": {"match_all": {}}}
        )

        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        documents.extend(hits)

        print(f"Batch 1: {len(hits)} documents")

        # Continuer avec scroll
        batch_num = 2
        while len(hits) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            documents.extend(hits)
            if hits:
                print(f"Batch {batch_num}: {len(hits)} documents")
                batch_num += 1

        # Nettoyer le scroll
        es.clear_scroll(scroll_id=scroll_id)

        print(f"\nTotal recupere: {len(documents)} documents")

        # Sauvegarder en JSON
        output_file = os.path.join(BACKUP_DIR, f"elasticsearch_transactions_{TIMESTAMP}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'index': ES_INDEX,
                'timestamp': datetime.now().isoformat(),
                'count': len(documents),
                'documents': documents
            }, f, ensure_ascii=False, indent=2)

        print(f"OK Sauvegarde: {output_file}")

        # Sauvegarder les mappings
        mappings_file = os.path.join(BACKUP_DIR, f"elasticsearch_mappings_{TIMESTAMP}.json")
        mappings = es.indices.get_mapping(index=ES_INDEX)
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)

        print(f"OK Mappings: {mappings_file}")

        # Sauvegarder les settings
        settings_file = os.path.join(BACKUP_DIR, f"elasticsearch_settings_{TIMESTAMP}.json")
        settings = es.indices.get_settings(index=ES_INDEX)
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

        print(f"OK Settings: {settings_file}")

    except Exception as e:
        print(f"ERROR sauvegarde documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Métadonnées
    metadata_file = os.path.join(BACKUP_DIR, f"elasticsearch_metadata_{TIMESTAMP}.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Backup Elasticsearch Harena\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Host: {ES_HOST}\n")
        f.write(f"Index: {ES_INDEX}\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"\nFichiers:\n")
        f.write(f"  - {output_file}\n")
        f.write(f"  - {mappings_file}\n")
        f.write(f"  - {settings_file}\n")

    print(f"OK Metadonnees: {metadata_file}")

    print()
    print("="*80)
    print(f"BACKUP COMPLETE: {len(documents)} documents sauvegardes")
    print(f"Repertoire: {BACKUP_DIR}")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = backup_elasticsearch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR FATALE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
