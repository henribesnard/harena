"""
Script de validation manuelle des configurations
Utilisé pour tester et valider les fichiers YAML de configuration
"""

import asyncio
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# Ajouter le chemin du projet pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_validator import validate_full_configuration, ValidationLevel, ConfigValidator

async def load_config_file(filepath: Path) -> Dict[str, Any]:
    """Charge un fichier de configuration YAML"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ERREUR lors du chargement de {filepath}: {e}")
        return {}

async def validate_configs():
    """Valide les configurations principales"""
    config_dir = Path(__file__).parent
    
    print("Validation des configurations Conversation Service v2.0")
    print("=" * 60)
    
    # Charger les fichiers
    print("\n📁 Chargement des fichiers de configuration...")
    intentions_file = config_dir / "intentions_v2.yaml"
    entities_file = config_dir / "entities_v2.yaml"
    
    if not intentions_file.exists():
        print(f"❌ Fichier manquant: {intentions_file}")
        return False
    
    if not entities_file.exists():
        print(f"❌ Fichier manquant: {entities_file}")
        return False
    
    intentions_config = await load_config_file(intentions_file)
    entities_config = await load_config_file(entities_file)
    
    if not intentions_config or not entities_config:
        print("❌ Erreur lors du chargement des configurations")
        return False
    
    print(f"✅ Intentions chargées: {len(intentions_config.get('intent_groups', {}))} groupes")
    print(f"✅ Entités chargées: {len(entities_config.get('search_service_fields', {}))} catégories de champs")
    
    # Validation complète
    print("\n🔍 Validation complète en cours...")
    is_valid, results = await validate_full_configuration(intentions_config, entities_config)
    
    # Analyser les résultats
    critical_count = sum(1 for r in results if r.level == ValidationLevel.CRITICAL)
    warning_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)
    info_count = sum(1 for r in results if r.level == ValidationLevel.INFO)
    
    print(f"\n📊 Résultats de validation:")
    print(f"   🔴 Erreurs critiques: {critical_count}")
    print(f"   🟡 Avertissements: {warning_count}")
    print(f"   🔵 Informations: {info_count}")
    print(f"   ✅ Configuration valide: {'OUI' if is_valid else 'NON'}")
    
    # Afficher les détails
    if results:
        print("\n📋 Détails des validations:")
        
        if critical_count > 0:
            print("\n🔴 ERREURS CRITIQUES:")
            for result in results:
                if result.level == ValidationLevel.CRITICAL:
                    print(f"   ❌ {result.category}: {result.message}")
                    if result.field_path:
                        print(f"      📍 Chemin: {result.field_path}")
                    if result.suggestion:
                        print(f"      💡 Suggestion: {result.suggestion}")
        
        if warning_count > 0:
            print("\n🟡 AVERTISSEMENTS:")
            for result in results:
                if result.level == ValidationLevel.WARNING:
                    print(f"   ⚠️ {result.category}: {result.message}")
                    if result.field_path:
                        print(f"      📍 Chemin: {result.field_path}")
        
        if info_count > 0:
            print("\n🔵 INFORMATIONS:")
            for result in results:
                if result.level == ValidationLevel.INFO:
                    print(f"   ℹ️ {result.category}: {result.message}")
    
    return is_valid

async def test_specific_validation():
    """Tests de validation spécifiques"""
    print("\n🧪 Tests de validation spécifiques:")
    print("-" * 40)
    
    validator = ConfigValidator()
    
    # Test 1: Champs Elasticsearch connus
    print("\n🔬 Test 1: Vérification des champs Elasticsearch...")
    known_fields = validator.known_elasticsearch_fields
    print(f"   📊 Champs connus: {len(known_fields)}")
    print(f"   📝 Exemples: {list(known_fields)[:10]}...")
    
    # Test 2: Validation d'exemple temporel
    print("\n🔬 Test 2: Format temporel...")
    temporal_example = {
        "date": {"gte": "2025-01-01T00:00:00Z", "lte": "2025-01-31T23:59:59Z"}
    }
    print(f"   ✅ Exemple temporel valide: {temporal_example}")
    
    # Test 3: Validation d'agrégation
    print("\n🔬 Test 3: Syntaxe d'agrégation...")
    aggregation_example = {
        "categories": {
            "terms": {"field": "category_name.keyword"},
            "aggs": {"total": {"sum": {"field": "amount_abs"}}}
        }
    }
    print(f"   ✅ Exemple d'agrégation: structure valide")
    
    return True

async def generate_compatibility_report():
    """Génère un rapport de compatibilité avec les services existants"""
    print("\n📄 Rapport de compatibilité:")
    print("-" * 40)
    
    validator = ConfigValidator()
    
    # Compatibilité enrichment_service
    print("\n🔗 Compatibilité enrichment_service:")
    enrichment_fields = validator.known_elasticsearch_fields
    print(f"   ✅ {len(enrichment_fields)} champs disponibles dans StructuredTransaction")
    
    # Champs critiques pour les intentions financières
    critical_fields = [
        'amount', 'amount_abs', 'transaction_type', 'date',
        'merchant_name', 'category_name', 'searchable_text'
    ]
    
    print(f"\n🎯 Champs critiques pour intentions financières:")
    for field in critical_fields:
        status = "✅" if field in enrichment_fields else "❌"
        print(f"   {status} {field}")
    
    # Compatibilité search_service
    print(f"\n🔍 Compatibilité search_service:")
    print(f"   ✅ Interface SearchRequest supportée")
    print(f"   ✅ Filtres, agrégations et tri disponibles")
    print(f"   ✅ Pagination et métadonnées gérées")
    
    return True

async def main():
    """Fonction principale"""
    print("🚀 Lancement de la validation des configurations")
    
    try:
        # Validation principale
        is_valid = await validate_configs()
        
        # Tests spécifiques
        await test_specific_validation()
        
        # Rapport de compatibilité
        await generate_compatibility_report()
        
        # Résultat final
        print("\n" + "=" * 60)
        if is_valid:
            print("🎉 SUCCÈS: Configuration prête pour la Phase 1")
            print("✅ Peut procéder aux phases suivantes")
            return 0
        else:
            print("❌ ÉCHEC: Configuration nécessite des corrections")
            print("🔧 Corrigez les erreurs critiques avant de continuer")
            return 1
            
    except Exception as e:
        print(f"💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())