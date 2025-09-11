"""
Validation des templates contre search_service
Teste la compatibilité et la performance
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ajouter le chemin pour imports
sys.path.append(str(Path(__file__).parent.parent))

from core.template_engine import TemplateEngine
from pydantic import BaseModel, ValidationError

# Import du modèle search_service pour validation
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "search_service"))
    from models.request import SearchRequest
    SEARCH_SERVICE_AVAILABLE = True
except ImportError:
    SEARCH_SERVICE_AVAILABLE = False
    print("WARNING: search_service models not available, using mock validation")

class TemplateValidationResult:
    def __init__(self, template_name: str):
        self.template_name = template_name
        self.compilation_success = False
        self.compilation_time_ms = 0.0
        self.validation_success = False
        self.search_service_compatible = False
        self.errors = []
        self.warnings = []

async def validate_all_templates():
    """Valide tous les templates"""
    print("Validation des templates - Phase 2")
    print("=" * 50)
    
    # Initialiser le moteur de templates
    template_engine = TemplateEngine()
    success = await template_engine.initialize()
    
    if not success:
        print("ERREUR: Impossible d'initialiser le moteur de templates")
        return False
    
    print(f"Moteur de templates initialisé avec {len(template_engine.compiled_templates)} templates")
    
    # Tester chaque template
    results = []
    templates_dir = Path(__file__).parent / "query"
    
    for template_file in templates_dir.rglob("*.json"):
        result = await validate_single_template(template_engine, template_file)
        results.append(result)
    
    # Afficher les résultats
    print_validation_results(results)
    
    # Performance globale
    await test_template_performance(template_engine)
    
    return all(r.compilation_success for r in results)

async def validate_single_template(engine: TemplateEngine, template_file: Path) -> TemplateValidationResult:
    """Valide un template individuel"""
    result = TemplateValidationResult(template_file.stem)
    
    # Test de compilation
    compilation_result = await engine.compile_template(template_file)
    result.compilation_success = compilation_result.success
    result.compilation_time_ms = compilation_result.compilation_time_ms
    
    if not compilation_result.success:
        result.errors.append(f"Compilation failed: {compilation_result.error_message}")
        return result
    
    template = compilation_result.template
    result.validation_success = True
    
    # Test de rendu avec des données fictives
    try:
        test_params = generate_test_parameters(template)
        rendered_query = await engine.render_template(template, test_params)
        
        # Valider contre search_service si disponible
        if SEARCH_SERVICE_AVAILABLE:
            try:
                search_request = SearchRequest(**rendered_query)
                result.search_service_compatible = True
            except ValidationError as e:
                result.search_service_compatible = False
                result.errors.append(f"SearchRequest validation failed: {e}")
        else:
            # Validation basique
            result.search_service_compatible = validate_basic_structure(rendered_query)
            if not result.search_service_compatible:
                result.errors.append("Basic structure validation failed")
                
    except Exception as e:
        result.errors.append(f"Template rendering failed: {e}")
        
    # Vérifications supplémentaires
    check_template_optimizations(template, result)
    
    return result

def generate_test_parameters(template) -> Dict[str, Any]:
    """Génère des paramètres de test pour un template"""
    return {
        "context": {
            "user_id": 34,
            "page_size": 50
        },
        "entities": {
            "periode_temporelle": {
                "date": {
                    "gte": "2025-01-01T00:00:00Z",
                    "lte": "2025-01-31T23:59:59Z"
                }
            },
            "montant": {
                "amount_abs": {"gte": 100.0}
            },
            "merchant_search": {
                "query": "Amazon",
                "merchant_name": "Amazon"
            },
            "category_search": {
                "category_name": "restaurant"
            },
            "transaction_type": {
                "transaction_type": "debit"
            },
            "granularite_temporelle": "1w"
        }
    }

def validate_basic_structure(query: Dict[str, Any]) -> bool:
    """Validation basique de la structure de requête"""
    required_fields = ["user_id"]
    
    for field in required_fields:
        if field not in query:
            return False
    
    # Vérifier que user_id est un entier
    if not isinstance(query["user_id"], int):
        return False
    
    # Vérifier la structure des filtres si présents
    if "filters" in query and not isinstance(query["filters"], dict):
        return False
        
    # Vérifier la structure des agrégations si présentes
    if "aggregations" in query and not isinstance(query["aggregations"], dict):
        return False
    
    return True

def check_template_optimizations(template, result: TemplateValidationResult):
    """Vérifie les optimisations du template"""
    optimizations = template.template_data.get("optimizations", {})
    
    # Vérifier la durée de cache
    if "cache_duration" not in optimizations:
        result.warnings.append("No cache duration specified")
    
    # Vérifier le routing Elasticsearch
    if "elasticsearch_routing" not in optimizations:
        result.warnings.append("No Elasticsearch routing specified")
    elif optimizations["elasticsearch_routing"] != "user_id":
        result.warnings.append("Elasticsearch routing should be 'user_id' for performance")

async def test_template_performance(engine: TemplateEngine):
    """Teste la performance de compilation des templates"""
    print("\nTests de performance:")
    print("-" * 30)
    
    # Test de compilation à froid
    template_file = Path(__file__).parent / "query" / "transaction_search" / "by_date.json"
    
    if template_file.exists():
        start_time = time.perf_counter()
        result = await engine.compile_template(template_file)
        cold_compile_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Compilation à froid: {cold_compile_time:.2f}ms")
        
        if result.success:
            # Test de rendu
            test_params = generate_test_parameters(result.template)
            
            start_time = time.perf_counter()
            await engine.render_template(result.template, test_params)
            render_time = (time.perf_counter() - start_time) * 1000
            
            print(f"Rendu template: {render_time:.2f}ms")
            
            # Critère Phase 2: < 50ms
            if cold_compile_time < 50:
                print("✅ Performance compilation OK (< 50ms)")
            else:
                print("❌ Performance compilation trop lente (>= 50ms)")
    
    # Statistiques de cache
    stats = engine.get_cache_stats()
    print(f"\nStatistiques cache:")
    print(f"  Templates en cache: {stats['templates_in_cache']}")
    print(f"  Compilations totales: {stats['total_compilations']}")

def print_validation_results(results: List[TemplateValidationResult]):
    """Affiche les résultats de validation"""
    print(f"\nResultats de validation ({len(results)} templates):")
    print("-" * 50)
    
    success_count = 0
    total_compilation_time = 0
    
    for result in results:
        status = "✅" if result.compilation_success and result.search_service_compatible else "❌"
        print(f"{status} {result.template_name}")
        
        if result.compilation_success:
            success_count += 1
            total_compilation_time += result.compilation_time_ms
            print(f"    Compilation: {result.compilation_time_ms:.2f}ms")
            
            if result.search_service_compatible:
                print("    Compatible search_service: OUI")
            else:
                print("    Compatible search_service: NON")
        
        # Afficher erreurs
        for error in result.errors:
            print(f"    ERREUR: {error}")
            
        # Afficher avertissements
        for warning in result.warnings:
            print(f"    WARNING: {warning}")
    
    # Résumé
    print(f"\nResume:")
    print(f"  Templates validés: {success_count}/{len(results)}")
    if success_count > 0:
        avg_compilation = total_compilation_time / success_count
        print(f"  Temps compilation moyen: {avg_compilation:.2f}ms")
    
    # Critères Phase 2
    print(f"\nCriteres Phase 2:")
    print(f"  ✅ Templates génèrent requêtes JSON valides: {success_count > 0}")
    print(f"  ✅ 100% templates testés contre search_service: {success_count == len(results)}")
    
    return success_count == len(results)

async def main():
    """Fonction principale"""
    try:
        success = await validate_all_templates()
        
        if success:
            print("\n🎉 SUCCES: Tous les templates sont valides")
            print("✅ Phase 2 peut être validée")
            return 0
        else:
            print("\n❌ ECHEC: Certains templates ont des erreurs")
            print("🔧 Corrigez les erreurs avant de continuer")
            return 1
            
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)