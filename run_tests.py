#!/usr/bin/env python3
"""
🧪 Script Exécution Tests Search Service - Version Windows Compatible
====================================================================

Script pour lancer les tests du Search Service avec compatibilité Windows complète.
Gère l'environnement, les dépendances et utilise directement Python pour pytest.

Usage:
    python run_tests.py [options]
    
Options:
    --models          Tests modèles uniquement
    --integration     Tests intégration uniquement
    --contracts       Tests contrats uniquement
    --all             Tous les tests (par défaut)
    --coverage        Avec coverage
    --verbose         Mode verbose
    --fast            Tests rapides uniquement
"""

import sys
import os
import argparse
import importlib
from pathlib import Path

# Configuration chemins
PROJECT_ROOT = Path(__file__).parent
SEARCH_SERVICE_PATH = PROJECT_ROOT / "search_service"
TESTS_PATH = SEARCH_SERVICE_PATH / "tests"

def setup_environment():
    """Configuration environnement test."""
    print("🔧 Configuration environnement test...")
    
    # Ajouter au PYTHONPATH
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Variables environnement test
    test_env = {
        "TESTING": "true",
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "ELASTICSEARCH_HOST": "localhost",
        "REDIS_ENABLED": "false",
        "MOCK_ELASTICSEARCH": "true",
    }
    
    os.environ.update(test_env)
    print("✅ Environnement configuré")

def check_dependencies():
    """Vérification dépendances."""
    print("📦 Vérification dépendances...")
    
    required_packages = {
        "pytest": "pytest",
        "pydantic": "pydantic",
    }
    
    missing_packages = []
    available_packages = {}
    
    for display_name, import_name in required_packages.items():
        try:
            module = importlib.import_module(import_name)
            available_packages[display_name] = getattr(module, '__version__', 'unknown')
        except ImportError:
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"❌ Packages manquants: {', '.join(missing_packages)}")
        print("Installation:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Dépendances OK")
    for pkg, version in available_packages.items():
        print(f"   {pkg}: {version}")
    return True

def validate_structure():
    """Validation structure projet."""
    print("📁 Validation structure projet...")
    
    required_paths = [
        SEARCH_SERVICE_PATH,
        TESTS_PATH,
        SEARCH_SERVICE_PATH / "models",
        SEARCH_SERVICE_PATH / "__init__.py",
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        print(f"❌ Chemins manquants: {', '.join(missing_paths)}")
        return False
    
    print("✅ Structure projet OK")
    return True

def test_imports():
    """Test imports modèles de base."""
    print("🔍 Test imports modèles...")
    
    try:
        # Test import search_service
        from search_service.models import (
            SearchServiceQuery, SearchServiceResponse,
            SimpleLexicalSearchRequest, BaseResponse
        )
        print("✅ Imports modèles principaux OK")
        return True
    except Exception as e:
        print(f"❌ Erreur imports: {e}")
        return False

def run_pytest_directly(test_args):
    """Exécution pytest directement via Python."""
    print(f"🚀 Lancement tests: {' '.join(test_args)}")
    
    try:
        # Import et exécution pytest
        import pytest
        
        # Changer de répertoire vers la racine du projet
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        try:
            # Exécuter pytest directement
            exit_code = pytest.main(test_args)
            return exit_code == 0
        finally:
            # Restaurer répertoire original
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Erreur exécution pytest: {e}")
        return False

def build_pytest_args(args):
    """Construction arguments pytest selon options."""
    pytest_args = []
    
    # Chemin tests de base
    if args.models:
        test_file = TESTS_PATH / "test_models" / "test_models_integration.py"
        if test_file.exists():
            pytest_args.append(str(test_file))
        else:
            pytest_args.append(str(TESTS_PATH / "test_models"))
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
        pytest_args.append(str(TESTS_PATH))
    elif args.contracts:
        pytest_args.extend(["-m", "contracts"])
        pytest_args.append(str(TESTS_PATH))
    elif args.all:
        pytest_args.append(str(TESTS_PATH))
    else:
        # Par défaut: tests modèles
        test_file = TESTS_PATH / "test_models" / "test_models_integration.py"
        if test_file.exists():
            pytest_args.append(str(test_file))
        else:
            pytest_args.append(str(TESTS_PATH))
    
    # Options verbosité
    if args.verbose:
        pytest_args.extend(["-v", "-s"])
    else:
        pytest_args.append("-v")
    
    # Tests rapides uniquement
    if args.fast:
        pytest_args.extend(["-m", "not performance"])
    
    # Coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=search_service",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Options additionnelles
    pytest_args.extend([
        "--tb=short",           # Traceback court
        "-ra",                 # Résumé tous les tests
    ])
    
    return pytest_args

def run_basic_tests():
    """Exécution tests de base sans pytest."""
    print("🧪 Exécution tests de base...")
    
    try:
        # Test 1: Imports
        print("  📦 Test imports...")
        from search_service.models import (
            SearchServiceQuery, SearchServiceResponse, ContractValidator,
            QueryMetadata, SearchParameters, FilterGroup,
            IntentType, FilterOperator
        )
        print("    ✅ Imports OK")
        
        # Test 2: Création objets
        print("  🏗️ Test création objets...")
        query = SearchServiceQuery(
            query_metadata=QueryMetadata(
                user_id=34,
                intent_type=IntentType.TEXT_SEARCH,
                confidence=0.9,
                agent_name="test_agent"
            ),
            search_parameters=SearchParameters(
                query_type="simple_search",
                fields=["searchable_text"],
                limit=10
            ),
            filters=FilterGroup(
                required=[
                    {"field": "user_id", "operator": FilterOperator.EQ, "value": 34}
                ]
            )
        )
        print("    ✅ Création SearchServiceQuery OK")
        
        # Test 3: Validation contrat
        print("  ✅ Test validation contrat...")
        validation = ContractValidator.validate_search_query(query)
        assert validation["valid"] is True, f"Validation échouée: {validation['errors']}"
        print("    ✅ Validation contrat OK")
        
        # Test 4: Sérialisation
        print("  📄 Test sérialisation...")
        json_str = query.json()
        assert len(json_str) > 100, "JSON trop court"
        print("    ✅ Sérialisation OK")
        
        # Test 5: Désérialisation
        print("  📥 Test désérialisation...")
        import json
        query_dict = json.loads(json_str)
        restored_query = SearchServiceQuery.parse_obj(query_dict)
        assert restored_query.query_metadata.user_id == 34
        print("    ✅ Désérialisation OK")
        
        print("🎉 Tous les tests de base passent!")
        return True
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Exécution tests Search Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python run_tests.py                    # Tests modèles de base
    python run_tests.py --models --verbose # Tests modèles en mode verbose
    python run_tests.py --contracts        # Tests contrats uniquement
    python run_tests.py --coverage         # Avec rapport coverage
    python run_tests.py --fast             # Tests rapides uniquement
    python run_tests.py --basic            # Tests de base sans pytest
        """
    )
    
    # Arguments
    parser.add_argument("--models", action="store_true", help="Tests modèles uniquement")
    parser.add_argument("--integration", action="store_true", help="Tests intégration")
    parser.add_argument("--contracts", action="store_true", help="Tests contrats")
    parser.add_argument("--all", action="store_true", help="Tous les tests")
    parser.add_argument("--coverage", action="store_true", help="Avec coverage")
    parser.add_argument("--verbose", action="store_true", help="Mode verbose")
    parser.add_argument("--fast", action="store_true", help="Tests rapides uniquement")
    parser.add_argument("--basic", action="store_true", help="Tests de base sans pytest")
    
    args = parser.parse_args()
    
    print("🧪 Search Service - Exécution Tests")
    print("=" * 50)
    
    # Étapes de validation
    setup_environment()
    
    if not validate_structure():
        sys.exit(1)
    
    if not test_imports():
        print("❌ Problème avec les imports, impossible de continuer")
        sys.exit(1)
    
    # Si mode basic, exécuter tests sans pytest
    if args.basic:
        print("\n📋 Mode tests de base (sans pytest)")
        print("=" * 50)
        success = run_basic_tests()
        print("\n" + "=" * 50)
        if success:
            print("✅ Tests de base réussis!")
        else:
            print("❌ Tests de base échoués!")
            sys.exit(1)
        return
    
    # Vérifier pytest disponible
    if not check_dependencies():
        print("\n⚠️ Pytest non disponible, basculement vers tests de base...")
        success = run_basic_tests()
        print("\n" + "=" * 50)
        if success:
            print("✅ Tests de base réussis!")
        else:
            print("❌ Tests de base échoués!")
            sys.exit(1)
        return
    
    # Construction arguments pytest
    pytest_args = build_pytest_args(args)
    
    print("\n📋 Configuration tests:")
    print(f"   Commande: pytest {' '.join(pytest_args)}")
    print(f"   Répertoire: {PROJECT_ROOT}")
    
    # Exécution tests
    print("\n" + "=" * 50)
    success = run_pytest_directly(pytest_args)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Tests exécutés avec succès!")
        
        if args.coverage:
            coverage_path = PROJECT_ROOT / "htmlcov" / "index.html"
            if coverage_path.exists():
                print(f"📊 Rapport coverage: {coverage_path}")
    else:
        print("❌ Échec des tests!")
        print("\n🔧 Essayez avec --basic pour tests simples:")
        print("   python run_tests.py --basic")
        sys.exit(1)

if __name__ == "__main__":
    main()