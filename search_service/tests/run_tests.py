#!/usr/bin/env python3
"""
🧪 Script Exécution Tests Search Service
========================================

Script pour lancer les tests du Search Service avec différentes options.
Gère l'environnement, les dépendances et les rapports de tests.

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
import subprocess
import argparse
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
    
    required_packages = [
        "pytest",
        "pydantic",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Packages manquants: {', '.join(missing_packages)}")
        print("Installation:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Dépendances OK")
    return True

def validate_structure():
    """Validation structure projet."""
    print("📁 Validation structure projet...")
    
    required_paths = [
        SEARCH_SERVICE_PATH,
        TESTS_PATH,
        SEARCH_SERVICE_PATH / "models",
        TESTS_PATH / "conftest.py",
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

def run_pytest(test_args):
    """Exécution pytest avec arguments."""
    print(f"🚀 Lancement tests: {' '.join(test_args)}")
    
    # Construction commande pytest
    cmd = ["python", "-m", "pytest"] + test_args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Erreur exécution pytest: {e}")
        return False

def build_pytest_args(args):
    """Construction arguments pytest selon options."""
    pytest_args = []
    
    # Chemin tests de base
    if args.models:
        pytest_args.append(str(TESTS_PATH / "test_models"))
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.contracts:
        pytest_args.extend(["-m", "contracts"])
    else:
        # Par défaut: tests modèles
        pytest_args.append(str(TESTS_PATH / "test_models" / "test_models_integration.py"))
    
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
            "--cov-report=term"
        ])
    
    # Options additionnelles
    pytest_args.extend([
        "--tb=short",           # Traceback court
        "--strict-markers",     # Markers stricts
        "-ra",                 # Résumé tous les tests
    ])
    
    return pytest_args

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
    
    args = parser.parse_args()
    
    print("🧪 Search Service - Exécution Tests")
    print("=" * 50)
    
    # Étapes de validation
    setup_environment()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not validate_structure():
        sys.exit(1)
    
    # Construction arguments pytest
    pytest_args = build_pytest_args(args)
    
    print("\n📋 Configuration tests:")
    print(f"   Commande: pytest {' '.join(pytest_args)}")
    print(f"   Répertoire: {PROJECT_ROOT}")
    
    # Exécution tests
    print("\n" + "=" * 50)
    success = run_pytest(pytest_args)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Tests exécutés avec succès!")
        
        if args.coverage:
            coverage_path = PROJECT_ROOT / "htmlcov" / "index.html"
            if coverage_path.exists():
                print(f"📊 Rapport coverage: {coverage_path}")
    else:
        print("❌ Échec des tests!")
        sys.exit(1)

if __name__ == "__main__":
    main()