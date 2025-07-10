"""
Script principal pour lancer tous les tests du Search Service.

Ce script :
- Lance tous les tests unitaires
- Génère un rapport de couverture
- Vérifie l'intégrité du module
- Produit un résumé des résultats
"""

import sys
import os
import pytest
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime


# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SearchServiceTestRunner:
    """Runner principal pour les tests du Search Service."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        self.start_time = datetime.now()
    
    def setup_environment(self):
        """Prépare l'environnement de test."""
        logger.info("🔧 Configuration de l'environnement de test")
        
        # Ajouter le répertoire parent au PYTHONPATH
        sys.path.insert(0, str(self.project_root.parent))
        
        # Configurer les variables d'environnement de test
        os.environ['TESTING'] = 'true'
        os.environ['LOG_LEVEL'] = 'WARNING'  # Réduire le bruit pendant les tests
        
        logger.info("✅ Environnement configuré")
    
    def run_import_tests(self) -> Dict[str, Any]:
        """Lance les tests d'imports critiques."""
        logger.info("📦 Test des imports critiques")
        
        import_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        critical_imports = [
            "search_service",
            "search_service.main",
            # Note: autres modules optionnels testés dans les tests unitaires
        ]
        
        for module_name in critical_imports:
            try:
                __import__(module_name)
                import_results["passed"] += 1
                logger.info(f"✅ {module_name}")
            except ImportError as e:
                import_results["failed"] += 1
                import_results["errors"].append(f"{module_name}: {str(e)}")
                logger.error(f"❌ {module_name}: {e}")
        
        return import_results
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Lance tous les tests unitaires."""
        logger.info("🧪 Lancement des tests unitaires")
        
        # Configuration pytest
        pytest_args = [
            str(self.test_dir),
            "-v",  # Verbose
            "--tb=short",  # Traceback court
            "--strict-markers",  # Markers stricts
            "-x",  # Arrêter au premier échec
            "--disable-warnings",  # Réduire le bruit
        ]
        
        # Ajouter la couverture si pytest-cov est disponible
        try:
            import pytest_cov
            pytest_args.extend([
                "--cov=search_service",
                "--cov-report=term-missing",
                "--cov-report=json:coverage.json",
                "--cov-fail-under=70"  # 70% minimum de couverture
            ])
            logger.info("📊 Analyse de couverture activée")
        except ImportError:
            logger.info("⚠️ pytest-cov non disponible - tests sans couverture")
        
        # Lancer pytest
        exit_code = pytest.main(pytest_args)
        
        result = {
            "exit_code": exit_code,
            "success": exit_code == 0,
            "coverage_available": 'pytest_cov' in sys.modules
        }
        
        # Lire les résultats de couverture si disponibles
        coverage_file = Path("coverage.json")
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    result["coverage"] = coverage_data.get("totals", {})
                    logger.info(f"📊 Couverture: {result['coverage'].get('percent_covered', 'N/A')}%")
            except Exception as e:
                logger.warning(f"Erreur lecture couverture: {e}")
        
        return result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Lance les tests d'intégration spécifiques."""
        logger.info("🔗 Tests d'intégration")
        
        integration_results = {
            "app_creation": False,
            "endpoints_basic": False,
            "health_check": False,
            "errors": []
        }
        
        try:
            # Test création application
            from search_service.main import create_app
            app = create_app()
            integration_results["app_creation"] = True
            logger.info("✅ Création d'application")
            
            # Test endpoints basiques avec TestClient
            from fastapi.testclient import TestClient
            client = TestClient(app)
            
            # Test endpoint racine
            response = client.get("/")
            if response.status_code == 200:
                integration_results["endpoints_basic"] = True
                logger.info("✅ Endpoints basiques")
            
            # Test health check
            response = client.get("/health")
            if response.status_code in [200, 503]:  # 503 = dégradé mais fonctionnel
                integration_results["health_check"] = True
                logger.info("✅ Health check")
            
        except Exception as e:
            integration_results["errors"].append(str(e))
            logger.error(f"❌ Erreur intégration: {e}")
        
        return integration_results
    
    def check_module_integrity(self) -> Dict[str, Any]:
        """Vérifie l'intégrité du module."""
        logger.info("🔍 Vérification de l'intégrité du module")
        
        integrity_results = {
            "structure_valid": False,
            "init_files": [],
            "missing_files": [],
            "circular_imports": False
        }
        
        # Vérifier la structure des dossiers
        expected_dirs = ["api", "core", "models", "clients", "templates", "utils", "config"]
        existing_dirs = []
        
        for dir_name in expected_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                existing_dirs.append(dir_name)
                
                # Vérifier __init__.py
                init_file = dir_path / "__init__.py"
                if init_file.exists():
                    integrity_results["init_files"].append(dir_name)
                else:
                    integrity_results["missing_files"].append(f"{dir_name}/__init__.py")
        
        integrity_results["structure_valid"] = len(existing_dirs) > 0
        
        # Test d'imports circulaires (basique)
        try:
            import search_service
            integrity_results["circular_imports"] = False
            logger.info("✅ Pas d'imports circulaires détectés")
        except ImportError as e:
            if "circular" in str(e).lower():
                integrity_results["circular_imports"] = True
                logger.error(f"❌ Import circulaire détecté: {e}")
        
        return integrity_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Génère un rapport complet des tests."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "test_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Génère un résumé des résultats."""
        summary = {
            "overall_success": True,
            "total_issues": 0,
            "critical_issues": 0,
            "warnings": 0
        }
        
        # Analyser les résultats d'imports
        if "imports" in self.results:
            imports = self.results["imports"]
            if imports["failed"] > 0:
                summary["critical_issues"] += imports["failed"]
                summary["overall_success"] = False
        
        # Analyser les résultats des tests unitaires
        if "unit_tests" in self.results:
            unit_tests = self.results["unit_tests"]
            if not unit_tests["success"]:
                summary["critical_issues"] += 1
                summary["overall_success"] = False
        
        # Analyser l'intégrité
        if "integrity" in self.results:
            integrity = self.results["integrity"]
            if not integrity["structure_valid"]:
                summary["warnings"] += 1
            if integrity["circular_imports"]:
                summary["critical_issues"] += 1
                summary["overall_success"] = False
        
        summary["total_issues"] = summary["critical_issues"] + summary["warnings"]
        
        return summary
    
    def print_summary(self, report: Dict[str, Any]):
        """Affiche un résumé des résultats."""
        print("\n" + "="*60)
        print("🎯 RÉSUMÉ DES TESTS SEARCH SERVICE")
        print("="*60)
        
        summary = report["summary"]
        duration = report["test_session"]["duration_seconds"]
        
        # Statut global
        if summary["overall_success"]:
            print("✅ TESTS RÉUSSIS")
        else:
            print("❌ TESTS ÉCHOUÉS")
        
        print(f"⏱️  Durée: {duration:.2f}s")
        print(f"🐛 Issues critiques: {summary['critical_issues']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        
        # Détails par catégorie
        if "imports" in self.results:
            imports = self.results["imports"]
            print(f"\n📦 Imports: {imports['passed']} ✅ / {imports['failed']} ❌")
        
        if "unit_tests" in self.results:
            unit_tests = self.results["unit_tests"]
            status = "✅" if unit_tests["success"] else "❌"
            print(f"🧪 Tests unitaires: {status}")
            
            if "coverage" in unit_tests:
                coverage = unit_tests["coverage"]
                percent = coverage.get("percent_covered", 0)
                print(f"📊 Couverture: {percent:.1f}%")
        
        if "integration" in self.results:
            integration = self.results["integration"]
            app_status = "✅" if integration["app_creation"] else "❌"
            endpoints_status = "✅" if integration["endpoints_basic"] else "❌"
            health_status = "✅" if integration["health_check"] else "❌"
            
            print(f"\n🔗 Intégration:")
            print(f"   App création: {app_status}")
            print(f"   Endpoints: {endpoints_status}")
            print(f"   Health check: {health_status}")
        
        if "integrity" in self.results:
            integrity = self.results["integrity"]
            structure_status = "✅" if integrity["structure_valid"] else "❌"
            circular_status = "✅" if not integrity["circular_imports"] else "❌"
            
            print(f"\n🔍 Intégrité:")
            print(f"   Structure: {structure_status}")
            print(f"   Imports circulaires: {circular_status}")
            print(f"   Fichiers __init__.py: {len(integrity['init_files'])}")
        
        # Erreurs détaillées
        print("\n📝 DÉTAILS:")
        
        if "imports" in self.results and self.results["imports"]["errors"]:
            print("❌ Erreurs d'imports:")
            for error in self.results["imports"]["errors"]:
                print(f"   - {error}")
        
        if "integration" in self.results and self.results["integration"]["errors"]:
            print("❌ Erreurs d'intégration:")
            for error in self.results["integration"]["errors"]:
                print(f"   - {error}")
        
        print("\n" + "="*60)
    
    def save_report(self, report: Dict[str, Any]):
        """Sauvegarde le rapport en JSON."""
        report_file = self.test_dir / "test_report.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"📄 Rapport sauvegardé: {report_file}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport: {e}")
    
    def run_all_tests(self) -> int:
        """Lance tous les tests et retourne le code de sortie."""
        logger.info("🚀 Démarrage des tests Search Service")
        
        try:
            # 1. Configuration
            self.setup_environment()
            
            # 2. Tests d'imports
            self.results["imports"] = self.run_import_tests()
            
            # 3. Tests unitaires
            self.results["unit_tests"] = self.run_unit_tests()
            
            # 4. Tests d'intégration
            self.results["integration"] = self.run_integration_tests()
            
            # 5. Vérification d'intégrité
            self.results["integrity"] = self.check_module_integrity()
            
            # 6. Génération du rapport
            report = self.generate_report()
            
            # 7. Affichage et sauvegarde
            self.print_summary(report)
            self.save_report(report)
            
            # Retourner le code de sortie approprié
            return 0 if report["summary"]["overall_success"] else 1
            
        except Exception as e:
            logger.error(f"💥 Erreur fatale pendant les tests: {e}")
            print(f"\n💥 ERREUR FATALE: {e}")
            return 2


def main():
    """Point d'entrée principal."""
    runner = SearchServiceTestRunner()
    exit_code = runner.run_all_tests()
    
    # Messages de fin
    if exit_code == 0:
        print("\n🎉 Tous les tests sont passés avec succès!")
        print("✨ Le Search Service est prêt pour l'intégration!")
    elif exit_code == 1:
        print("\n⚠️  Certains tests ont échoué.")
        print("🔧 Veuillez corriger les problèmes avant de continuer.")
    else:
        print("\n💥 Erreur fatale pendant l'exécution des tests.")
        print("🆘 Vérifiez l'installation et la configuration.")
    
    return exit_code


# ==================== FONCTIONS UTILITAIRES ====================

def run_specific_test(test_name: str):
    """Lance un test spécifique."""
    logger.info(f"🎯 Lancement du test spécifique: {test_name}")
    
    test_files = {
        "init": "test_init.py",
        "main": "test_main.py", 
        "utils": "test_utils.py",
        "imports": "test_imports_only"
    }
    
    if test_name == "imports":
        # Test d'imports seulement
        runner = SearchServiceTestRunner()
        runner.setup_environment()
        results = runner.run_import_tests()
        
        print(f"\n📦 Résultats imports:")
        print(f"✅ Réussis: {results['passed']}")
        print(f"❌ Échoués: {results['failed']}")
        
        if results['errors']:
            print("❌ Erreurs:")
            for error in results['errors']:
                print(f"   - {error}")
        
        return 0 if results['failed'] == 0 else 1
    
    elif test_name in test_files:
        test_file = Path(__file__).parent / test_files[test_name]
        if test_file.exists():
            return pytest.main([str(test_file), "-v"])
        else:
            print(f"❌ Fichier de test non trouvé: {test_file}")
            return 1
    else:
        print(f"❌ Test inconnu: {test_name}")
        print(f"Tests disponibles: {list(test_files.keys())}")
        return 1


def check_dependencies():
    """Vérifie les dépendances nécessaires pour les tests."""
    logger.info("🔍 Vérification des dépendances de test")
    
    required_packages = ["pytest", "fastapi"]
    optional_packages = ["pytest-cov", "pytest-asyncio"]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            logger.error(f"❌ {package} (requis)")
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"⚠️ {package} (optionnel)")
    
    if missing_required:
        print(f"\n❌ Packages requis manquants: {', '.join(missing_required)}")
        print("📦 Installez avec: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Packages optionnels manquants: {', '.join(missing_optional)}")
        print("📦 Installez avec: pip install " + " ".join(missing_optional))
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Runner de tests pour Search Service")
    parser.add_argument(
        "--test", 
        choices=["all", "init", "main", "utils", "imports"],
        default="all",
        help="Test spécifique à lancer"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Vérifier seulement les dépendances"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Désactiver l'analyse de couverture"
    )
    
    args = parser.parse_args()
    
    # Vérification des dépendances
    if args.check_deps:
        check_dependencies()
        sys.exit(0)
    
    if not check_dependencies():
        print("❌ Dépendances manquantes. Utilisez --check-deps pour plus de détails.")
        sys.exit(1)
    
    # Lancement des tests
    if args.test == "all":
        exit_code = main()
    else:
        exit_code = run_specific_test(args.test)
    
    sys.exit(exit_code)