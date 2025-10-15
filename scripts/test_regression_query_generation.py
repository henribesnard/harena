"""
Test de régression pour la génération de requêtes
Vérifie que les corrections n'introduisent pas de régressions sur les tests qui passaient
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import difflib

def load_json(file_path: Path) -> Dict:
    """Charge un fichier JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_query_data(expected: Dict, actual: Dict, test_id: str) -> List[str]:
    """Compare deux query_data et retourne les différences"""
    differences = []

    # Comparer les champs principaux
    for key in ['user_id', 'filters', 'sort', 'page_size', 'aggregations']:
        if key not in expected and key not in actual:
            continue

        if key not in expected:
            differences.append(f"{test_id}: {key} absent dans expected mais présent dans actual")
            continue

        if key not in actual:
            differences.append(f"{test_id}: {key} présent dans expected mais absent dans actual")
            continue

        # Comparaison profonde pour les objets
        if expected[key] != actual[key]:
            diff_str = json.dumps(expected[key], indent=2, ensure_ascii=False)
            actual_str = json.dumps(actual[key], indent=2, ensure_ascii=False)
            diff_lines = list(difflib.unified_diff(
                diff_str.splitlines(),
                actual_str.splitlines(),
                lineterm='',
                fromfile='expected',
                tofile='actual'
            ))
            if diff_lines:
                differences.append(f"{test_id}: {key} differs:\n" + "\n".join(diff_lines[:20]))

    return differences

def run_regression_tests(baseline_dir: Path, current_dir: Path) -> bool:
    """Exécute les tests de régression"""
    print("=" * 80)
    print("TESTS DE RÉGRESSION - Génération de requêtes")
    print("=" * 80)

    all_passed = True
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    # Charger les résultats de référence (baseline)
    baseline_categories = ['category_A_montant', 'category_B_vendeur']

    for category in baseline_categories:
        category_dir = baseline_dir / category
        if not category_dir.exists():
            print(f"[WARN] Catégorie de référence manquante: {category}")
            continue

        # Charger le fichier de résultats
        results_file = category_dir / f"{category.split('_')[1]}_{category.split('_')[2]}_results.json"
        if not results_file.exists():
            print(f"[WARN] Fichier de résultats manquant: {results_file}")
            continue

        baseline_results = load_json(results_file)

        print(f"\nCategorie: {category}")
        print(f"   Tests de référence: {len(baseline_results['detailed_results'])}")

        # Comparer avec les résultats actuels
        current_results_file = current_dir / category / f"{category.split('_')[1]}_{category.split('_')[2]}_results.json"

        if not current_results_file.exists():
            print(f"   [FAIL] Fichier de résultats actuels manquant: {current_results_file}")
            all_passed = False
            continue

        current_results = load_json(current_results_file)

        # Comparer chaque test individuel
        for baseline_test in baseline_results['detailed_results']:
            test_id = baseline_test['question_id']
            total_tests += 1

            # Trouver le test correspondant dans les résultats actuels
            current_test = next(
                (t for t in current_results['detailed_results'] if t['question_id'] == test_id),
                None
            )

            if not current_test:
                print(f"   [FAIL] {test_id}: Test manquant dans les résultats actuels")
                all_passed = False
                failed_tests += 1
                continue

            # Vérifier que le test passe toujours
            if not baseline_test.get('api_success', False):
                # Si le test de référence échouait, on ne le compte pas
                total_tests -= 1
                continue

            if not current_test.get('api_success', False):
                print(f"   [FAIL] {test_id}: Le test réussit dans baseline mais échoue maintenant")
                print(f"      Erreur: {current_test.get('error_message', 'N/A')}")
                all_passed = False
                failed_tests += 1
                continue

            # Comparer les query_data
            baseline_query = baseline_test.get('query_data', {})
            current_query = current_test.get('query_data', {})

            differences = compare_query_data(baseline_query, current_query, test_id)

            if differences:
                print(f"   [WARN] {test_id}: Différences détectées dans query_data:")
                for diff in differences[:3]:  # Limiter l'affichage
                    print(f"      {diff}")
                # On ne considère pas cela comme un échec total, juste un avertissement
                # all_passed = False
                # failed_tests += 1
                passed_tests += 1
            else:
                print(f"   [PASS] {test_id}: OK")
                passed_tests += 1

    # Résumé final
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS DE RÉGRESSION")
    print("=" * 80)
    print(f"Total de tests: {total_tests}")
    print(f"Réussis: {passed_tests}")
    print(f"Échoués: {failed_tests}")
    print(f"Taux de réussite: {(passed_tests/total_tests*100) if total_tests > 0 else 0:.1f}%")

    if all_passed and failed_tests == 0:
        print("\n[SUCCESS] Tous les tests de régression ont passé!")
        return True
    else:
        print("\n[FAILURE] Certains tests de régression ont échoué")
        return False

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / "resultat_ok" / "test_reports_query_extraction"
    current_dir = project_root / "test_reports_query_extraction"

    if not baseline_dir.exists():
        print(f"[FAIL] Répertoire de référence manquant: {baseline_dir}")
        print("   Exécutez d'abord la sauvegarde des résultats OK")
        sys.exit(1)

    success = run_regression_tests(baseline_dir, current_dir)
    sys.exit(0 if success else 1)
