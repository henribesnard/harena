"""
Test simple des templates
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.template_engine import TemplateEngine

async def test_templates():
    print("Test des templates Phase 2")
    print("=" * 40)
    
    # Initialiser le moteur
    engine = TemplateEngine()
    success = await engine.initialize()
    
    if not success:
        print("ERREUR: Initialisation échouée")
        return False
    
    print(f"Templates charges: {len(engine.compiled_templates)}")
    
    # Tester la compilation d'un template
    templates_dir = Path(__file__).parent / "query"
    test_file = templates_dir / "transaction_search" / "by_date.json"
    
    if test_file.exists():
        result = await engine.compile_template(test_file)
        print(f"Compilation by_date.json: {'SUCCES' if result.success else 'ECHEC'}")
        print(f"Temps compilation: {result.compilation_time_ms:.2f}ms")
        
        if result.success:
            # Test de rendu
            test_params = {
                "context": {"user_id": 34},
                "entities": {
                    "periode_temporelle": {
                        "date": {"gte": "2025-01-01T00:00:00Z", "lte": "2025-01-31T23:59:59Z"}
                    }
                }
            }
            
            try:
                rendered = await engine.render_template(result.template, test_params)
                print("Rendu template: SUCCES")
                print(f"User ID généré: {rendered.get('user_id')}")
                print(f"Filtre date présent: {'filters' in rendered and 'date' in rendered['filters']}")
                
                # Vérifier performance < 50ms
                if result.compilation_time_ms < 50:
                    print("Performance: OK (< 50ms)")
                else:
                    print("Performance: LENTE (>= 50ms)")
                    
                return True
                
            except Exception as e:
                print(f"Erreur rendu: {e}")
                return False
        else:
            print(f"Erreurs: {result.error_message}")
            return False
    else:
        print("Template by_date.json non trouvé")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_templates())
    print(f"Resultat: {'SUCCES' if result else 'ECHEC'}")