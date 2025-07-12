"""
🔍 Diagnostic et correction rapide des imports manquants
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("🔍 Diagnostic imports manquants...")

# 1. Vérifier ce qui est dans models.requests
print("\n1️⃣ Contenu actuel de models.requests:")
try:
    from search_service.models import requests as req_module
    print("  Attributs disponibles:")
    for attr in dir(req_module):
        if not attr.startswith('_'):
            print(f"    - {attr}")
except Exception as e:
    print(f"  ❌ Erreur import requests: {e}")

# 2. Vérifier ce qui est demandé dans api.routes
print("\n2️⃣ Imports demandés dans api/routes.py:")
routes_file = PROJECT_ROOT / "search_service" / "api" / "routes.py"
if routes_file.exists():
    with open(routes_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if 'from models.requests import' in line:
            print(f"  Ligne {i}: {line.strip()}")
            break
else:
    print("  ❌ routes.py non trouvé")

# 3. Vérification rapide - qu'est-ce qui manque exactement
print("\n3️⃣ Classes manquantes à ajouter:")
missing_classes = [
    "ValidationRequest", 
    "TemplateRequest"
]

for class_name in missing_classes:
    print(f"  - {class_name}: À créer dans models/requests.py")

print("\n🔧 Solution rapide:")
print("Ajouter ces classes dans search_service/models/requests.py:")
print("""
class ValidationRequest(BaseModel):
    \"\"\"Requête de validation\"\"\"
    query: Dict[str, Any] = Field(..., description="Requête à valider")
    validate_security: bool = Field(default=True, description="Valider sécurité")

class TemplateRequest(BaseModel):
    \"\"\"Requête de template\"\"\"
    template_name: str = Field(..., description="Nom du template")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres")
""")