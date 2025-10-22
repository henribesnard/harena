"""
Metadata Service - Fournit les métadonnées dynamiques aux agents
Injecte les catégories et operation_types disponibles dans les prompts
"""
import sys
import os
import logging
from typing import Dict, List, Optional

# Ajouter le chemin parent pour importer enrichment_service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from enrichment_service.category_service import CategoryService
    category_service_available = True
except ImportError:
    category_service_available = False
    logging.warning("CategoryService not available, using fallback metadata")

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Service pour fournir les métadonnées dynamiques aux agents

    Responsabilités:
    - Charger les catégories depuis la base (via CategoryService)
    - Fournir le mapping catégories -> synonymes
    - Générer les sections de prompt enrichies
    - Cache automatique (1h via CategoryService)
    """

    # Mapping des synonymes pour améliorer le matching
    # Ce mapping peut être enrichi au fil du temps
    CATEGORY_SYNONYMS = {
        "Restaurant": ["restaurant", "restaurants", "resto", "restos", "restauration"],
        "Santé/pharmacie": [
            "santé", "sante", "pharmacie", "frais de santé", "frais de sante",
            "frais médicaux", "medical", "médical", "soins"
        ],
        "Virement sortants": [
            "virement", "virements", "virement sortant", "virements sortants",
            "transfert", "transferts"
        ],
        "Virement entrants": [
            "virement entrant", "virements entrants", "virement reçu",
            "virements reçus", "transfert reçu"
        ],
        "Retrait especes": [
            "retrait", "retraits", "retrait espèces", "retrait especes",
            "retraits d'argent", "cash", "espèces", "especes", "ATM"
        ],
        "Transport": [
            "transport", "transports", "déplacement", "déplacements",
            "mobilité", "mobilite"
        ],
        "Alimentation": [
            "alimentation", "courses", "alimentaire", "nourriture",
            "supermarché", "supermarche", "épicerie", "epicerie"
        ],
        "Loisirs": [
            "loisirs", "loisir", "divertissement", "divertissements",
            "entertainment", "hobbies"
        ],
        "Abonnements": [
            "abonnement", "abonnements", "subscription", "subscriptions",
            "mensualité", "mensualites"
        ],
        "Chèques émis": [
            "chèque", "cheque", "chèques", "cheques", "check", "checks"
        ],
        "Téléphones/internet": [
            "téléphone", "telephone", "internet", "mobile", "wifi",
            "téléphonie", "telephonie", "box internet"
        ],
        "Vêtements": [
            "vêtement", "vetement", "vêtements", "vetements",
            "habillement", "mode", "fringues"
        ],
        "Impôts": [
            "impôt", "impot", "impôts", "impots", "taxe", "taxes",
            "fiscal", "fiscalité", "fiscalite"
        ],
        "Assurances": [
            "assurance", "assurances", "insurance", "cotisation", "cotisations"
        ],
        "Carburant": [
            "carburant", "essence", "diesel", "gasoil", "gas-oil",
            "fuel", "station-service", "station service"
        ],
        "Frais bancaires": [
            "frais bancaire", "frais bancaires", "commission", "commissions",
            "frais de tenue de compte", "agios"
        ],
    }

    # Operation types disponibles
    # Ces valeurs correspondent aux valeurs exactes dans la base
    OPERATION_TYPES = {
        "Carte": ["carte", "carte bancaire", "CB", "card", "paiement carte"],
        "Prélèvement": ["prélèvement", "prelevement", "direct debit", "prélevement"],
        "Virement": ["virement", "transfer", "transfert"],
        "Chèque": ["chèque", "cheque", "check"],
        "Retrait": ["retrait", "withdrawal", "ATM", "distributeur"],
    }

    def __init__(self):
        if category_service_available:
            self.category_service = CategoryService()
            logger.info("MetadataService initialized with CategoryService")
        else:
            self.category_service = None
            logger.warning("MetadataService initialized without CategoryService (using fallback)")

    def get_all_categories(self) -> Dict[int, any]:
        """Récupère toutes les catégories disponibles"""
        if self.category_service:
            try:
                return self.category_service.get_all_categories()
            except Exception as e:
                logger.error(f"Error fetching categories: {e}")
                return {}
        return {}

    def get_categories_prompt_section(self) -> str:
        """
        Génère la section du prompt avec les catégories disponibles

        Returns:
            str: Section de prompt formatée avec toutes les catégories et leurs synonymes
        """
        categories = self.get_all_categories()

        if not categories:
            logger.warning("No categories available, returning empty prompt section")
            return ""

        prompt = "\n" + "=" * 80 + "\n"
        prompt += "## CATÉGORIES DISPONIBLES EN BASE\n"
        prompt += "=" * 80 + "\n"
        prompt += "Utilise UNIQUEMENT ces catégories exactes lors de la construction des filtres.\n"
        prompt += "Ne jamais inventer de catégories, utiliser EXACTEMENT les noms ci-dessous.\n\n"

        # Grouper par groupe
        by_group = {}
        for cat_id, cat in categories.items():
            group = cat.group_name or "Autres"
            if group not in by_group:
                by_group[group] = []
            by_group[group].append(cat)

        # Afficher par groupe
        for group_name in sorted(by_group.keys()):
            cats = by_group[group_name]
            prompt += f"\n### 📁 {group_name}\n"

            for cat in sorted(cats, key=lambda x: x.category_name):
                # Nom exact de la catégorie
                prompt += f"- **`{cat.category_name}`**"

                # Ajouter les synonymes si disponibles
                synonyms = self.CATEGORY_SYNONYMS.get(cat.category_name, [])
                if synonyms:
                    prompt += f"\n  → Synonymes: {', '.join(synonyms)}"

                prompt += "\n"

        # Instructions de mapping
        prompt += "\n" + "=" * 80 + "\n"
        prompt += "## RÈGLES DE MAPPING DES REQUÊTES UTILISATEUR\n"
        prompt += "=" * 80 + "\n\n"

        prompt += """**IMPORTANT - Comment mapper les requêtes utilisateur vers les catégories:**

1. **Normalisation des pluriels et variantes:**
   - "restaurants" → utilise **`Restaurant`**
   - "virements" → utilise **`Virement sortants`** (si contexte = dépenses/débit)
   - "retraits", "retrait d'argent" → utilise **`Retrait especes`**
   - "santé", "frais de santé" → utilise **`Santé/pharmacie`**
   - "transports" → utilise **`Transport`**

2. **Recherche intelligente:**
   - D'abord chercher une correspondance EXACTE dans les noms de catégories
   - Ensuite utiliser les synonymes ci-dessus
   - Gérer les accents : "sante" = "santé"
   - Gérer les pluriels : "restaurants" = "restaurant"

3. **Si aucune catégorie ne correspond:**
   - NE PAS filtrer sur category_name
   - Utiliser la recherche textuelle sur merchant_name et primary_description
   - Exemple: si l'utilisateur cherche "pizza" et qu'il n'y a pas de catégorie "pizza",
     faire une recherche textuelle au lieu d'utiliser une catégorie approximative

4. **Contexte des virements:**
   - "mes virements" sans contexte → **`Virement sortants`** (dépenses)
   - "virements reçus" → **`Virement entrants`** (revenus)
   - Toujours ajouter le filtre transaction_type approprié (debit/credit)

5. **Format des filtres category_name:**
   ```json
   // Pour correspondance exacte (préféré si catégorie connue)
   "category_name": {"term": "Restaurant"}

   // Pour recherche floue (si variante incertaine)
   "category_name": {"match": "Restaurant"}

   // Pour plusieurs catégories
   "category_name": {"terms": ["Restaurant", "Alimentation"]}
   ```

**RÈGLE D'OR:** Toujours utiliser le nom EXACT de la catégorie tel qu'il apparaît dans la liste ci-dessus.
Ne JAMAIS utiliser les synonymes dans les filtres, uniquement les noms exacts.
"""

        return prompt

    def get_operation_types_prompt_section(self) -> str:
        """
        Génère la section du prompt avec les operation_types disponibles

        Returns:
            str: Section de prompt formatée avec les operation_types
        """
        prompt = "\n" + "=" * 80 + "\n"
        prompt += "## OPERATION_TYPES DISPONIBLES\n"
        prompt += "=" * 80 + "\n"
        prompt += "Types d'opérations bancaires disponibles dans la base:\n\n"

        for op_type, synonyms in self.OPERATION_TYPES.items():
            prompt += f"- **`{op_type}`**\n"
            if synonyms:
                prompt += f"  → Synonymes: {', '.join(synonyms)}\n"

        prompt += """
**Usage des operation_types:**

⚠️ IMPORTANT: Les catégories sont généralement plus précises que les operation_types.
Privilégier l'utilisation des catégories sauf si l'utilisateur demande explicitement un type d'opération.

Exemples:
- "mes paiements par carte" → operation_type: "Carte"
- "tous mes prélèvements" → operation_type: "Prélèvement"
- "mes virements" → préférer category_name: "Virement sortants" + transaction_type: "debit"

Note: Les retraits et virements sont mieux gérés par les catégories que par operation_type.
"""
        return prompt

    def get_transaction_types_prompt_section(self) -> str:
        """
        Génère la section du prompt avec les transaction_types

        Returns:
            str: Section de prompt formatée avec les transaction_types
        """
        prompt = "\n" + "=" * 80 + "\n"
        prompt += "## TRANSACTION_TYPES (déjà bien géré)\n"
        prompt += "=" * 80 + "\n\n"

        prompt += """**Valeurs possibles:**
- **`debit`**: Dépenses, sorties d'argent (amount négatif dans la base)
- **`credit`**: Revenus, entrées d'argent (amount positif dans la base)

**Mapping des requêtes utilisateur:**
- "dépenses", "achats", "sorties" → transaction_type: "debit"
- "revenus", "salaire", "entrées", "virements reçus" → transaction_type: "credit"
- "mes virements" (sans précision) → par défaut transaction_type: "debit" (virements sortants)

**Important:**
- Toujours utiliser transaction_type pour différencier dépenses/revenus
- Ne pas confondre avec operation_type qui indique le moyen de paiement
"""
        return prompt

    def get_full_metadata_prompt(self) -> str:
        """
        Retourne le prompt complet avec toutes les métadonnées

        Returns:
            str: Prompt complet formaté incluant toutes les métadonnées
        """
        sections = [
            self.get_categories_prompt_section(),
            self.get_operation_types_prompt_section(),
            self.get_transaction_types_prompt_section()
        ]

        return "\n".join(section for section in sections if section)

    def search_category_by_name(self, query: str) -> Optional[str]:
        """
        Recherche une catégorie par nom ou synonyme

        Args:
            query: Nom ou synonyme de catégorie recherché

        Returns:
            str: Nom exact de la catégorie ou None si non trouvée
        """
        if not query:
            return None

        query_lower = query.lower().strip()

        # Récupérer toutes les catégories
        categories = self.get_all_categories()

        # Si CategoryService est disponible, utiliser la base de données
        if categories:
            # 1. Recherche exacte (insensible à la casse)
            for cat_id, cat in categories.items():
                if cat.category_name.lower() == query_lower:
                    return cat.category_name

            # 2. Recherche dans les synonymes avec validation en base
            for category_name, synonyms in self.CATEGORY_SYNONYMS.items():
                if query_lower in [s.lower() for s in synonyms]:
                    # Vérifier que cette catégorie existe bien en base
                    for cat_id, cat in categories.items():
                        if cat.category_name == category_name:
                            return category_name

            # 3. Recherche partielle dans les noms de catégories
            for cat_id, cat in categories.items():
                if query_lower in cat.category_name.lower():
                    return cat.category_name

        # MODE FALLBACK : Si CategoryService n'est pas disponible
        # Utiliser uniquement CATEGORY_SYNONYMS (hardcodé)
        else:
            logger.debug(f"Using fallback mode for category search: {query}")

            # 1. Recherche exacte dans les clés de CATEGORY_SYNONYMS
            for category_name in self.CATEGORY_SYNONYMS.keys():
                if category_name.lower() == query_lower:
                    return category_name

            # 2. Recherche dans les synonymes
            for category_name, synonyms in self.CATEGORY_SYNONYMS.items():
                if query_lower in [s.lower() for s in synonyms]:
                    return category_name

            # 3. Recherche partielle dans les clés
            for category_name in self.CATEGORY_SYNONYMS.keys():
                if query_lower in category_name.lower():
                    return category_name

        # Aucune correspondance trouvée
        logger.warning(f"No category found for query: {query}")
        return None


# Instance singleton
metadata_service = MetadataService()


# Pour debug/test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    service = MetadataService()

    print("\n" + "=" * 80)
    print("METADATA SERVICE - TEST")
    print("=" * 80)

    # Test récupération catégories
    categories = service.get_all_categories()
    print(f"\n✓ {len(categories)} categories loaded from database")

    # Test génération prompt
    prompt = service.get_full_metadata_prompt()
    print(f"\n✓ Generated prompt ({len(prompt)} chars)")
    print("\nPrompt preview (first 1000 chars):")
    print(prompt[:1000])

    # Test recherche
    print("\n" + "=" * 80)
    print("CATEGORY SEARCH TESTS")
    print("=" * 80)

    test_queries = [
        "restaurant", "restaurants", "santé", "sante", "frais de santé",
        "virement", "virements", "retrait", "retraits d'argent", "transport"
    ]

    for query in test_queries:
        result = service.search_category_by_name(query)
        status = "✓" if result else "✗"
        print(f"{status} '{query}' -> {result}")
