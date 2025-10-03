"""
Category Service pour le Conversation Service

Fournit la logique d'arbitrage des catégories pour l'extraction d'entités.
Récupère dynamiquement les catégories depuis PostgreSQL.
"""

from typing import List, Set, Optional, Dict
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CategoryService:
    """Service pour gérer les catégories de transactions"""

    def __init__(self):
        """Initialise le service avec cache des catégories"""
        self._categories_cache: Optional[List[str]] = None
        self._expense_categories_cache: Optional[List[str]] = None
        self._income_categories_cache: Optional[List[str]] = None
        self._categories_by_group: Optional[Dict[str, List[str]]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)  # Rafraîchir toutes les heures

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if self._cache_timestamp is None:
            return False
        return (datetime.now() - self._cache_timestamp) < self._cache_duration

    def _fetch_categories_from_db(self) -> tuple[List[str], Dict[str, List[str]]]:
        """Récupère les catégories depuis PostgreSQL avec leurs groupes"""
        try:
            from sqlalchemy import text
            from db_service.session import get_db

            db = next(get_db())
            try:
                # Récupérer toutes les catégories avec leurs groupes via JOIN
                result = db.execute(text("""
                    SELECT c.category_name, cg.group_name
                    FROM categories c
                    LEFT JOIN category_groups cg ON c.group_id = cg.group_id
                    ORDER BY cg.group_name, c.category_name
                """))

                all_categories = []
                categories_by_group = {}

                for row in result:
                    category_name = row.category_name
                    group_name = row.group_name if row.group_name else "Sans groupe"

                    all_categories.append(category_name)

                    if group_name not in categories_by_group:
                        categories_by_group[group_name] = []
                    categories_by_group[group_name].append(category_name)

                logger.info(f"Catégories chargées depuis DB: {len(all_categories)} catégories, {len(categories_by_group)} groupes")

                return all_categories, categories_by_group

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Erreur récupération catégories depuis DB: {e}")
            # Fallback sur les catégories par défaut
            return self._get_fallback_categories()

    def _get_fallback_categories(self) -> tuple[List[str], Dict[str, List[str]]]:
        """Catégories de fallback si la DB n'est pas accessible - Version française"""
        logger.warning("Utilisation des catégories de fallback (françaises)")

        # Catégories françaises de la base harena_sync
        fallback_categories = [
            "Achats en ligne",
            "Aide sociale/CAF",
            "Alimentation/Supermarché",
            "Assurance/Mutuelle",
            "Culture/Loisirs",
            "Dons/Charité",
            "Eau/Gaz/Électricité",
            "Enseignement/Formation",
            "Épargne/Investissement",
            "Frais bancaires",
            "Impôts/Taxes",
            "Loyer/Charges",
            "Logement/Équipement",
            "Loisirs/Divertissement",
            "Prêts/Crédits",
            "Restaurants/Sorties",
            "Salaire/Revenus",
            "Santé/Pharmacie",
            "Sport/Bien-être",
            "Télécom/Internet",
            "Transport/Carburant",
            "Vêtements/Mode",
            "Voyages/Vacances"
        ]

        fallback_groups = {
            "Revenus": ["Salaire/Revenus", "Aide sociale/CAF"],
            "Alimentation": ["Alimentation/Supermarché", "Restaurants/Sorties"],
            "Transport": ["Transport/Carburant"],
            "Santé": ["Santé/Pharmacie", "Assurance/Mutuelle"],
            "Loisirs": ["Loisirs/Divertissement", "Culture/Loisirs", "Sport/Bien-être", "Voyages/Vacances"],
            "Shopping": ["Vêtements/Mode", "Achats en ligne", "Logement/Équipement"],
            "Logement": ["Loyer/Charges", "Eau/Gaz/Électricité"],
            "Services": ["Télécom/Internet", "Frais bancaires", "Impôts/Taxes"],
            "Autres": ["Enseignement/Formation", "Épargne/Investissement", "Prêts/Crédits", "Dons/Charité"]
        }

        return fallback_categories, fallback_groups

    def _load_categories(self):
        """Charge les catégories (depuis cache ou DB)"""
        if self._is_cache_valid():
            return

        all_categories, categories_by_group = self._fetch_categories_from_db()

        self._categories_cache = all_categories
        self._categories_by_group = categories_by_group

        # Identifier les catégories de revenus et dépenses
        income_groups = ["Income", "Revenus"]
        self._income_categories_cache = []
        self._expense_categories_cache = []

        for group_name, cats in categories_by_group.items():
            if group_name in income_groups:
                self._income_categories_cache.extend(cats)
            else:
                self._expense_categories_cache.extend(cats)

        self._cache_timestamp = datetime.now()

        logger.info(f"Cache mis à jour: {len(self._categories_cache)} catégories "
                   f"({len(self._expense_categories_cache)} dépenses, "
                   f"{len(self._income_categories_cache)} revenus)")

    @property
    def CATEGORIES(self) -> List[str]:
        """Retourne toutes les catégories (compatibilité avec ancien code)"""
        self._load_categories()
        return self._categories_cache or []

    @property
    def EXPENSE_CATEGORIES(self) -> List[str]:
        """Retourne les catégories de dépenses (compatibilité avec ancien code)"""
        self._load_categories()
        return self._expense_categories_cache or []

    @property
    def INCOME_CATEGORIES(self) -> List[str]:
        """Retourne les catégories de revenus (compatibilité avec ancien code)"""
        self._load_categories()
        return self._income_categories_cache or []

    # Mapping des termes ambigus vers catégories (basé sur les groupes dynamiques)
    @property
    def TERM_MAPPINGS(self) -> Dict[str, List[str]]:
        """Mapping des termes vers catégories"""
        self._load_categories()

        # Mapping basique
        mappings = {
            "achats": self.EXPENSE_CATEGORIES,
            "alimentation": [],
            "restaurants": [],
            "transport": [],
            "santé": [],
            "divertissement": [],
            "beauté": [],
            "maison": [],
            "revenus": self.INCOME_CATEGORIES
        }

        # Enrichir avec les groupes dynamiques
        if self._categories_by_group:
            food_groups = ["Food & Dining", "Vie quotidienne"]
            transport_groups = ["Transportation", "Transport"]
            health_groups = ["Health & Medicine", "Santé"]
            entertainment_groups = ["Entertainment", "Loisirs", "Divertissement"]
            personal_care_groups = ["Personal care"]
            home_groups = ["Bills & Utilities", "Charges fixes", "Maison"]

            for group_name, cats in self._categories_by_group.items():
                if group_name in food_groups:
                    mappings["alimentation"].extend(cats)
                    if "Restaurant" in group_name or any("Restaurant" in c for c in cats):
                        mappings["restaurants"].extend([c for c in cats if "Restaurant" in c])
                if group_name in transport_groups:
                    mappings["transport"].extend(cats)
                if group_name in health_groups:
                    mappings["santé"].extend(cats)
                if group_name in entertainment_groups:
                    mappings["divertissement"].extend(cats)
                if group_name in personal_care_groups:
                    mappings["beauté"].extend(cats)
                if group_name in home_groups:
                    mappings["maison"].extend(cats)

            # Dédupliquer
            for key in mappings:
                mappings[key] = list(set(mappings[key]))

        return mappings

    def get_all_categories(self) -> List[str]:
        """Retourne toutes les catégories disponibles"""
        return self.CATEGORIES.copy()
    
    def get_expense_categories(self) -> List[str]:
        """Retourne uniquement les catégories de dépenses"""
        return self.EXPENSE_CATEGORIES.copy()
    
    def get_income_categories(self) -> List[str]:
        """Retourne uniquement les catégories de revenus"""
        return self.INCOME_CATEGORIES.copy()
    
    def get_categories_for_term(self, term: str) -> List[str]:
        """
        Retourne les catégories correspondant à un terme donné
        
        Args:
            term: Le terme à analyser (ex: "achats", "alimentation", etc.)
            
        Returns:
            Liste des catégories correspondantes
        """
        term_lower = term.lower().strip()
        
        # Mapping direct
        if term_lower in self.TERM_MAPPINGS:
            return self.TERM_MAPPINGS[term_lower].copy()
        
        # Recherche par similarité dans les noms de catégories
        matches = []
        for category in self.CATEGORIES:
            if term_lower in category.lower():
                matches.append(category)
        
        if matches:
            return matches
            
        # Si pas de correspondance, retourner toutes les dépenses pour les termes génériques
        generic_expense_terms = ["dépenses", "sorties", "débits"]
        if term_lower in generic_expense_terms:
            return self.EXPENSE_CATEGORIES.copy()
            
        # Par défaut, retourner une liste vide
        logger.warning(f"Aucune catégorie trouvée pour le terme: {term}")
        return []
    
    def is_expense_term(self, term: str) -> bool:
        """Vérifie si un terme correspond à une dépense"""
        categories = self.get_categories_for_term(term)
        return any(cat in self.EXPENSE_CATEGORIES for cat in categories)
    
    def is_income_term(self, term: str) -> bool:
        """Vérifie si un terme correspond à un revenu"""
        categories = self.get_categories_for_term(term)
        return any(cat in self.INCOME_CATEGORIES for cat in categories)
    
    def build_categories_context(self) -> str:
        """Construit le contexte des catégories pour le prompt LLM avec groupes"""
        self._load_categories()

        # Nouvelles définitions des achats et abonnements
        purchase_categories = [
            "Carburant", "Transport", "Loisirs", "Entretien maison",
            "achats en ligne", "Alimentation", "Vêtements"
        ]

        subscription_categories = [
            "streaming", "Téléphones/internet", "Services", "Abonnements"
        ]

        # Construire le contexte avec groupes
        context = "=== CATÉGORIES DISPONIBLES EN BASE (avec leurs groupes) ===\n\n"

        # Afficher par groupe
        if self._categories_by_group:
            for group_name, categories in sorted(self._categories_by_group.items()):
                context += f"📂 Groupe: {group_name}\n"
                context += f"   Catégories: {', '.join(categories)}\n\n"

        context += f"\n💡 TOTAL: {len(self.CATEGORIES)} catégories réparties en {len(self._categories_by_group)} groupes\n\n"

        context += "=== RÈGLES IMPORTANTES ===\n\n"

        context += "🎯 RÈGLE MARCHANDS vs CATÉGORIES:\n"
        context += "• Si un MARCHAND précis est mentionné → utiliser merchant: '[nom]' (PAS de categories)\n"
        context += "• Si la requête est VAGUE → utiliser categories: [liste]\n\n"

        context += "🛍️ RÈGLES POUR 'ACHATS':\n"
        context += f"• 'Mes achats' (sans marchand) → categories: {purchase_categories}\n"
        context += "• Ces catégories regroupent: Carburant, Transport, Loisirs, Entretien maison, achats en ligne, Alimentation, Vêtements\n"
        context += "• EXCEPTION: 'Mes achats chez [marchand]' → merchant: '[marchand]' (PAS de categories)\n\n"

        context += "📺 RÈGLES POUR 'ABONNEMENTS':\n"
        context += f"• 'Mes abonnements' (sans marchand) → categories: {subscription_categories}\n"
        context += "• Ces catégories regroupent: streaming, Téléphones/internet, Services, Abonnements\n"
        context += "• EXCEPTION: 'Mes abonnements Netflix' → merchant: 'Netflix' (PAS de categories)\n\n"

        context += "⚠️ IMPORTANT:\n"
        context += "• Les GROUPES sont informatifs (ex: 'Vie quotidienne') mais ce sont les CATÉGORIES qui doivent être retournées\n"
        context += "• Parfois les groupes sont plus parlants mais TOUJOURS retourner les catégories, pas les groupes\n"
        context += "• Utiliser UNIQUEMENT les catégories listées ci-dessus\n"

        return context


# Instance globale
category_service = CategoryService()