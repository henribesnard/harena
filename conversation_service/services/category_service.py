"""
Category Service pour le Conversation Service

Fournit la logique d'arbitrage des catégories pour l'extraction d'entités.
Basé sur les vraies catégories récupérées depuis PostgreSQL.
"""

from typing import List, Set, Optional
import logging

logger = logging.getLogger(__name__)

class CategoryService:
    """Service pour gérer les catégories de transactions"""
    
    # Catégories récupérées depuis PostgreSQL (via enrichment_service)
    CATEGORIES = [
        "Bank Fees",
        "Beauty care", 
        "Books & Media",
        "Car Maintenance",
        "Clothing",
        "Coffee shop",
        "Concerts & Shows",
        "Cosmetics",
        "Dentist",
        "Doctor Visits",
        "Electricity",
        "Electronics",
        "Fast foods",
        "Food - Others",
        "Freelance",
        "Fuel",
        "Gaming",
        "Government Benefits",
        "Hairdresser",
        "Home & Garden",
        "Insurance",
        "Internet/Phone",
        "Investment Returns",
        "Medical Equipment",
        "Medical Insurance",
        "Movies & Cinema",
        "Online Shopping",
        "Other Income",
        "Parking",
        "Personal care - Others",
        "Pharmacy",
        "Public Transportation",
        "Restaurants",
        "Salary",
        "Spa & Massage",
        "Sports Events",
        "Streaming Services",
        "Supermarkets / Groceries",
        "Taxi/Uber",
        "Water"
    ]
    
    # Groupes logiques pour l'arbitrage
    EXPENSE_CATEGORIES = [
        "Bank Fees", "Beauty care", "Books & Media", "Car Maintenance", "Clothing",
        "Coffee shop", "Concerts & Shows", "Cosmetics", "Dentist", "Doctor Visits",
        "Electricity", "Electronics", "Fast foods", "Food - Others", "Fuel",
        "Gaming", "Hairdresser", "Home & Garden", "Insurance", "Internet/Phone",
        "Medical Equipment", "Medical Insurance", "Movies & Cinema", "Online Shopping",
        "Parking", "Personal care - Others", "Pharmacy", "Public Transportation",
        "Restaurants", "Spa & Massage", "Sports Events", "Streaming Services",
        "Supermarkets / Groceries", "Taxi/Uber", "Water"
    ]
    
    INCOME_CATEGORIES = [
        "Freelance", "Government Benefits", "Investment Returns", "Other Income", "Salary"
    ]
    
    # Mapping des termes ambigus vers catégories
    TERM_MAPPINGS = {
        "achats": EXPENSE_CATEGORIES,  # "achats" = toutes les dépenses
        "courses": ["Supermarkets / Groceries"],
        "alimentation": ["Supermarkets / Groceries", "Restaurants", "Fast foods", "Coffee shop", "Food - Others"],
        "restaurants": ["Restaurants"],
        "transport": ["Public Transportation", "Taxi/Uber", "Fuel", "Car Maintenance", "Parking"],
        "santé": ["Doctor Visits", "Dentist", "Pharmacy", "Medical Equipment", "Medical Insurance"],
        "divertissement": ["Movies & Cinema", "Concerts & Shows", "Gaming", "Sports Events", "Streaming Services"],
        "beauté": ["Beauty care", "Cosmetics", "Hairdresser", "Spa & Massage"],
        "maison": ["Home & Garden", "Electricity", "Water", "Internet/Phone", "Insurance"],
        "revenus": INCOME_CATEGORIES
    }
    
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
        """Construit le contexte des catégories pour le prompt LLM"""
        context = "CATÉGORIES DISPONIBLES:\n"
        context += f"• Dépenses ({len(self.EXPENSE_CATEGORIES)}): {', '.join(self.EXPENSE_CATEGORIES[:10])}...\n"
        context += f"• Revenus ({len(self.INCOME_CATEGORIES)}): {', '.join(self.INCOME_CATEGORIES)}\n\n"
        
        context += "LOGIQUE D'ARBITRAGE:\n"
        context += "• 'achats' = TOUTES les catégories de dépenses (sauf revenus)\n"
        context += "• 'alimentation' = Supermarkets, Restaurants, Fast foods, etc.\n"
        context += "• Terme spécifique = catégorie exacte correspondante\n"
        
        return context


# Instance globale
category_service = CategoryService()