"""
Service de gestion des catégories pour enrichment_service
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
import os

# Ajouter le répertoire racine au chemin Python  
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from config_service.config import settings
    from sqlalchemy import create_engine, text
    config_available = True
except ImportError:
    config_available = False
    settings = None

logger = logging.getLogger(__name__)

@dataclass
class Category:
    """Représentation d'une catégorie"""
    category_id: int
    category_name: str
    group_id: int
    group_name: str

class CategoryService:
    """Service de gestion des catégories avec cache"""
    
    def __init__(self):
        self._categories_cache: Dict[int, Category] = {}
        self._cache_expiry: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)  # Cache valide 1 heure
        self._engine = None
    
    def _get_database_engine(self):
        """Créer ou récupérer le moteur de base de données"""
        if self._engine is None:
            if config_available and settings:
                database_url = settings.DATABASE_URL or getattr(settings, "SQLALCHEMY_DATABASE_URI", "")
                if database_url.startswith("postgres://"):
                    database_url = database_url.replace("postgres://", "postgresql://", 1)
            else:
                database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/harena_db")
                database_url = database_url.replace("postgres://", "postgresql://")
            
            self._engine = create_engine(database_url, pool_pre_ping=True)
            
        return self._engine
    
    def _is_cache_valid(self) -> bool:
        """Vérifier si le cache est encore valide"""
        return (
            self._cache_expiry is not None and 
            datetime.utcnow() < self._cache_expiry and 
            len(self._categories_cache) > 0
        )
    
    def _refresh_cache(self) -> None:
        """Rafraîchir le cache depuis PostgreSQL"""
        try:
            engine = self._get_database_engine()
            
            with engine.connect() as conn:
                result = conn.execute(text(
                    """
                    SELECT c.category_id, c.category_name, c.group_id, cg.group_name
                    FROM categories c
                    LEFT JOIN category_groups cg ON c.group_id = cg.group_id
                    ORDER BY cg.group_name, c.category_name
                    """
                ))
                
                # Vider le cache actuel
                self._categories_cache.clear()
                
                # Remplir le cache
                for row in result:
                    category = Category(
                        category_id=row.category_id,
                        category_name=row.category_name,
                        group_id=row.group_id,
                        group_name=row.group_name
                    )
                    self._categories_cache[category.category_id] = category
                
                # Mettre à jour l'expiration du cache
                self._cache_expiry = datetime.utcnow() + self._cache_ttl
                
                logger.info(f"Cache catégories rafraîchi: {len(self._categories_cache)} catégories chargées")
                
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement du cache catégories: {e}")
            raise
    
    def get_category_name(self, category_id: Optional[int]) -> Optional[str]:
        """Récupérer le nom d'une catégorie par son ID"""
        if category_id is None:
            return None
        
        # Vérifier et rafraîchir le cache si nécessaire
        if not self._is_cache_valid():
            try:
                self._refresh_cache()
            except Exception as e:
                logger.error(f"Impossible de rafraîchir le cache: {e}")
                return None
        
        # Récupérer depuis le cache
        category = self._categories_cache.get(category_id)
        return category.category_name if category else None
    
    def get_category(self, category_id: Optional[int]) -> Optional[Category]:
        """Récupérer une catégorie complète par son ID"""
        if category_id is None:
            return None
        
        # Vérifier et rafraîchir le cache si nécessaire
        if not self._is_cache_valid():
            try:
                self._refresh_cache()
            except Exception as e:
                logger.error(f"Impossible de rafraîchir le cache: {e}")
                return None
        
        return self._categories_cache.get(category_id)
    
    def get_all_categories(self) -> Dict[int, Category]:
        """Récupérer toutes les catégories"""
        # Vérifier et rafraîchir le cache si nécessaire
        if not self._is_cache_valid():
            try:
                self._refresh_cache()
            except Exception as e:
                logger.error(f"Impossible de rafraîchir le cache: {e}")
                return {}
        
        return self._categories_cache.copy()
    
    def search_category_by_name(self, name: str) -> Optional[Category]:
        """Chercher une catégorie par nom (recherche approximative)"""
        if not name:
            return None
        
        # Vérifier et rafraîchir le cache si nécessaire
        if not self._is_cache_valid():
            try:
                self._refresh_cache()
            except Exception as e:
                logger.error(f"Impossible de rafraîchir le cache: {e}")
                return None
        
        name_lower = name.lower()
        
        # Recherche exacte d'abord
        for category in self._categories_cache.values():
            if category.category_name.lower() == name_lower:
                return category
        
        # Recherche partielle
        for category in self._categories_cache.values():
            if name_lower in category.category_name.lower():
                return category
        
        return None
    
    def get_categories_by_group(self, group_name: str) -> List[Category]:
        """Récupérer toutes les catégories d'un groupe"""
        # Vérifier et rafraîchir le cache si nécessaire
        if not self._is_cache_valid():
            try:
                self._refresh_cache()
            except Exception as e:
                logger.error(f"Impossible de rafraîchir le cache: {e}")
                return []
        
        return [
            category for category in self._categories_cache.values()
            if category.group_name.lower() == group_name.lower()
        ]
    
    def invalidate_cache(self):
        """Forcer l'invalidation du cache (pour les tests ou mises à jour)"""
        self._cache_expiry = None
        self._categories_cache.clear()
        logger.info("Cache catégories invalidé")

# Instance globale singleton
_category_service = None

def get_category_service() -> CategoryService:
    """Récupérer l'instance singleton du service de catégories"""
    global _category_service
    if _category_service is None:
        _category_service = CategoryService()
    return _category_service

# Tests unitaires
async def test_category_service():
    """Test du service de catégories"""
    print("=== TEST CATEGORY SERVICE ===")
    
    service = get_category_service()
    
    # Test 1: Récupérer une catégorie par ID
    category = service.get_category(83)  # Restaurants
    if category:
        print(f"Catégorie ID 83: {category.category_name} (Groupe: {category.group_name})")
    else:
        print("Catégorie ID 83 non trouvée")
    
    # Test 2: Récupérer nom de catégorie
    name = service.get_category_name(235)  # Hairdresser
    print(f"Nom catégorie ID 235: {name}")
    
    # Test 3: Recherche par nom
    found = service.search_category_by_name("Restaurants")
    if found:
        print(f"Recherche 'Restaurants': ID {found.category_id}")
    
    # Test 4: Catégories par groupe
    food_categories = service.get_categories_by_group("Food & Dining")
    print(f"Catégories Food & Dining: {len(food_categories)}")
    for cat in food_categories[:3]:
        print(f"  - {cat.category_name}")
    
    # Test 5: Toutes les catégories
    all_cats = service.get_all_categories()
    print(f"Total catégories en cache: {len(all_cats)}")

if __name__ == "__main__":
    asyncio.run(test_category_service())