"""
Endpoints pour les catégories de transactions.

Ce module expose les endpoints pour accéder aux catégories 
fournies par Bridge API et stockées dans la base de données.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from conversation_service.api.dependencies import get_db
from user_service.api.deps import get_current_active_user, get_current_active_superuser
from db_service.models.user import User
from db_service.models.sync import BridgeCategory
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_categories(
    parent_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère toutes les catégories, éventuellement filtrées par catégorie parente.
    
    Args:
        parent_id: Filtrer par ID de catégorie parente
        
    Returns:
        List: Liste des catégories
    """
    # Construire la requête de base
    query = db.query(BridgeCategory)
    
    # Appliquer le filtre par parent si nécessaire
    if parent_id is not None:
        query = query.filter(BridgeCategory.parent_id == parent_id)
    
    # Exécuter la requête
    categories = query.all()
    
    # Organiser les catégories en hiérarchie
    if parent_id is None:
        # Si pas de filtre, organiser en hiérarchie
        return organize_categories_hierarchy(categories)
    else:
        # Sinon, retourner la liste plate
        return [format_category(cat) for cat in categories]

@router.get("/{category_id}")
async def get_category(
    category_id: int,
    include_children: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère une catégorie spécifique par son ID.
    
    Args:
        category_id: ID de la catégorie Bridge
        include_children: Inclure les sous-catégories
        
    Returns:
        Dict: Détails de la catégorie
    """
    # Récupérer la catégorie
    category = db.query(BridgeCategory).filter(
        BridgeCategory.bridge_category_id == category_id
    ).first()
    
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found"
        )
    
    # Formater la réponse
    result = format_category(category)
    
    # Inclure les sous-catégories si demandé
    if include_children:
        children = db.query(BridgeCategory).filter(
            BridgeCategory.parent_id == category_id
        ).all()
        
        result["children"] = [format_category(child) for child in children]
    
    return result

@router.post("/refresh")
async def refresh_categories(
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
):
    """
    Rafraîchit les catégories depuis Bridge API (admin uniquement).
    
    Returns:
        Dict: Résultat du rafraîchissement
    """
    try:
        # Récupérer le token Bridge
        from user_service.services.bridge import get_bridge_token
        token_data = await get_bridge_token(db, current_user.id)
        access_token = token_data["access_token"]
        
        # Récupérer les catégories depuis Bridge API
        from user_service.services.bridge import get_bridge_categories
        categories = await get_bridge_categories(db, current_user.id)
        
        if not categories:
            return {
                "status": "warning",
                "message": "No categories available from Bridge API"
            }
        
        # Stocker les catégories
        from sync_service.sync_manager.category_handler import store_bridge_categories
        result = await store_bridge_categories(db, categories)
        
        return {
            "status": result.get("status", "error"),
            "message": "Categories refreshed successfully" if result.get("status") == "success" else "Failed to refresh categories",
            "categories_created": result.get("categories_created", 0),
            "categories_updated": result.get("categories_updated", 0)
        }
    except Exception as e:
        logger.error(f"Error refreshing categories: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh categories: {str(e)}"
        )

# Fonctions utilitaires pour formater les réponses
def format_category(category: BridgeCategory) -> Dict[str, Any]:
    """
    Formate une catégorie pour la réponse API.
    
    Args:
        category: Catégorie depuis la BDD
        
    Returns:
        Dict: Catégorie formatée
    """
    return {
        "id": category.bridge_category_id,
        "name": category.name,
        "parent_id": category.parent_id,
        "parent_name": category.parent_name
    }

def organize_categories_hierarchy(categories: List[BridgeCategory]) -> List[Dict[str, Any]]:
    """
    Organise les catégories en hiérarchie.
    
    Args:
        categories: Liste de toutes les catégories
        
    Returns:
        List: Catégories organisées en hiérarchie
    """
    # Créer un dictionnaire pour accéder aux catégories par ID
    categories_dict = {cat.bridge_category_id: format_category(cat) for cat in categories}
    
    # Identifier les catégories racines (sans parent) et les enfants
    root_categories = []
    child_categories = {}
    
    for category in categories:
        if category.parent_id is None:
            # Catégorie racine
            cat_dict = categories_dict[category.bridge_category_id]
            cat_dict["children"] = []
            root_categories.append(cat_dict)
        else:
            # Catégorie enfant
            if category.parent_id not in child_categories:
                child_categories[category.parent_id] = []
            
            child_categories[category.parent_id].append(categories_dict[category.bridge_category_id])
    
    # Attacher les enfants à leurs parents
    for parent_id, children in child_categories.items():
        if parent_id in categories_dict:
            categories_dict[parent_id]["children"] = children
    
    return root_categories