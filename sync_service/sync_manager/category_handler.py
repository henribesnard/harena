"""
Gestionnaire des catégories Bridge.

Ce module gère le stockage et la gestion des catégories fournies par Bridge API.
MISE À JOUR: Utilise maintenant la table 'categories' au lieu de 'bridge_categories'
"""

import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional

from db_service.models.sync import Category
from sync_service.utils.logging import get_contextual_logger

logger = logging.getLogger(__name__)

async def store_bridge_categories(db: Session, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stocke les catégories Bridge dans la base de données SQL.
    
    Args:
        db: Session de base de données
        categories: Liste des catégories depuis Bridge API
        
    Returns:
        Dict: Résultat de l'opération
    """
    ctx_logger = get_contextual_logger("sync_service.category_handler")
    ctx_logger.info(f"Stockage de {len(categories)} catégories Bridge")
    
    result = {
        "status": "pending",
        "categories_processed": 0,
        "categories_created": 0,
        "categories_updated": 0
    }
    
    try:
        # Récupérer les IDs des catégories existantes
        existing_category_ids = {cat.category_id for cat in 
                                db.query(Category.category_id).all()}
        
        for category_data in categories:
            # Support des deux formats: Bridge API natif et notre format
            category_id = category_data.get("id") or category_data.get("category_id")
            category_name = category_data.get("name") or category_data.get("category_name")
            
            if not category_id:
                ctx_logger.warning(f"Catégorie sans ID, ignorée: {category_data}")
                continue
                
            try:
                # Vérifier si la catégorie existe déjà
                if category_id in existing_category_ids:
                    # Mise à jour
                    category = db.query(Category).filter(
                        Category.category_id == category_id
                    ).first()
                    
                    if category:
                        category.category_name = category_name or category.category_name
                        category.group_id = category_data.get("group_id", category.group_id)
                        category.group_name = category_data.get("group_name", category.group_name)
                        
                        db.add(category)
                        result["categories_updated"] += 1
                else:
                    # Création
                    new_category = Category(
                        category_id=category_id,
                        category_name=category_name or "",
                        group_id=category_data.get("group_id", 0),
                        group_name=category_data.get("group_name", "")
                    )
                    
                    db.add(new_category)
                    result["categories_created"] += 1
                    existing_category_ids.add(category_id)
                    
                result["categories_processed"] += 1
                
                # Commit périodique pour éviter les transactions trop longues
                if result["categories_processed"] % 100 == 0:
                    db.commit()
                    
            except Exception as e:
                ctx_logger.error(f"Erreur lors du stockage de la catégorie {category_id}: {e}", exc_info=True)
                # Continuer avec la catégorie suivante
        
        # Commit final
        db.commit()
        
        # Mettre à jour le statut final
        result["status"] = "success"
        ctx_logger.info(f"Stockage des catégories terminé: {result['categories_created']} créées, {result['categories_updated']} mises à jour")
        
        return result
    except Exception as e:
        db.rollback()
        ctx_logger.error(f"Erreur lors du stockage des catégories: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)
        return result