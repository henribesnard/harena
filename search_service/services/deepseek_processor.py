"""
Module de traitement des requêtes avec DeepSeek.

Ce module fournit des fonctionnalités pour l'analyse et l'enrichissement
des requêtes de recherche en utilisant le modèle de raisonnement DeepSeek.
"""
import logging
import json
from typing import Dict, Any, Optional
import httpx
import re

from search_service.schemas.query import (
    SearchQuery, FilterSet, DateRange, AmountRange, 
    AggregationType, GroupBy, OperationType, AggregationRequest
)
from search_service.core.config import settings
from search_service.storage.memory_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

# Vérification conditionnelle de OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("Le module OpenAI n'est pas disponible. Installation nécessaire pour DeepSeek: pip install openai")

async def process_query_with_deepseek(query: SearchQuery, user_id: int) -> SearchQuery:
    """
    Utilise DeepSeek pour traiter et enrichir la requête.
    
    Args:
        query: Requête de recherche originale
        user_id: ID de l'utilisateur
        
    Returns:
        Requête enrichie et structurée
    """
    # Vérifier si DeepSeek et OpenAI sont configurés
    if not settings.DEEPSEEK_API_KEY or not OPENAI_AVAILABLE:
        logger.warning("DeepSeek n'est pas configuré ou OpenAI n'est pas disponible. Traitement basique utilisé.")
        if not query.query.expanded_text:
            query.query.expanded_text = query.query.text
        return query
    
    # Vérifier le cache
    cache_key = f"deepseek_process:{user_id}:{query.query.text}"
    cached_result = await get_cache(cache_key)
    
    if cached_result:
        logger.debug(f"Résultat DeepSeek récupéré du cache pour '{query.query.text[:30]}...'")
        return cached_result
    
    try:
        # Initialiser le client OpenAI avec les paramètres DeepSeek
        client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        
        # Construire le prompt pour DeepSeek
        prompt = f"""
        En tant qu'assistant financier pour la plateforme Harena, analyse cette requête:

        Requête: "{query.query.text}"
        
        1. Décompose et comprends l'intention financière de la requête.
        2. Extrais les filtres et paramètres d'agrégation en suivant exactement ce format JSON:
        
        ```json
        {{
          "expanded_text": "Version expansée et clarifiée de la requête",
          "filters": {{
            "date_range": {{"relative": "last_month"}} OU {{"start": "2025-01-01", "end": "2025-01-31"}},
            "amount_range": {{"min": 50.0, "max": 100.0}},
            "categories": ["alimentation", "loisirs"],
            "merchants": ["amazon", "carrefour"],
            "operation_types": ["debit", "credit"]
          }},
          "aggregation": {{
            "type": "sum",
            "group_by": "category",
            "field": "amount"
          }}
        }}
        ```
        
        Règles importantes:
        - Pour date_range, utilise les valeurs relatives suivantes quand c'est pertinent: "last_week", "last_month", "this_month", "last_year", "this_year", "last_30_days", "today", "yesterday".
        - Pour operation_types, utilise "debit" pour les dépenses et "credit" pour les revenus.
        - Pour le type d'agrégation, utilise: "sum", "avg", "count", "min", "max", "ratio".
        - Pour group_by, utilise: "category", "merchant", "day", "week", "month", "year".
        - Ne retourne que le JSON, sans texte avant ou après.
        """
        
        # Appel au modèle de raisonnement de DeepSeek
        response = client.chat.completions.create(
            model=settings.DEEPSEEK_REASONER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.DEEPSEEK_TEMPERATURE,
            max_tokens=settings.DEEPSEEK_MAX_TOKENS,
            top_p=settings.DEEPSEEK_TOP_P,
        )
        
        # Extraire la réponse
        response_text = response.choices[0].message.content
        
        # Extraire le JSON de la réponse
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
        else:
            # Si le format ```json n'est pas trouvé, essayer de parser directement
            json_text = response_text.strip()
        
        try:
            parsed_data = json.loads(json_text)
            
            # Mettre à jour la requête avec les données analysées
            if "expanded_text" in parsed_data:
                query.query.expanded_text = parsed_data["expanded_text"]
                
            if "filters" in parsed_data:
                filters_data = parsed_data["filters"]
                query.filters = query.filters or FilterSet()
                    
                # Traiter les filtres de date
                if "date_range" in filters_data:
                    date_range_data = filters_data["date_range"]
                    query.filters.date_range = DateRange(**date_range_data)
                    
                # Traiter les filtres de montant
                if "amount_range" in filters_data:
                    amount_range_data = filters_data["amount_range"]
                    query.filters.amount_range = AmountRange(**amount_range_data)
                
                # Traiter les filtres de catégorie
                if "categories" in filters_data:
                    query.filters.categories = filters_data["categories"]
                
                # Traiter les filtres de marchand
                if "merchants" in filters_data:
                    query.filters.merchants = filters_data["merchants"]
                
                # Traiter les filtres de type d'opération
                if "operation_types" in filters_data:
                    query.filters.operation_types = [
                        OperationType(op_type) 
                        for op_type in filters_data["operation_types"]
                    ]
                        
            # Traiter l'agrégation
            if "aggregation" in parsed_data:
                agg_data = parsed_data["aggregation"]
                query.aggregation = AggregationRequest(
                    type=AggregationType(agg_data["type"]),
                    group_by=GroupBy(agg_data["group_by"]) if "group_by" in agg_data else None,
                    field=agg_data.get("field", "amount")
                )
            
            logger.info(f"Requête traitée avec succès par DeepSeek: {query.query.expanded_text or query.query.text}")
            
            # Mettre en cache le résultat
            await set_cache(cache_key, query, ttl=3600)
            
            return query
        except json.JSONDecodeError as je:
            logger.error(f"Erreur lors du décodage JSON de la réponse DeepSeek: {je}")
            logger.debug(f"Texte JSON problématique: {json_text}")
            # Continuer avec la requête non modifiée
            if not query.query.expanded_text:
                query.query.expanded_text = query.query.text
            return query
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement avec DeepSeek: {str(e)}", exc_info=True)
        # Continuer avec la requête non modifiée
        if not query.query.expanded_text:
            query.query.expanded_text = query.query.text
        return query