"""
Module de filtrage structuré des résultats.

Ce module applique des filtres précis sur les données structurées
après le reranking pour garantir la conformité aux critères spécifiés.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from search_service.schemas.query import FilterSet, DateRange

logger = logging.getLogger(__name__)

async def apply_filters(
    results: List[Dict[str, Any]],
    filters: Optional[FilterSet] = None
) -> List[Dict[str, Any]]:
    """
    Applique des filtres structurés sur les résultats.
    
    Args:
        results: Liste des résultats à filtrer
        filters: Ensemble de filtres à appliquer
        
    Returns:
        Liste filtrée des résultats
    """
    # Si pas de filtres, retourner les résultats inchangés
    if not filters:
        return results
    
    logger.debug(f"Application de filtres structurés sur {len(results)} résultats")
    
    filtered_results = results.copy()
    
    # Appliquer les différents types de filtres
    if filters.date_range:
        filtered_results = filter_by_date(filtered_results, filters.date_range)
    
    if filters.amount_range:
        filtered_results = filter_by_amount(filtered_results, filters.amount_range)
    
    if filters.categories:
        filtered_results = filter_by_category(filtered_results, filters.categories)
    
    if filters.merchants:
        filtered_results = filter_by_merchant(filtered_results, filters.merchants)
    
    if filters.operation_types:
        filtered_results = filter_by_operation_type(filtered_results, filters.operation_types)
    
    if filters.custom_filters:
        filtered_results = apply_custom_filters(filtered_results, filters.custom_filters)
    
    logger.info(f"Filtrage terminé: {len(filtered_results)} résultats après filtrage")
    return filtered_results

def filter_by_date(
    results: List[Dict[str, Any]],
    date_range: DateRange
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats par plage de dates.
    
    Args:
        results: Liste des résultats à filtrer
        date_range: Plage de dates à appliquer
        
    Returns:
        Liste filtrée des résultats
    """
    # Si la date_range contient une expression relative, la résoudre
    start_date = date_range.start
    end_date = date_range.end
    
    if date_range.relative:
        start_date, end_date = resolve_relative_date_expression(date_range.relative)
    
    # Si ni start ni end n'est défini, retourner les résultats inchangés
    if not start_date and not end_date:
        return results
    
    filtered = []
    for result in results:
        content = result["content"]
        transaction_date_str = content.get("transaction_date")
        
        if not transaction_date_str:
            continue
        
        # Convertir la date de transaction en objet datetime
        try:
            # Gérer différents formats possibles
            if isinstance(transaction_date_str, datetime):
                transaction_date = transaction_date_str
            else:
                transaction_date = datetime.fromisoformat(transaction_date_str.replace('Z', '+00:00'))
            
            # Appliquer le filtre de date
            matches = True
            if start_date and transaction_date < start_date:
                matches = False
            if end_date and transaction_date > end_date:
                matches = False
            
            if matches:
                filtered.append(result)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Erreur de conversion de date '{transaction_date_str}': {str(e)}")
            # En cas d'erreur, inclure le résultat par prudence
            filtered.append(result)
    
    return filtered

def resolve_relative_date_expression(expression: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Résout une expression de date relative en dates concrètes.
    
    Args:
        expression: Expression de date relative (ex: "last_week", "this_month")
        
    Returns:
        Tuple (date_début, date_fin)
    """
    now = datetime.now()
    today = datetime(now.year, now.month, now.day)
    
    # Expressions pour les semaines
    if expression == "last_week":
        # Lundi de la semaine précédente jusqu'au dimanche
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return start, end
    
    elif expression == "this_week":
        # Lundi de cette semaine jusqu'à aujourd'hui
        start = today - timedelta(days=today.weekday())
        return start, today
    
    # Expressions pour les mois
    elif expression == "last_month":
        # Premier jour du mois précédent jusqu'au dernier jour
        if now.month == 1:
            start = datetime(now.year - 1, 12, 1)
            end = datetime(now.year, 1, 1) - timedelta(days=1)
        else:
            start = datetime(now.year, now.month - 1, 1)
            end = datetime(now.year, now.month, 1) - timedelta(days=1)
        return start, end
    
    elif expression == "this_month":
        # Premier jour du mois en cours jusqu'à aujourd'hui
        start = datetime(now.year, now.month, 1)
        return start, today
    
    # Expressions pour les années
    elif expression == "last_year":
        # Année précédente complète
        start = datetime(now.year - 1, 1, 1)
        end = datetime(now.year, 1, 1) - timedelta(days=1)
        return start, end
    
    elif expression == "this_year":
        # Début de l'année en cours jusqu'à aujourd'hui
        start = datetime(now.year, 1, 1)
        return start, today
    
    # Expressions pour des périodes plus courtes
    elif expression == "yesterday":
        # Hier uniquement
        start = today - timedelta(days=1)
        end = today - timedelta(seconds=1)
        return start, end
    
    elif expression == "today":
        # Aujourd'hui uniquement
        return today, datetime.now()
    
    elif expression == "last_30_days":
        # 30 derniers jours
        start = today - timedelta(days=30)
        return start, today
    
    elif expression == "last_90_days":
        # 90 derniers jours
        start = today - timedelta(days=90)
        return start, today
    
    # Expression inconnue
    logger.warning(f"Expression de date relative inconnue: {expression}")
    return None, None

def filter_by_amount(
    results: List[Dict[str, Any]],
    amount_range: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats par plage de montants.
    
    Args:
        results: Liste des résultats à filtrer
        amount_range: Plage de montants à appliquer
        
    Returns:
        Liste filtrée des résultats
    """
    min_amount = amount_range.min
    max_amount = amount_range.max
    
    # Si ni min ni max n'est défini, retourner les résultats inchangés
    if min_amount is None and max_amount is None:
        return results
    
    filtered = []
    for result in results:
        content = result["content"]
        amount = content.get("amount")
        
        if amount is None:
            continue
        
        # Convertir en float si nécessaire
        if not isinstance(amount, (int, float)):
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                continue
        
        # Appliquer le filtre de montant
        matches = True
        if min_amount is not None and amount < min_amount:
            matches = False
        if max_amount is not None and amount > max_amount:
            matches = False
        
        if matches:
            filtered.append(result)
    
    return filtered

def filter_by_category(
    results: List[Dict[str, Any]],
    categories: List[str]
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats par catégories.
    
    Args:
        results: Liste des résultats à filtrer
        categories: Liste des catégories à inclure
        
    Returns:
        Liste filtrée des résultats
    """
    if not categories:
        return results
    
    # Normaliser les catégories pour la comparaison
    normalized_categories = [c.lower() for c in categories]
    
    filtered = []
    for result in results:
        content = result["content"]
        category = content.get("category")
        
        if not category:
            continue
        
        # Vérifier si la catégorie correspond
        if category.lower() in normalized_categories:
            filtered.append(result)
            
        # Vérifier aussi le category_id pour les cas où seul l'ID est présent
        elif "category_id" in content:
            category_id = content["category_id"]
            # Vérifier si l'ID de catégorie correspond à un ID dans les catégories demandées
            # Cela nécessiterait une correspondance ID-nom plus sophistiquée en production
            for cat in categories:
                if cat.isdigit() and int(cat) == category_id:
                    filtered.append(result)
                    break
    
    return filtered

def filter_by_merchant(
    results: List[Dict[str, Any]],
    merchants: List[str]
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats par marchands.
    
    Args:
        results: Liste des résultats à filtrer
        merchants: Liste des marchands à inclure
        
    Returns:
        Liste filtrée des résultats
    """
    if not merchants:
        return results
    
    # Normaliser les noms de marchands pour la comparaison
    normalized_merchants = [m.lower() for m in merchants]
    
    filtered = []
    for result in results:
        content = result["content"]
        merchant_name = content.get("merchant_name", "")
        
        if not merchant_name:
            continue
        
        # Recherche non-stricte (contient) pour les marchands
        merchant_matches = False
        merchant_name_lower = merchant_name.lower()
        
        for m in normalized_merchants:
            if m in merchant_name_lower or merchant_name_lower in m:
                merchant_matches = True
                break
        
        if merchant_matches:
            filtered.append(result)
    
    return filtered

def filter_by_operation_type(
    results: List[Dict[str, Any]],
    operation_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Filtre les résultats par type d'opération (crédit/débit).
    
    Args:
        results: Liste des résultats à filtrer
        operation_types: Liste des types d'opération à inclure
        
    Returns:
        Liste filtrée des résultats
    """
    if not operation_types or "all" in operation_types:
        return results
    
    filtered = []
    for result in results:
        content = result["content"]
        amount = content.get("amount")
        
        if amount is None:
            continue
        
        # Convertir en float si nécessaire
        if not isinstance(amount, (int, float)):
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                continue
        
        # Débit (montant négatif) ou crédit (montant positif)
        if "debit" in operation_types and amount < 0:
            filtered.append(result)
        elif "credit" in operation_types and amount > 0:
            filtered.append(result)
    
    return filtered

def apply_custom_filters(
    results: List[Dict[str, Any]],
    custom_filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Applique des filtres personnalisés sur les résultats.
    
    Args:
        results: Liste des résultats à filtrer
        custom_filters: Filtres personnalisés à appliquer
        
    Returns:
        Liste filtrée des résultats
    """
    if not custom_filters:
        return results
    
    filtered = results.copy()
    
    # Traitement des filtres personnalisés
    for field, value in custom_filters.items():
        filtered = [
            result for result in filtered 
            if field in result["content"] and result["content"][field] == value
        ]
    
    return filtered