"""
Module de calcul d'agrégations financières.

Ce module fournit les fonctionnalités pour effectuer des calculs financiers
sur les résultats de recherche (sommes, moyennes, tendances, etc.).
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from search_service.schemas.query import AggregationRequest, AggregationType, GroupBy, TimeUnit

logger = logging.getLogger(__name__)

async def calculate_aggregations(
    results: List[Dict[str, Any]],
    aggregation: Optional[AggregationRequest] = None
) -> Optional[Dict[str, Any]]:
    """
    Effectue des calculs d'agrégation sur les résultats.
    
    Args:
        results: Liste des résultats à agréger
        aggregation: Configuration de l'agrégation à effectuer
        
    Returns:
        Résultats d'agrégation ou None si aucune agrégation demandée
    """
    if not aggregation or not results:
        return None
    
    logger.debug(f"Calcul d'agrégation de type {aggregation.type} sur {len(results)} résultats")
    
    # Mapper les types d'agrégation aux fonctions correspondantes
    aggregation_functions = {
        AggregationType.SUM: calculate_sum,
        AggregationType.AVG: calculate_average,
        AggregationType.COUNT: calculate_count,
        AggregationType.MIN: calculate_min,
        AggregationType.MAX: calculate_max,
        AggregationType.RATIO: calculate_ratio
    }
    
    # Appeler la fonction d'agrégation appropriée
    if aggregation.type in aggregation_functions:
        return await aggregation_functions[aggregation.type](results, aggregation)
    
    logger.warning(f"Type d'agrégation non supporté: {aggregation.type}")
    return None

async def calculate_sum(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Calcule la somme des valeurs d'un champ dans les résultats.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    group_by = aggregation.group_by or GroupBy.NONE
    
    if group_by == GroupBy.NONE:
        # Somme simple sans groupage
        total = 0.0
        count = 0
        currency = None
        
        for result in results:
            content = result["content"]
            if field in content and content[field] is not None:
                try:
                    value = float(content[field])
                    total += value
                    count += 1
                    # Déterminer la devise (si disponible)
                    if not currency and "currency_code" in content:
                        currency = content["currency_code"]
                except (ValueError, TypeError):
                    continue
        
        return {
            "aggregation_type": "sum",
            "field": field,
            "total": total,
            "count": count,
            "currency": currency
        }
    
    elif group_by == GroupBy.CATEGORY:
        # Somme groupée par catégorie
        category_sums = defaultdict(float)
        category_counts = defaultdict(int)
        currency = None
        
        for result in results:
            content = result["content"]
            if field in content and content[field] is not None:
                category = content.get("category", "Inconnu")
                
                try:
                    value = float(content[field])
                    category_sums[category] += value
                    category_counts[category] += 1
                    # Déterminer la devise (si disponible)
                    if not currency and "currency_code" in content:
                        currency = content["currency_code"]
                except (ValueError, TypeError):
                    continue
        
        # Construire le résultat
        category_results = [
            {
                "category": category,
                "total": total,
                "count": category_counts[category]
            }
            for category, total in category_sums.items()
        ]
        
        # Trier par valeur absolue décroissante
        category_results.sort(key=lambda x: abs(x["total"]), reverse=True)
        
        return {
            "aggregation_type": "sum",
            "field": field,
            "group_by": "category",
            "currency": currency,
            "total": sum(category_sums.values()),
            "categories": category_results
        }
    
    elif group_by in [GroupBy.DAY, GroupBy.WEEK, GroupBy.MONTH, GroupBy.YEAR]:
        # Somme groupée par période
        time_sums = defaultdict(float)
        time_counts = defaultdict(int)
        currency = None
        
        for result in results:
            content = result["content"]
            if field in content and content[field] is not None and "transaction_date" in content:
                try:
                    value = float(content[field])
                    date_str = content["transaction_date"]
                    
                    # Convertir la date
                    if isinstance(date_str, datetime):
                        date = date_str
                    else:
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    # Formater la clé de période selon le groupage
                    if group_by == GroupBy.DAY:
                        period_key = date.strftime("%Y-%m-%d")
                    elif group_by == GroupBy.WEEK:
                        # ISO week format: YYYY-Www
                        period_key = f"{date.isocalendar()[0]}-W{date.isocalendar()[1]:02d}"
                    elif group_by == GroupBy.MONTH:
                        period_key = date.strftime("%Y-%m")
                    else:  # group_by == GroupBy.YEAR
                        period_key = str(date.year)
                    
                    time_sums[period_key] += value
                    time_counts[period_key] += 1
                    
                    # Déterminer la devise (si disponible)
                    if not currency and "currency_code" in content:
                        currency = content["currency_code"]
                        
                except (ValueError, TypeError):
                    continue
        
        # Construire le résultat
        time_results = [
            {
                "period": period,
                "total": total,
                "count": time_counts[period]
            }
            for period, total in time_sums.items()
        ]
        
        # Trier par période chronologiquement
        time_results.sort(key=lambda x: x["period"])
        
        return {
            "aggregation_type": "sum",
            "field": field,
            "group_by": group_by.value,
            "currency": currency,
            "total": sum(time_sums.values()),
            "periods": time_results
        }
    
    # Groupage non supporté
    logger.warning(f"Type de groupage non supporté: {group_by}")
    return {
        "aggregation_type": "sum",
        "field": field,
        "error": f"Unsupported group_by: {group_by}"
    }

async def calculate_average(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Calcule la moyenne des valeurs d'un champ dans les résultats.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    
    # Obtenir d'abord les résultats de somme
    sum_result = await calculate_sum(results, aggregation)
    
    # Cas sans groupage
    if "categories" not in sum_result and "periods" not in sum_result:
        total = sum_result["total"]
        count = sum_result["count"]
        
        if count > 0:
            average = total / count
        else:
            average = 0.0
        
        return {
            "aggregation_type": "average",
            "field": field,
            "average": average,
            "total": total,
            "count": count,
            "currency": sum_result.get("currency")
        }
    
    # Cas avec groupage par catégorie
    elif "categories" in sum_result:
        # Calculer la moyenne pour chaque catégorie
        for category in sum_result["categories"]:
            count = category["count"]
            if count > 0:
                category["average"] = category["total"] / count
            else:
                category["average"] = 0.0
        
        # Calculer la moyenne globale
        total = sum_result["total"]
        total_count = sum(category["count"] for category in sum_result["categories"])
        
        if total_count > 0:
            global_average = total / total_count
        else:
            global_average = 0.0
        
        return {
            "aggregation_type": "average",
            "field": field,
            "group_by": "category",
            "average": global_average,
            "total": total,
            "count": total_count,
            "currency": sum_result.get("currency"),
            "categories": sum_result["categories"]
        }
    
    # Cas avec groupage par période
    elif "periods" in sum_result:
        # Calculer la moyenne pour chaque période
        for period in sum_result["periods"]:
            count = period["count"]
            if count > 0:
                period["average"] = period["total"] / count
            else:
                period["average"] = 0.0
        
        # Calculer la moyenne globale
        total = sum_result["total"]
        total_count = sum(period["count"] for period in sum_result["periods"])
        
        if total_count > 0:
            global_average = total / total_count
        else:
            global_average = 0.0
        
        return {
            "aggregation_type": "average",
            "field": field,
            "group_by": sum_result["group_by"],
            "average": global_average,
            "total": total,
            "count": total_count,
            "currency": sum_result.get("currency"),
            "periods": sum_result["periods"]
        }
    
    # Cas non géré
    return sum_result

async def calculate_count(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Compte le nombre d'occurrences dans les résultats, éventuellement groupés.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    group_by = aggregation.group_by or GroupBy.NONE
    
    if group_by == GroupBy.NONE:
        # Comptage simple
        return {
            "aggregation_type": "count",
            "field": field,
            "count": len(results)
        }
    
    elif group_by == GroupBy.CATEGORY:
        # Comptage par catégorie
        category_counts = defaultdict(int)
        
        for result in results:
            content = result["content"]
            category = content.get("category", "Inconnu")
            category_counts[category] += 1
        
        # Construire le résultat
        category_results = [
            {
                "category": category,
                "count": count
            }
            for category, count in category_counts.items()
        ]
        
        # Trier par nombre décroissant
        category_results.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "aggregation_type": "count",
            "field": field,
            "group_by": "category",
            "total_count": len(results),
            "categories": category_results
        }
    
    elif group_by in [GroupBy.DAY, GroupBy.WEEK, GroupBy.MONTH, GroupBy.YEAR]:
        # Comptage par période
        time_counts = defaultdict(int)
        
        for result in results:
            content = result["content"]
            if "transaction_date" in content:
                try:
                    date_str = content["transaction_date"]
                    
                    # Convertir la date
                    if isinstance(date_str, datetime):
                        date = date_str
                    else:
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    # Formater la clé de période selon le groupage
                    if group_by == GroupBy.DAY:
                        period_key = date.strftime("%Y-%m-%d")
                    elif group_by == GroupBy.WEEK:
                        # ISO week format: YYYY-Www
                        period_key = f"{date.isocalendar()[0]}-W{date.isocalendar()[1]:02d}"
                    elif group_by == GroupBy.MONTH:
                        period_key = date.strftime("%Y-%m")
                    else:  # group_by == GroupBy.YEAR
                        period_key = str(date.year)
                    
                    time_counts[period_key] += 1
                        
                except (ValueError, TypeError):
                    continue
        
        # Construire le résultat
        time_results = [
            {
                "period": period,
                "count": count
            }
            for period, count in time_counts.items()
        ]
        
        # Trier par période chronologiquement
        time_results.sort(key=lambda x: x["period"])
        
        return {
            "aggregation_type": "count",
            "field": field,
            "group_by": group_by.value,
            "total_count": len(results),
            "periods": time_results
        }
    
    # Groupage non supporté
    logger.warning(f"Type de groupage non supporté pour count: {group_by}")
    return {
        "aggregation_type": "count",
        "field": field,
        "count": len(results),
        "error": f"Unsupported group_by: {group_by}"
    }

async def calculate_min(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Trouve la valeur minimale d'un champ dans les résultats.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    group_by = aggregation.group_by or GroupBy.NONE
    
    if group_by == GroupBy.NONE:
        # Minimum global
        values = []
        min_result = None
        currency = None
        
        for result in results:
            content = result["content"]
            if field in content and content[field] is not None:
                try:
                    value = float(content[field])
                    values.append(value)
                    
                    # Conserver le résultat avec la valeur minimale
                    if min_result is None or value < min_result["value"]:
                        min_result = {
                            "value": value,
                            "transaction": content
                        }
                    
                    # Déterminer la devise (si disponible)
                    if not currency and "currency_code" in content:
                        currency = content["currency_code"]
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {
                "aggregation_type": "min",
                "field": field,
                "min": None,
                "count": 0
            }
        
        return {
            "aggregation_type": "min",
            "field": field,
            "min": min(values),
            "min_transaction": min_result["transaction"] if min_result else None,
            "count": len(values),
            "currency": currency
        }
    
    # Pour les autres types de groupage, implémenter selon les besoins
    logger.warning(f"Groupage {group_by} non implémenté pour l'agrégation min")
    return {
        "aggregation_type": "min",
        "field": field,
        "error": f"Grouping not implemented for min: {group_by}"
    }

async def calculate_max(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Trouve la valeur maximale d'un champ dans les résultats.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    group_by = aggregation.group_by or GroupBy.NONE
    
    if group_by == GroupBy.NONE:
        # Maximum global
        values = []
        max_result = None
        currency = None
        
        for result in results:
            content = result["content"]
            if field in content and content[field] is not None:
                try:
                    value = float(content[field])
                    values.append(value)
                    
                    # Conserver le résultat avec la valeur maximale
                    if max_result is None or value > max_result["value"]:
                        max_result = {
                            "value": value,
                            "transaction": content
                        }
                    
                    # Déterminer la devise (si disponible)
                    if not currency and "currency_code" in content:
                        currency = content["currency_code"]
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {
                "aggregation_type": "max",
                "field": field,
                "max": None,
                "count": 0
            }
        
        return {
            "aggregation_type": "max",
            "field": field,
            "max": max(values),
            "max_transaction": max_result["transaction"] if max_result else None,
            "count": len(values),
            "currency": currency
        }
    
    # Pour les autres types de groupage, implémenter selon les besoins
    logger.warning(f"Groupage {group_by} non implémenté pour l'agrégation max")
    return {
        "aggregation_type": "max",
        "field": field,
        "error": f"Grouping not implemented for max: {group_by}"
    }

async def calculate_ratio(
    results: List[Dict[str, Any]],
    aggregation: AggregationRequest
) -> Dict[str, Any]:
    """
    Calcule un ratio entre différentes catégories ou valeurs.
    Cette implémentation simple permet de calculer le ratio dépenses/revenus.
    
    Args:
        results: Liste des résultats
        aggregation: Configuration de l'agrégation
        
    Returns:
        Résultat de l'agrégation
    """
    field = aggregation.field
    
    # Séparer les résultats en dépenses (négatifs) et revenus (positifs)
    expenses = []
    incomes = []
    currency = None
    
    for result in results:
        content = result["content"]
        if field in content and content[field] is not None:
            try:
                value = float(content[field])
                if value < 0:
                    expenses.append(abs(value))  # Convertir en positif pour faciliter les calculs
                else:
                    incomes.append(value)
                
                # Déterminer la devise (si disponible)
                if not currency and "currency_code" in content:
                    currency = content["currency_code"]
            except (ValueError, TypeError):
                continue
    
    # Calculer les totaux
    total_expenses = sum(expenses)
    total_incomes = sum(incomes)
    
    # Calculer le ratio dépenses/revenus
    if total_incomes > 0:
        expense_to_income_ratio = (total_expenses / total_incomes) * 100
    else:
        expense_to_income_ratio = None
    
    # Calculer le ratio revenus/dépenses
    if total_expenses > 0:
        income_to_expense_ratio = (total_incomes / total_expenses) * 100
    else:
        income_to_expense_ratio = None
    
    return {
        "aggregation_type": "ratio",
        "field": field,
        "expense_to_income_ratio": expense_to_income_ratio,
        "income_to_expense_ratio": income_to_expense_ratio,
        "total_expenses": total_expenses,
        "total_incomes": total_incomes,
        "expense_count": len(expenses),
        "income_count": len(incomes),
        "currency": currency
    }