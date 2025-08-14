"""
🔍 Search Query Generation Prompts - Génération Requêtes Search Service

Ce module contient les prompts optimisés DeepSeek pour transformer les intentions
et entités utilisateur en requêtes structurées pour le Search Service.

Responsabilité :
- Génération de contrats SearchServiceQuery optimisés
- Mapping intentions → templates Elasticsearch
- Optimisation des requêtes pour performances et pertinence
- Gestion des cas complexes et multi-critères
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPTS SYSTÈME PRINCIPAUX
# =============================================================================

SEARCH_GENERATION_SYSTEM_PROMPT = """Vous êtes un expert en génération de requêtes Elasticsearch pour un système financier.

VOTRE MISSION :
Transformer les intentions utilisateur et entités extraites en requêtes Search Service optimisées pour récupérer les transactions financières les plus pertinentes.

STRUCTURE DU SEARCH SERVICE :
Le Search Service expose des endpoints standardisés avec des contrats précis :
- `/search/lexical` : Recherche textuelle BM25 + filtres
- `/search/semantic` : Recherche vectorielle (embeddings)
- `/aggregations` : Agrégations et calculs financiers

CHAMPS ELASTICSEARCH DISPONIBLES :

**Champs de RECHERCHE (BM25) :**
- `searchable_text` : Texte enrichi principal (boost 2.0)
- `primary_description` : Description transaction (boost 1.5)
- `merchant_name` : Nom marchand textuel (boost 1.8)
- `category_name` : Recherche par nom catégorie

**Champs de FILTRAGE :**
- `user_id` : OBLIGATOIRE - Isolation utilisateur
- `category_name.keyword` : Filtre exact catégorie
- `merchant_name.keyword` : Filtre exact marchand
- `transaction_type` : "debit", "credit"
- `currency_code` : "EUR", "USD", "GBP"
- `amount` : Montant avec signe (-200.00, +1500.00)
- `amount_abs` : Valeur absolue (200.00, 1500.00)
- `date` : Format YYYY-MM-DD
- `month_year` : Format YYYY-MM
- `weekday` : "Monday", "Tuesday", etc.

**Champs d'AGRÉGATION :**
- Groupement par : category_name.keyword, merchant_name.keyword, month_year
- Calculs sur : amount, amount_abs
- Métriques : sum, avg, min, max, count

FORMAT DE RÉPONSE OBLIGATOIRE (JSON) :
```json
{
  "query_type": "lexical|semantic|aggregation",
  "search_text": "texte pour recherche BM25",
    "filters": {
    "user_id": "OBLIGATOIRE",
    "date": {"gte": "2024-01-01", "lte": "2024-01-31"},
    "amount": {"gte": 0, "lte": 1000},
    "categories": ["alimentation", "transport"],
    "merchants": ["carrefour", "amazon"],
    "transaction_types": ["debit"]
  },
  "aggregations": {
    "group_by": "category_name.keyword",
    "metrics": ["sum", "count"],
    "date_histogram": "monthly"
  },
  "sorting": [{"date": "desc"}, {"amount_abs": "desc"}],
  "size": 20,
  "explanation": "Justification de la stratégie de requête"
}
```

STRATÉGIES D'OPTIMISATION :
1. **Lexical + Filtres** : Pour recherches textuelles avec critères précis
2. **Filtres Uniquement** : Pour critères très spécifiques sans recherche textuelle
3. **Agrégations** : Pour analyses, totaux, moyennes, tendances
4. **Combinaisons** : Recherche + agrégation pour analyses contextuelles

RÈGLES CRITIQUES :
- TOUJOURS inclure `user_id` dans les filtres
- Optimiser les plages de dates (éviter les ranges trop larges)
- Utiliser `.keyword` pour les filtres exacts
- Privilégier les filtres aux recherches textuelles quand possible
- Limiter `size` selon le type de requête (20 pour listes, 100+ pour analyses)"""

# =============================================================================
# TEMPLATE UTILISATEUR AVEC INTENTION
# =============================================================================

SEARCH_GENERATION_TEMPLATE = """Générez une requête Search Service optimisée pour cette demande utilisateur :

INTENTION DÉTECTÉE : {intent_type}
CONFIANCE : {confidence}
ENTITÉS EXTRAITES : {entities}

MESSAGE UTILISATEUR ORIGINAL : "{user_message}"

{context_section}

OBJECTIF : Créer la requête la plus pertinente et performante pour récupérer exactement les données demandées par l'utilisateur.

Répondez avec le JSON de requête dans le format spécifié."""

# =============================================================================
# EXEMPLES FEW-SHOT POUR OPTIMISATION
# =============================================================================

SEARCH_EXAMPLES_FEW_SHOT = """EXEMPLES DE GÉNÉRATION DE REQUÊTES :

**Exemple 1 - Transaction Query Simple :**
INTENTION: transaction_query | ENTITÉS: {"merchants": ["Carrefour"], "periods": ["mois dernier"]}
MESSAGE: "Mes achats chez Carrefour le mois dernier"

REQUÊTE GÉNÉRÉE:
```json
{
  "query_type": "lexical",
  "search_text": "Carrefour",
  "filters": {
    "user_id": "USER_ID_PLACEHOLDER",
    "date": {"gte": "2024-12-01", "lte": "2024-12-31"},
    "merchants": ["carrefour"]
  },
  "sorting": [{"date": "desc"}],
  "size": 20,
  "explanation": "Recherche textuelle + filtre marchand et période pour transactions spécifiques"
}
```

**Exemple 2 - Spending Analysis avec Agrégation :**
INTENTION: spending_analysis | ENTITÉS: {"categories": ["restaurant"], "periods": ["ces 3 derniers mois"]}
MESSAGE: "Combien j'ai dépensé en restaurant ces 3 derniers mois ?"

REQUÊTE GÉNÉRÉE:
```json
{
  "query_type": "aggregation",
  "search_text": "",
  "filters": {
    "user_id": "USER_ID_PLACEHOLDER",
    "date": {"gte": "2024-10-01", "lte": "2024-12-31"},
    "categories": ["restaurant"],
    "transaction_types": ["debit"]
  },
  "aggregations": {
    "group_by": "month_year",
    "metrics": ["sum", "count"],
    "date_histogram": "monthly"
  },
  "size": 0,
  "explanation": "Agrégation mensuelle pour analyse des dépenses restaurant sur période"
}
```

**Exemple 3 - Trend Analysis Complexe :**
INTENTION: trend_analysis | ENTITÉS: {"amounts": ["500€"], "periods": ["par mois"], "analysis_type": ["average"]}
MESSAGE: "Est-ce que je dépense plus que 500€ par mois en moyenne ?"

REQUÊTE GÉNÉRÉE:
```json
{
  "query_type": "aggregation",
  "search_text": "",
  "filters": {
    "user_id": "USER_ID_PLACEHOLDER",
    "date": {"gte": "2024-01-01", "lte": "2024-12-31"},
    "transaction_types": ["debit"]
  },
  "aggregations": {
    "group_by": "month_year",
    "metrics": ["sum", "avg"],
    "date_histogram": "monthly"
  },
  "size": 0,
  "explanation": "Analyse mensuelle pour calculer moyenne et comparer au seuil 500€"
}
```

**Exemple 4 - Recherche Textuelle Libre :**
INTENTION: transaction_query | ENTITÉS: {"periods": ["cette semaine"]}
MESSAGE: "Mes achats pharmacie cette semaine"

REQUÊTE GÉNÉRÉE:
```json
{
  "query_type": "lexical",
  "search_text": "pharmacie",
  "filters": {
    "user_id": "USER_ID_PLACEHOLDER",
    "date": {"gte": "2024-12-23", "lte": "2024-12-29"}
  },
  "sorting": [{"date": "desc"}, {"amount_abs": "desc"}],
  "size": 20,
  "explanation": "Recherche textuelle 'pharmacie' sur période récente avec tri chronologique"
}
```"""

# =============================================================================
# FONCTIONS DE FORMATAGE
# =============================================================================

def format_search_prompt(
    intent_result: Dict[str, Any], 
    user_message: str,
    user_id: Optional[str] = None,
    context: str = ""
) -> str:
    """
    Formate le prompt complet pour la génération de requête Search Service.
    
    Args:
        intent_result: Résultat de détection d'intention avec entities
        user_message: Message utilisateur original
        user_id: ID utilisateur pour les filtres (optionnel pour le prompt)
        context: Contexte conversationnel optionnel
        
    Returns:
        Prompt formaté prêt pour DeepSeek
        
    Raises:
        ValueError: Si intent_result invalide
        
    Example:
        >>> intent = {"intent": "transaction_query", "confidence": 0.9, "entities": {...}}
        >>> prompt = format_search_prompt(intent, "Mes achats Amazon")
    """
    if not isinstance(intent_result, dict):
        raise ValueError("intent_result doit être un dictionnaire")
    
    required_keys = ["intent", "confidence", "entities"]
    for key in required_keys:
        if key not in intent_result:
            raise ValueError(f"Clé manquante dans intent_result: {key}")
    
    if not user_message or not user_message.strip():
        raise ValueError("user_message ne peut pas être vide")
    
    # Formatage des entités pour affichage
    entities_formatted = json.dumps(intent_result["entities"], ensure_ascii=False, indent=2)
    
    # Construction de la section contexte
    context_section = ""
    if context and context.strip():
        context_section = f"\nCONTEXTE CONVERSATIONNEL:\n{context.strip()}\n"
    
    # Formatage du prompt utilisateur
    user_prompt = SEARCH_GENERATION_TEMPLATE.format(
        intent_type=intent_result["intent"],
        confidence=intent_result["confidence"],
        entities=entities_formatted,
        user_message=user_message.strip(),
        context_section=context_section
    )
    
    return user_prompt

def build_date_range_from_period(period_text: str) -> Dict[str, str]:
    """
    Convertit une expression de période en plage de dates pour Elasticsearch.
    
    Args:
        period_text: Expression temporelle ("mois dernier", "cette semaine", etc.)
        
    Returns:
        Dict avec gte/lte au format YYYY-MM-DD
        
    Example:
        >>> date_range = build_date_range_from_period("mois dernier")
        >>> # {"gte": "2024-11-01", "lte": "2024-11-30"}
    """
    today = date.today()
    
    # Normalisation du texte
    period_lower = period_text.lower().strip()
    
    # Mapping périodes communes
    if "mois dernier" in period_lower or "le mois dernier" in period_lower:
        if today.month == 1:
            start_date = date(today.year - 1, 12, 1)
            end_date = date(today.year - 1, 12, 31)
        else:
            start_date = date(today.year, today.month - 1, 1)
            # Dernier jour du mois précédent
            if today.month - 1 in [1, 3, 5, 7, 8, 10, 12]:
                end_date = date(today.year, today.month - 1, 31)
            elif today.month - 1 in [4, 6, 9, 11]:
                end_date = date(today.year, today.month - 1, 30)
            else:  # Février
                end_date = date(today.year, today.month - 1, 28)
    
    elif "cette semaine" in period_lower:
        # Début de semaine (lundi)
        from datetime import timedelta
        days_since_monday = today.weekday()
        start_date = today - timedelta(days=days_since_monday)
        end_date = start_date + timedelta(days=6)
    
    elif "semaine dernière" in period_lower or "la semaine dernière" in period_lower:
        from datetime import timedelta
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        start_date = this_monday - timedelta(days=7)
        end_date = start_date + timedelta(days=6)
    
    elif "ce mois" in period_lower or "ce mois-ci" in period_lower:
        start_date = date(today.year, today.month, 1)
        end_date = today
    
    elif "3 derniers mois" in period_lower or "trois derniers mois" in period_lower:
        start_date = date(today.year, max(1, today.month - 2), 1)
        end_date = today
    
    elif "6 derniers mois" in period_lower or "six derniers mois" in period_lower:
        start_date = date(today.year, max(1, today.month - 5), 1)
        end_date = today
    
    elif "cette année" in period_lower or "2024" in period_lower:
        start_date = date(today.year, 1, 1)
        end_date = today
    
    elif "hier" in period_lower:
        from datetime import timedelta
        start_date = end_date = today - timedelta(days=1)
    
    elif "aujourd'hui" in period_lower or "aujourd hui" in period_lower:
        start_date = end_date = today
    
    else:
        # Fallback : dernier mois par défaut
        logger.warning(f"Période non reconnue: {period_text}, utilisation du mois dernier")
        if today.month == 1:
            start_date = date(today.year - 1, 12, 1)
            end_date = date(today.year - 1, 12, 31)
        else:
            start_date = date(today.year, today.month - 1, 1)
            end_date = date(today.year, today.month - 1, 28)  # Sécurisé
    
    return {
        "gte": start_date.strftime("%Y-%m-%d"),
        "lte": end_date.strftime("%Y-%m-%d")
    }

def normalize_merchant_name(merchant: str) -> str:
    """
    Normalise un nom de marchand pour la recherche.
    
    Args:
        merchant: Nom de marchand brut
        
    Returns:
        Nom normalisé pour recherche
        
    Example:
        >>> normalized = normalize_merchant_name("CARREFOUR MARKET")
        >>> # "carrefour"
    """
    if not merchant:
        return ""
    
    # Normalisation basique
    normalized = merchant.lower().strip()
    
    # Suppression des suffixes courants
    suffixes = [
        " market", " express", " city", " contact", " premium",
        " store", " shop", " magasin", " supermarché", " sa",
        " sas", " sarl", " eurl", " sasu"
    ]
    
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    # Suppression des préfixes courants
    prefixes = ["magasin ", "supermarché ", "station "]
    
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    
    return normalized

def extract_amount_range(amount_entities: List[str]) -> Dict[str, float]:
    """
    Extrait une plage de montants à partir des entités détectées.
    
    Args:
        amount_entities: Liste d'expressions de montants ["50€", "moins de 100€"]
        
    Returns:
        Dict avec gte/lte pour filtrage Elasticsearch
        
    Example:
        >>> range_filter = extract_amount_range(["plus de 50€", "moins de 200€"])
        >>> # {"gte": 50.0, "lte": 200.0}
    """
    import re
    
    amount_range = {}
    
    for amount_expr in amount_entities:
        if not amount_expr:
            continue
            
        expr_lower = amount_expr.lower().strip()
        
        # Extraction des nombres avec regex
        numbers = re.findall(r'(\d+(?:[,\.]\d+)?)', expr_lower)
        if not numbers:
            continue
            
        # Conversion du premier nombre trouvé
        try:
            number_str = numbers[0].replace(',', '.')
            amount = float(number_str)
        except ValueError:
            continue
        
        # Détermination du type de contrainte
        if any(word in expr_lower for word in ["plus de", "supérieur", "minimum", "au moins"]):
            amount_range["gte"] = amount
        elif any(word in expr_lower for word in ["moins de", "inférieur", "maximum", "au plus"]):
            amount_range["lte"] = amount
        elif any(word in expr_lower for word in ["environ", "autour de", "approximativement"]):
            # Plage approximative ±20%
            margin = amount * 0.2
            amount_range["gte"] = amount - margin
            amount_range["lte"] = amount + margin
        else:
            # Montant exact -> plage étroite ±5%
            margin = amount * 0.05
            amount_range["gte"] = amount - margin
            amount_range["lte"] = amount + margin
    
    return amount_range

def parse_search_response(response: str, user_id: str) -> Dict[str, Any]:
    """
    Parse la réponse JSON de DeepSeek et injecte l'user_id obligatoire.
    
    Args:
        response: Réponse brute JSON de DeepSeek
        user_id: ID utilisateur à injecter dans les filtres
        
    Returns:
        Dict avec requête Search Service validée
        
    Raises:
        ValueError: Si le JSON est invalide ou incomplet
        
    Example:
        >>> query = parse_search_response(json_response, "user123")
        >>> # query["filters"]["user_id"] == "user123"
    """
    if not response or not response.strip():
        raise ValueError("Réponse vide de DeepSeek")
    
    if not user_id:
        raise ValueError("user_id est obligatoire")
    
    try:
        # Extraction du JSON depuis la réponse (peut contenir du texte autour)
        response_clean = response.strip()
        
        # Recherche du JSON dans la réponse
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("Aucun JSON trouvé dans la réponse")
        
        json_str = response_clean[json_start:json_end]
        query_dict = json.loads(json_str)
        
        # Validation des champs obligatoires
        required_fields = ["query_type", "filters"]
        for field in required_fields:
            if field not in query_dict:
                raise ValueError(f"Champ obligatoire manquant: {field}")
        
        # Injection de l'user_id obligatoire
        if "filters" not in query_dict:
            query_dict["filters"] = {}
        
        query_dict["filters"]["user_id"] = user_id
        
        # Validation du query_type
        valid_types = ["lexical", "semantic", "aggregation"]
        if query_dict["query_type"] not in valid_types:
            logger.warning(f"Query type non standard: {query_dict['query_type']}")
            query_dict["query_type"] = "lexical"  # Fallback
        
        # Valeurs par défaut
        if "size" not in query_dict:
            query_dict["size"] = 20 if query_dict["query_type"] != "aggregation" else 0
        
        if "sorting" not in query_dict and query_dict["query_type"] != "aggregation":
            query_dict["sorting"] = [{"date": "desc"}]
        
        return query_dict
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Erreur parsing réponse search: {e}")
        logger.error(f"Réponse brute: {response}")
        
        # Fallback gracieux - requête basique
        return {
            "query_type": "lexical",
            "search_text": "",
            "filters": {
                "user_id": user_id,
                "date": build_date_range_from_period("ce mois")
            },
            "sorting": [{"date": "desc"}],
            "size": 20,
            "explanation": f"Requête fallback suite à erreur: {str(e)}"
        }

def optimize_query_for_performance(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimise une requête pour améliorer les performances Elasticsearch.
    
    Args:
        query: Requête Search Service à optimiser
        
    Returns:
        Requête optimisée
        
    Example:
        >>> optimized = optimize_query_for_performance(base_query)
    """
    optimized = query.copy()
    
    # Optimisation 1: Limiter les plages de dates trop larges
    if "filters" in optimized and "date" in optimized["filters"]:
        date_filter = optimized["filters"]["date"]
        if "gte" in date_filter and "lte" in date_filter:
            try:
                start_date = datetime.strptime(date_filter["gte"], "%Y-%m-%d")
                end_date = datetime.strptime(date_filter["lte"], "%Y-%m-%d")
                days_diff = (end_date - start_date).days
                
                # Si plus de 2 ans, limiter à 1 an
                if days_diff > 730:
                    logger.info("Limitation de la plage de dates pour performance")
                    new_start = end_date - datetime.timedelta(days=365)
                    optimized["filters"]["date"]["gte"] = new_start.strftime("%Y-%m-%d")
            except ValueError:
                pass  # Ignore les erreurs de parsing de dates
    
    # Optimisation 2: Ajuster la taille selon le type de requête
    if optimized.get("query_type") == "aggregation":
        optimized["size"] = 0  # Pas besoin de documents pour les agrégations
    elif optimized.get("query_type") == "lexical":
        # Limiter la taille pour les recherches textuelles
        if optimized.get("size", 20) > 100:
            optimized["size"] = 100
    
    # Optimisation 3: Privilégier les filtres aux recherches textuelles
    if (optimized.get("search_text", "").strip() == "" and 
        optimized.get("filters", {}) and
        len(optimized["filters"]) > 1):  # Plus que juste user_id
        optimized["query_type"] = "lexical"  # Utilisera principalement les filtres
    
    return optimized

# =============================================================================
# CONSTANTES UTILES
# =============================================================================

ELASTICSEARCH_FIELD_MAPPING = {
    # Champs textuels
    "searchable_text": {"type": "search", "boost": 2.0},
    "primary_description": {"type": "search", "boost": 1.5},
    "merchant_name": {"type": "search", "boost": 1.8},
    "category_name": {"type": "search", "boost": 1.0},
    
    # Champs de filtrage exact
    "category_name.keyword": {"type": "filter"},
    "merchant_name.keyword": {"type": "filter"},
    "transaction_type": {"type": "filter"},
    "currency_code": {"type": "filter"},
    
    # Champs numériques/temporels
    "amount": {"type": "range"},
    "amount_abs": {"type": "range"},
    "date": {"type": "range"},
    "month_year": {"type": "filter"},
    "weekday": {"type": "filter"}
}

QUERY_TYPE_STRATEGIES = {
    "transaction_query": "lexical",
    "spending_analysis": "aggregation", 
    "budget_inquiry": "aggregation",
    "category_analysis": "aggregation",
    "merchant_inquiry": "lexical",
    "balance_inquiry": "aggregation",
    "trend_analysis": "aggregation",
    "comparison_query": "aggregation",
    "goal_tracking": "aggregation",
    "alert_management": "lexical",
    "conversational": "lexical",
    "other": "lexical"
}

# Export des éléments principaux
__all__ = [
    "SEARCH_GENERATION_SYSTEM_PROMPT",
    "SEARCH_GENERATION_TEMPLATE",
    "SEARCH_EXAMPLES_FEW_SHOT",
    "format_search_prompt",
    "build_date_range_from_period",
    "normalize_merchant_name",
    "extract_amount_range", 
    "parse_search_response",
    "optimize_query_for_performance",
    "ELASTICSEARCH_FIELD_MAPPING",
    "QUERY_TYPE_STRATEGIES"
]