"""
Utilitaires pour la fusion des résultats de recherche.

Ce module contient des fonctions utilitaires pour créer,
manipuler et traiter les résultats fusionnés.
"""
from typing import Dict, Any, List, Optional

from search_service.models.responses import SearchResultItem


def create_fused_item(
    base_item: SearchResultItem,
    lexical_item: Optional[SearchResultItem],
    semantic_item: Optional[SearchResultItem],
    combined_score: float,
    fusion_method: str,
    fusion_metadata: Dict[str, Any]
) -> SearchResultItem:
    """
    Crée un SearchResultItem fusionné.
    
    Args:
        base_item: Item de base (le plus complet)
        lexical_item: Item du moteur lexical (peut être None)
        semantic_item: Item du moteur sémantique (peut être None)
        combined_score: Score combiné calculé
        fusion_method: Méthode de fusion utilisée
        fusion_metadata: Métadonnées de fusion
        
    Returns:
        SearchResultItem fusionné
    """
    # Fusionner les métadonnées
    metadata = base_item.metadata.copy() if base_item.metadata else {}
    metadata.update({
        "fusion_method": fusion_method,
        "found_in_lexical": lexical_item is not None,
        "found_in_semantic": semantic_item is not None,
        **fusion_metadata
    })
    
    # Privilégier les highlights du lexical s'ils existent
    highlights = None
    if lexical_item and hasattr(lexical_item, 'highlights') and lexical_item.highlights:
        highlights = lexical_item.highlights
    elif semantic_item and hasattr(semantic_item, 'highlights') and semantic_item.highlights:
        highlights = semantic_item.highlights
    
    return SearchResultItem(
        transaction_id=base_item.transaction_id,
        user_id=base_item.user_id,
        account_id=base_item.account_id,
        score=combined_score,
        lexical_score=lexical_item.score if lexical_item else None,
        semantic_score=semantic_item.score if semantic_item else None,
        combined_score=combined_score,
        primary_description=base_item.primary_description,
        searchable_text=base_item.searchable_text,
        merchant_name=base_item.merchant_name,
        amount=base_item.amount,
        currency_code=base_item.currency_code,
        transaction_type=base_item.transaction_type,
        transaction_date=base_item.transaction_date,
        created_at=base_item.created_at,
        category_id=base_item.category_id,
        operation_type=base_item.operation_type,
        highlights=highlights,
        metadata=metadata,
        explanation=getattr(base_item, 'explanation', None)
    )


def create_transaction_signature(result: SearchResultItem) -> str:
    """
    Crée une signature pour identifier les doublons.
    
    Args:
        result: SearchResultItem à signer
        
    Returns:
        Signature unique pour la transaction
    """
    # Utiliser plusieurs champs pour créer une signature
    components = [
        str(result.amount or 0),
        result.transaction_date or "",
        (result.merchant_name or "").lower()[:20],
        (result.primary_description or "").lower()[:30]
    ]
    
    return "|".join(components)


def calculate_signature_similarity(sig1: str, sig2: str) -> float:
    """
    Calcule la similarité entre deux signatures.
    
    Args:
        sig1: Première signature
        sig2: Deuxième signature
        
    Returns:
        Score de similarité entre 0 et 1
    """
    if sig1 == sig2:
        return 1.0
    
    # Similarité basée sur les composants
    parts1 = sig1.split("|")
    parts2 = sig2.split("|")
    
    if len(parts1) != len(parts2):
        return 0.0
    
    similarities = []
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            similarities.append(1.0)
        elif p1 and p2:
            # Similarité de chaînes simple
            longer = max(len(p1), len(p2))
            common = sum(c1 == c2 for c1, c2 in zip(p1, p2))
            similarities.append(common / longer)
        else:
            similarities.append(0.0)
    
    return sum(similarities) / len(similarities)


def deduplicate_results(
    results: List[SearchResultItem],
    similarity_threshold: float = 0.95,
    debug: bool = False
) -> List[SearchResultItem]:
    """
    Déduplique les résultats basés sur la similarité.
    
    Args:
        results: Liste des résultats à dédupliquer
        similarity_threshold: Seuil de similarité pour considérer un doublon
        debug: Si True, marque les doublons dans les métadonnées
        
    Returns:
        Liste des résultats dédupliqués
    """
    if len(results) <= 1:
        return results
    
    deduplicated = []
    seen_signatures = set()
    
    for result in results:
        # Créer une signature pour la transaction
        signature = create_transaction_signature(result)
        
        # Vérifier la similarité avec les transactions déjà vues
        is_duplicate = False
        for seen_sig in seen_signatures:
            if calculate_signature_similarity(signature, seen_sig) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(result)
            seen_signatures.add(signature)
        elif debug:
            # Marquer comme dupliqué dans les métadonnées pour debug
            if result.metadata:
                result.metadata["duplicate_removed"] = True
    
    return deduplicated


def diversify_results(
    results: List[SearchResultItem],
    max_same_merchant: int = 3,
    debug: bool = False
) -> List[SearchResultItem]:
    """
    Diversifie les résultats pour éviter la redondance.
    
    Args:
        results: Liste des résultats à diversifier
        max_same_merchant: Nombre maximum de résultats par marchand
        debug: Si True, marque les résultats filtrés
        
    Returns:
        Liste des résultats diversifiés
    """
    if len(results) <= max_same_merchant:
        return results
    
    diversified = []
    merchant_counts = {}
    
    # Trier par score décroissant d'abord
    sorted_results = sorted(results, key=lambda x: x.score or 0, reverse=True)
    
    for result in sorted_results:
        merchant = result.merchant_name or "unknown"
        current_count = merchant_counts.get(merchant, 0)
        
        # Ajouter si on n'a pas atteint la limite pour ce marchand
        if current_count < max_same_merchant:
            diversified.append(result)
            merchant_counts[merchant] = current_count + 1
        elif debug:
            # Marquer comme filtré pour diversité
            if result.metadata:
                result.metadata["diversity_filtered"] = True
    
    return diversified


def apply_quality_boost(
    results: List[SearchResultItem],
    key_terms: List[str],
    quality_boost_factor: float = 0.2,
    debug: bool = False
) -> List[SearchResultItem]:
    """
    Applique un boost de qualité aux résultats pertinents.
    
    Args:
        results: Liste des résultats à booster
        key_terms: Termes clés de la requête
        quality_boost_factor: Facteur de boost
        debug: Si True, ajoute les infos de debug
        
    Returns:
        Liste des résultats avec boost appliqué
    """
    for result in results:
        boost_factor = 1.0
        boost_reasons = []
        
        # Boost pour correspondance avec les termes de la requête
        if key_terms:
            text_content = " ".join(filter(None, [
                result.primary_description,
                result.merchant_name,
                result.searchable_text
            ])).lower()
            
            matching_terms = sum(
                1 for term in key_terms
                if term.lower() in text_content
            )
            
            if matching_terms > 0:
                term_boost = 1 + (matching_terms / len(key_terms)) * quality_boost_factor
                boost_factor *= term_boost
                boost_reasons.append(f"term_match_{matching_terms}")
        
        # Boost pour transactions récentes
        if result.transaction_date:
            # Simple boost pour les transactions récentes (implémentation basique)
            boost_factor *= 1.05
            boost_reasons.append("recent_transaction")
        
        # Boost pour montants ronds (souvent plus mémorables)
        if result.amount and result.amount == int(result.amount):
            boost_factor *= 1.02
            boost_reasons.append("round_amount")
        
        # Appliquer le boost
        if boost_factor > 1.0:
            original_score = result.score
            result.score *= boost_factor
            result.combined_score *= boost_factor
            
            if debug and result.metadata:
                result.metadata.update({
                    "quality_boost_applied": boost_factor,
                    "boost_reasons": boost_reasons,
                    "original_score": original_score
                })
    
    return results