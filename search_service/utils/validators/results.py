"""
Validateur de résultats pour le Search Service.

Ce module fournit la validation complète des résultats de recherche,
incluant la structure, intégrité des données et conformité du format.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime

from .base import (
    BaseValidator, ValidationResult, ValidationLevel,
    ResultValidationError, validate_user_id, validate_amount, validate_date
)
from .config import (
    ALLOWED_SEARCH_FIELDS, VALIDATION_LIMITS, DATA_TYPE_LIMITS,
    VALIDATION_CONFIG, ERROR_MESSAGES
)

logger = logging.getLogger(__name__)

class ResultValidator(BaseValidator):
    """
    Validateur spécialisé pour les résultats de recherche.
    
    Valide la structure et l'intégrité des résultats retournés
    par Elasticsearch avant de les transmettre au client.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.config = VALIDATION_CONFIG[validation_level.value]
    
    def validate(self, results: Dict[str, Any], **kwargs) -> ValidationResult:
        """
        Valide des résultats de recherche Elasticsearch.
        
        Args:
            results: Résultats Elasticsearch à valider
            **kwargs: Options supplémentaires
            
        Returns:
            ValidationResult avec détails de validation
        """
        start_time = time.time()
        result = self._create_result()
        
        try:
            # Validation de la structure de base
            if not self._validate_basic_structure(results, result):
                return result
            
            # Validation des métadonnées
            self._validate_metadata(results, result)
            
            # Validation des hits
            if "hits" in results:
                self._validate_hits_structure(results["hits"], result)
                
                # Validation de chaque hit individuel
                if "hits" in results["hits"] and isinstance(results["hits"]["hits"], list):
                    sanitized_hits = []
                    for i, hit in enumerate(results["hits"]["hits"]):
                        hit_result = self._validate_single_hit(hit, i)
                        
                        result.errors.extend(hit_result.errors)
                        result.warnings.extend(hit_result.warnings)
                        result.security_flags.extend(hit_result.security_flags)
                        
                        if hit_result.errors:
                            result.is_valid = False
                        else:
                            sanitized_hits.append(hit_result.sanitized_data or hit)
                    
                    # Mise à jour des hits sanitisés
                    if self.config.get("sanitize_input", False):
                        if "sanitized_data" not in result.__dict__:
                            result.sanitized_data = results.copy()
                        result.sanitized_data["hits"]["hits"] = sanitized_hits
            
            # Validation des agrégations
            if "aggregations" in results:
                self._validate_aggregations(results["aggregations"], result)
            
            # Validation de la cohérence globale
            self._validate_result_consistency(results, result)
            
            # Sanitisation si demandée
            if self.config.get("sanitize_input", False) and not hasattr(result, 'sanitized_data'):
                result.sanitized_data = self._sanitize_results(results)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des résultats: {e}")
            result.add_error(f"Erreur interne de validation: {str(e)}")
        
        result.validation_time_ms = self._measure_validation_time(start_time)
        return result
    
    def validate_hit(self, hit: Dict[str, Any], index: int = 0) -> ValidationResult:
        """
        Valide un hit individuel.
        
        Args:
            hit: Hit à valider
            index: Index du hit dans les résultats
            
        Returns:
            ValidationResult
        """
        return self._validate_single_hit(hit, index)
    
    def validate_aggregation(self, aggregation: Dict[str, Any], agg_name: str) -> ValidationResult:
        """
        Valide une agrégation spécifique.
        
        Args:
            aggregation: Agrégation à valider
            agg_name: Nom de l'agrégation
            
        Returns:
            ValidationResult
        """
        result = self._create_result()
        
        # Validation de base
        if not isinstance(aggregation, dict):
            result.add_error(f"L'agrégation {agg_name} doit être un dictionnaire")
            return result
        
        # Validation des types d'agrégation courants
        if "buckets" in aggregation:
            self._validate_bucket_aggregation(aggregation, agg_name, result)
        elif "value" in aggregation:
            self._validate_metric_aggregation(aggregation, agg_name, result)
        else:
            result.add_warning(f"Structure d'agrégation non reconnue pour {agg_name}")
        
        return result
    
    def _validate_basic_structure(self, results: Dict[str, Any], result: ValidationResult) -> bool:
        """Valide la structure de base des résultats."""
        if not isinstance(results, dict):
            result.add_error("Les résultats doivent être un dictionnaire")
            return False
        
        # Vérification des champs requis
        required_fields = {"took", "hits"}
        missing_fields = required_fields - set(results.keys())
        
        if missing_fields:
            result.add_error(f"Champs requis manquants: {missing_fields}")
            return False
        
        return True
    
    def _validate_metadata(self, results: Dict[str, Any], result: ValidationResult):
        """Valide les métadonnées des résultats."""
        # Validation du temps de réponse
        if "took" in results:
            took = results["took"]
            if not isinstance(took, (int, float)) or took < 0:
                result.add_error("Le champ 'took' doit être un nombre positif")
            elif took > 30000:  # 30 secondes
                result.add_warning(f"Temps de réponse très élevé: {took}ms")
        
        # Validation du timeout
        if "timed_out" in results:
            if not isinstance(results["timed_out"], bool):
                result.add_error("Le champ 'timed_out' doit être un booléen")
            elif results["timed_out"]:
                result.add_warning("La requête a expiré (timeout)")
        
        # Validation des shards
        if "_shards" in results:
            shards = results["_shards"]
            if isinstance(shards, dict):
                if shards.get("failed", 0) > 0:
                    result.add_warning(f"Échec de {shards['failed']} shard(s)")
                if shards.get("skipped", 0) > 0:
                    result.add_warning(f"{shards['skipped']} shard(s) ignoré(s)")
    
    def _validate_hits_structure(self, hits: Dict[str, Any], result: ValidationResult):
        """Valide la structure des hits."""
        if not isinstance(hits, dict):
            result.add_error("La section 'hits' doit être un dictionnaire")
            return
        
        # Validation du total
        if "total" in hits:
            total = hits["total"]
            if isinstance(total, dict):
                # Format Elasticsearch 7+
                if "value" not in total:
                    result.add_error("Format total invalide: 'value' manquant")
                elif not isinstance(total["value"], int) or total["value"] < 0:
                    result.add_error("total.value doit être un entier positif")
            elif isinstance(total, int):
                # Format Elasticsearch 6-
                if total < 0:
                    result.add_error("total doit être positif")
            else:
                result.add_error("Format total invalide")
        
        # Validation du score maximum
        if "max_score" in hits:
            max_score = hits["max_score"]
            if max_score is not None and (not isinstance(max_score, (int, float)) or max_score < 0):
                result.add_error("max_score doit être un nombre positif ou null")
        
        # Validation de la liste des hits
        if "hits" in hits:
            if not isinstance(hits["hits"], list):
                result.add_error("hits.hits doit être une liste")
            elif len(hits["hits"]) > VALIDATION_LIMITS.get("max_results_limit", 1000):
                result.add_error(f"Trop de résultats (max: {VALIDATION_LIMITS.get('max_results_limit', 1000)})")
    
    def _validate_single_hit(self, hit: Dict[str, Any], index: int) -> ValidationResult:
        """Valide un hit individuel."""
        result = self._create_result()
        
        # Validation de base
        if not isinstance(hit, dict):
            result.add_error(f"Le hit {index} doit être un dictionnaire")
            return result
        
        # Validation des champs requis
        required_fields = {"_id", "_source"}
        missing_fields = required_fields - set(hit.keys())
        
        if missing_fields:
            result.add_error(f"Hit {index}: champs requis manquants: {missing_fields}")
        
        # Validation de l'ID
        if "_id" in hit:
            if not isinstance(hit["_id"], str) or not hit["_id"].strip():
                result.add_error(f"Hit {index}: _id invalide")
        
        # Validation du score
        if "_score" in hit:
            score = hit["_score"]
            if score is not None and (not isinstance(score, (int, float)) or score < 0):
                result.add_error(f"Hit {index}: _score invalide")
        
        # Validation de la source
        if "_source" in hit:
            source_result = self._validate_hit_source(hit["_source"], index)
            result.errors.extend(source_result.errors)
            result.warnings.extend(source_result.warnings)
            result.security_flags.extend(source_result.security_flags)
        
        # Validation des highlights
        if "highlight" in hit:
            highlight_result = self._validate_hit_highlights(hit["highlight"], index)
            result.errors.extend(highlight_result.errors)
            result.warnings.extend(highlight_result.warnings)
        
        # Validation des champs calculés
        if "fields" in hit:
            if not isinstance(hit["fields"], dict):
                result.add_error(f"Hit {index}: 'fields' doit être un dictionnaire")
        
        # Sanitisation
        if self.config.get("sanitize_input", False):
            result.sanitized_data = self._sanitize_hit(hit)
        
        return result
    
    def _validate_hit_source(self, source: Dict[str, Any], hit_index: int) -> ValidationResult:
        """Valide la source d'un hit."""
        result = self._create_result()
        
        if not isinstance(source, dict):
            result.add_error(f"Hit {hit_index}: _source doit être un dictionnaire")
            return result
        
        # Validation des champs financiers spécifiques
        if "user_id" in source:
            if not validate_user_id(source["user_id"]):
                result.add_error(f"Hit {hit_index}: user_id invalide")
        
        if "amount" in source:
            if not validate_amount(source["amount"]):
                result.add_error(f"Hit {hit_index}: montant invalide")
        
        if "transaction_date" in source:
            if not validate_date(source["transaction_date"]):
                result.add_error(f"Hit {hit_index}: date de transaction invalide")
        
        # Validation des champs texte
        text_fields = ["merchant_name", "description", "clean_description", "searchable_text"]
        for field in text_fields:
            if field in source:
                value = source[field]
                if isinstance(value, str):
                    if len(value) > DATA_TYPE_LIMITS.get("description", {}).get("max_length", 5000):
                        result.add_error(f"Hit {hit_index}: {field} trop long")
                    
                    # Vérification de sécurité basique
                    if self.config.get("check_security_patterns", False):
                        if not self._check_security_patterns(value, result):
                            result.add_security_flag(f"Hit {hit_index}: contenu suspect in {field}")
        
        return result
    
    def _validate_hit_highlights(self, highlights: Dict[str, Any], hit_index: int) -> ValidationResult:
        """Valide les highlights d'un hit."""
        result = self._create_result()
        
        if not isinstance(highlights, dict):
            result.add_error(f"Hit {hit_index}: highlights doit être un dictionnaire")
            return result
        
        for field, fragments in highlights.items():
            if not isinstance(fragments, list):
                result.add_error(f"Hit {hit_index}: highlight {field} doit être une liste")
                continue
            
            for i, fragment in enumerate(fragments):
                if not isinstance(fragment, str):
                    result.add_error(f"Hit {hit_index}: fragment highlight {field}[{i}] invalide")
                elif len(fragment) > 1000:
                    result.add_warning(f"Hit {hit_index}: fragment highlight {field}[{i}] très long")
        
        return result
    
    def _validate_aggregations(self, aggregations: Dict[str, Any], result: ValidationResult):
        """Valide les agrégations."""
        if not isinstance(aggregations, dict):
            result.add_error("Les agrégations doivent être un dictionnaire")
            return
        
        if len(aggregations) > 20:
            result.add_warning("Beaucoup d'agrégations retournées")
        
        for agg_name, agg_data in aggregations.items():
            agg_result = self.validate_aggregation(agg_data, agg_name)
            result.errors.extend(agg_result.errors)
            result.warnings.extend(agg_result.warnings)
    
    def _validate_bucket_aggregation(self, aggregation: Dict[str, Any], agg_name: str, result: ValidationResult):
        """Valide une agrégation de type bucket."""
        buckets = aggregation.get("buckets", [])
        
        if not isinstance(buckets, list):
            result.add_error(f"Agrégation {agg_name}: buckets doit être une liste")
            return
        
        if len(buckets) > VALIDATION_LIMITS.get("max_aggregation_buckets", 10000):
            result.add_warning(f"Agrégation {agg_name}: beaucoup de buckets ({len(buckets)})")
        
        for i, bucket in enumerate(buckets):
            if not isinstance(bucket, dict):
                result.add_error(f"Agrégation {agg_name}: bucket {i} invalide")
                continue
            
            # Validation doc_count
            if "doc_count" in bucket:
                doc_count = bucket["doc_count"]
                if not isinstance(doc_count, int) or doc_count < 0:
                    result.add_error(f"Agrégation {agg_name}: doc_count invalide dans bucket {i}")
            
            # Validation de la clé
            if "key" not in bucket:
                result.add_warning(f"Agrégation {agg_name}: clé manquante dans bucket {i}")
    
    def _validate_metric_aggregation(self, aggregation: Dict[str, Any], agg_name: str, result: ValidationResult):
        """Valide une agrégation de type métrique."""
        value = aggregation.get("value")
        
        if value is not None and not isinstance(value, (int, float)):
            result.add_error(f"Agrégation {agg_name}: valeur invalide")
        
        # Validation des métriques statistiques
        for metric in ["count", "min", "max", "avg", "sum"]:
            if metric in aggregation:
                metric_value = aggregation[metric]
                if metric_value is not None and not isinstance(metric_value, (int, float)):
                    result.add_error(f"Agrégation {agg_name}: {metric} invalide")
    
    def _validate_result_consistency(self, results: Dict[str, Any], result: ValidationResult):
        """Valide la cohérence globale des résultats."""
        # Cohérence entre total et nombre de hits
        if "hits" in results and isinstance(results["hits"], dict):
            hits_data = results["hits"]
            
            if "total" in hits_data and "hits" in hits_data:
                total_value = hits_data["total"]
                if isinstance(total_value, dict):
                    total_value = total_value.get("value", 0)
                
                actual_hits = len(hits_data["hits"])
                
                # Si on a demandé moins de résultats que le total, c'est normal
                # Mais si on a plus de hits que le total déclaré, c'est incohérent
                if actual_hits > total_value:
                    result.add_error("Incohérence: plus de hits que le total déclaré")
        
        # Validation des scores (doivent être décroissants si triés par pertinence)
        if "hits" in results and isinstance(results["hits"], dict):
            hits_list = results["hits"].get("hits", [])
            if len(hits_list) > 1:
                scores = [hit.get("_score") for hit in hits_list if hit.get("_score") is not None]
                if len(scores) > 1:
                    # Vérifier si les scores sont en ordre décroissant (tri par pertinence)
                    is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                    if not is_descending:
                        result.add_warning("Les scores ne sont pas en ordre décroissant")
    
    def _sanitize_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise un hit."""
        sanitized = hit.copy()
        
        # Sanitisation de la source
        if "_source" in sanitized and isinstance(sanitized["_source"], dict):
            sanitized["_source"] = self._sanitize_source(sanitized["_source"])
        
        # Sanitisation des highlights
        if "highlight" in sanitized and isinstance(sanitized["highlight"], dict):
            sanitized["highlight"] = self._sanitize_highlights(sanitized["highlight"])
        
        return sanitized
    
    def _sanitize_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise la source d'un hit."""
        sanitized = {}
        
        for field, value in source.items():
            if isinstance(value, str):
                sanitized[field] = self._sanitize_string(value)
            elif isinstance(value, (list, dict)):
                # Pas de sanitisation pour les structures complexes par défaut
                sanitized[field] = value
            else:
                sanitized[field] = value
        
        return sanitized
    
    def _sanitize_highlights(self, highlights: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise les highlights."""
        sanitized = {}
        
        for field, fragments in highlights.items():
            if isinstance(fragments, list):
                sanitized[field] = [
                    self._sanitize_string(fragment) if isinstance(fragment, str) else fragment
                    for fragment in fragments
                ]
            else:
                sanitized[field] = fragments
        
        return sanitized
    
    def _sanitize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitise l'ensemble des résultats."""
        sanitized = results.copy()
        
        # Sanitisation des hits
        if "hits" in sanitized and isinstance(sanitized["hits"], dict):
            hits_data = sanitized["hits"]
            if "hits" in hits_data and isinstance(hits_data["hits"], list):
                hits_data["hits"] = [
                    self._sanitize_hit(hit) for hit in hits_data["hits"]
                ]
        
        return sanitized

# ==================== VALIDATIONS SPÉCIALISÉES ====================

class FinancialResultValidator(ResultValidator):
    """
    Validateur spécialisé pour les résultats financiers.
    
    Étend ResultValidator avec des validations spécifiques
    aux données de transactions financières.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        
        # Champs requis pour les transactions financières
        self.required_financial_fields = {
            "user_id", "amount", "transaction_date", "merchant_name"
        }
        
        # Champs optionnels mais recommandés
        self.recommended_fields = {
            "category_id", "description", "currency_code"
        }
    
    def _validate_hit_source(self, source: Dict[str, Any], hit_index: int) -> ValidationResult:
        """Validation spécialisée pour les sources financières."""
        result = super()._validate_hit_source(source, hit_index)
        
        # Vérification des champs financiers requis
        missing_required = self.required_financial_fields - set(source.keys())
        if missing_required:
            result.add_warning(f"Hit {hit_index}: champs financiers manquants: {missing_required}")
        
        # Vérification des champs recommandés
        missing_recommended = self.recommended_fields - set(source.keys())
        if missing_recommended:
            result.add_warning(f"Hit {hit_index}: champs recommandés manquants: {missing_recommended}")
        
        # Validations spécifiques aux montants
        if "amount" in source:
            amount = source["amount"]
            try:
                amount_float = float(amount)
                if amount_float == 0:
                    result.add_warning(f"Hit {hit_index}: montant zéro")
                elif amount_float < 0:
                    result.add_warning(f"Hit {hit_index}: montant négatif")
                elif amount_float > 1000000:
                    result.add_warning(f"Hit {hit_index}: montant très élevé")
            except (ValueError, TypeError):
                pass  # Erreur déjà signalée dans la validation parent
        
        # Validation des devises
        if "currency_code" in source:
            currency = source["currency_code"]
            if isinstance(currency, str):
                if len(currency) != 3 or not currency.isupper():
                    result.add_warning(f"Hit {hit_index}: code devise non standard")
        
        # Validation des catégories
        if "category_id" in source:
            category = source["category_id"]
            if isinstance(category, str) and not category.strip():
                result.add_warning(f"Hit {hit_index}: catégorie vide")
        
        return result

class SearchResultMetrics:
    """
    Collecteur de métriques pour les résultats de recherche.
    
    Analyse les résultats et génère des métriques de qualité.
    """
    
    def __init__(self):
        self.metrics = {
            "total_results": 0,
            "avg_score": 0.0,
            "score_distribution": {},
            "field_coverage": {},
            "quality_score": 0.0
        }
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse les résultats et génère des métriques.
        
        Args:
            results: Résultats Elasticsearch
            
        Returns:
            Dictionnaire de métriques
        """
        if not isinstance(results, dict) or "hits" not in results:
            return self.metrics
        
        hits_data = results["hits"]
        if not isinstance(hits_data, dict) or "hits" not in hits_data:
            return self.metrics
        
        hits_list = hits_data["hits"]
        if not isinstance(hits_list, list):
            return self.metrics
        
        # Métriques de base
        self.metrics["total_results"] = len(hits_list)
        
        # Analyse des scores
        scores = [hit.get("_score", 0) for hit in hits_list if hit.get("_score") is not None]
        if scores:
            self.metrics["avg_score"] = sum(scores) / len(scores)
            self.metrics["max_score"] = max(scores)
            self.metrics["min_score"] = min(scores)
            
            # Distribution des scores
            self._analyze_score_distribution(scores)
        
        # Analyse de la couverture des champs
        self._analyze_field_coverage(hits_list)
        
        # Calcul du score de qualité global
        self.metrics["quality_score"] = self._calculate_quality_score()
        
        return self.metrics
    
    def _analyze_score_distribution(self, scores: List[float]):
        """Analyse la distribution des scores."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for score in scores:
            if score >= 2.0:
                distribution["high"] += 1
            elif score >= 1.0:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        self.metrics["score_distribution"] = distribution
    
    def _analyze_field_coverage(self, hits: List[Dict[str, Any]]):
        """Analyse la couverture des champs."""
        field_counts = {}
        total_hits = len(hits)
        
        for hit in hits:
            source = hit.get("_source", {})
            if isinstance(source, dict):
                for field in source.keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
        
        # Calcul des pourcentages de couverture
        coverage = {}
        for field, count in field_counts.items():
            coverage[field] = (count / total_hits) * 100 if total_hits > 0 else 0
        
        self.metrics["field_coverage"] = coverage
    
    def _calculate_quality_score(self) -> float:
        """Calcule un score de qualité global."""
        score = 0.0
        
        # Score basé sur le nombre de résultats
        if self.metrics["total_results"] > 0:
            score += 0.3
        
        # Score basé sur les scores moyens
        if self.metrics.get("avg_score", 0) > 1.0:
            score += 0.4
        
        # Score basé sur la couverture des champs
        coverage = self.metrics.get("field_coverage", {})
        if coverage:
            avg_coverage = sum(coverage.values()) / len(coverage)
            if avg_coverage > 80:
                score += 0.3
            elif avg_coverage > 50:
                score += 0.2
            elif avg_coverage > 20:
                score += 0.1
        
        return min(score, 1.0)

# ==================== FONCTIONS UTILITAIRES ====================

def validate_search_results(results: Dict[str, Any], 
                           validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """
    Fonction utilitaire pour valider des résultats de recherche.
    
    Args:
        results: Résultats à valider
        validation_level: Niveau de validation
        
    Returns:
        ValidationResult
    """
    validator = ResultValidator(validation_level)
    return validator.validate(results)

def validate_financial_results(results: Dict[str, Any]) -> ValidationResult:
    """
    Valide des résultats de recherche financière.
    
    Args:
        results: Résultats à valider
        
    Returns:
        ValidationResult
    """
    validator = FinancialResultValidator()
    return validator.validate(results)

def sanitize_search_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitise des résultats de recherche.
    
    Args:
        results: Résultats à sanitiser
        
    Returns:
        Résultats sanitisés
    """
    validator = ResultValidator()
    validation_result = validator.validate(results)
    return validation_result.sanitized_data if validation_result.sanitized_data else results

def extract_result_metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrait les métadonnées des résultats.
    
    Args:
        results: Résultats Elasticsearch
        
    Returns:
        Métadonnées extraites
    """
    metadata = {
        "took": results.get("took", 0),
        "timed_out": results.get("timed_out", False),
        "total_hits": 0,
        "max_score": None,
        "hits_count": 0
    }
    
    if "hits" in results and isinstance(results["hits"], dict):
        hits_data = results["hits"]
        
        # Total
        total = hits_data.get("total", 0)
        if isinstance(total, dict):
            metadata["total_hits"] = total.get("value", 0)
        else:
            metadata["total_hits"] = total
        
        # Score maximum
        metadata["max_score"] = hits_data.get("max_score")
        
        # Nombre de hits retournés
        if "hits" in hits_data and isinstance(hits_data["hits"], list):
            metadata["hits_count"] = len(hits_data["hits"])
    
    # Métadonnées des shards
    if "_shards" in results:
        metadata["shards"] = results["_shards"]
    
    return metadata

def analyze_result_quality(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse la qualité des résultats de recherche.
    
    Args:
        results: Résultats à analyser
        
    Returns:
        Analyse de qualité
    """
    analyzer = SearchResultMetrics()
    return analyzer.analyze_results(results)

def format_validation_report(validation_result: ValidationResult) -> Dict[str, Any]:
    """
    Formate un rapport de validation lisible.
    
    Args:
        validation_result: Résultat de validation
        
    Returns:
        Rapport formaté
    """
    return {
        "is_valid": validation_result.is_valid,
        "summary": {
            "errors": len(validation_result.errors),
            "warnings": len(validation_result.warnings),
            "security_flags": len(validation_result.security_flags)
        },
        "details": {
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "security_flags": validation_result.security_flags
        },
        "performance": {
            "validation_time_ms": validation_result.validation_time_ms
        }
    }

# ==================== EXPORTS ====================

__all__ = [
    'ResultValidator',
    'FinancialResultValidator', 
    'SearchResultMetrics',
    'validate_search_results',
    'validate_financial_results',
    'sanitize_search_results',
    'extract_result_metadata',
    'analyze_result_quality',
    'format_validation_report'
]