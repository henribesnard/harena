"""
Validation des configurations Conversation Service v2.0
Validation croisée avec enrichment_service et search_service
"""

import re
import logging
from typing import Dict, Any, List, Set, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Niveaux de validation"""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    level: ValidationLevel
    category: str
    message: str
    field_path: Optional[str] = None
    suggestion: Optional[str] = None

class ConfigValidator:
    """Validateur principal des configurations"""
    
    def __init__(self):
        # Champs réels extraits d'enrichment_service/models.py
        self.known_elasticsearch_fields = {
            # Identifiants
            'transaction_id', 'user_id', 'account_id',
            # Contenu textuel
            'searchable_text', 'primary_description', 'merchant_name',
            # Financier
            'amount', 'amount_abs', 'transaction_type', 'currency_code',
            # Temporel
            'date', 'date_str', 'month_year', 'weekday', 'timestamp',
            # Catégorisation
            'category_id', 'category_name', 'operation_type',
            # Flags
            'is_future', 'is_deleted',
            # Qualité
            'balance_check_passed', 'quality_score', 'indexed_at', 'version'
        }
        
        # Types de champs pour validation
        self.field_types = {
            'transaction_id': int,
            'user_id': int,
            'account_id': int,
            'amount': float,
            'amount_abs': float,
            'transaction_type': str,  # enum: credit/debit
            'currency_code': str,
            'date': str,  # ISO format
            'category_id': int,
            'category_name': str,
            'merchant_name': str,
            'is_future': bool,
            'is_deleted': bool,
            'quality_score': float
        }
        
        # Valeurs énumérées valides
        self.enum_values = {
            'transaction_type': {'credit', 'debit'},
            'currency_code': {'EUR', 'USD', 'GBP'}  # extensible
        }
        
    async def validate_intentions_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la configuration des intentions"""
        results = []
        
        # Validation structure de base
        results.extend(self._validate_intentions_structure(config))
        
        # Validation des groupes d'intentions
        results.extend(self._validate_intent_groups(config.get('intent_groups', {})))
        
        # Validation des exemples few-shot
        results.extend(self._validate_few_shot_examples(config.get('intent_groups', {})))
        
        # Validation des stratégies de requête
        results.extend(self._validate_query_strategies(config.get('query_strategies', {})))
        
        return results
    
    async def validate_entities_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la configuration des entités"""
        results = []
        
        # Validation structure de base
        results.extend(self._validate_entities_structure(config))
        
        # Validation des champs search_service
        results.extend(self._validate_search_service_fields(config.get('search_service_fields', {})))
        
        # Validation des entités temporelles
        results.extend(self._validate_temporal_entities(config.get('temporal_entities', {})))
        
        # Validation des entités monétaires
        results.extend(self._validate_monetary_entities(config.get('monetary_entities', {})))
        
        # Validation des entités textuelles
        results.extend(self._validate_textual_entities(config.get('textual_entities', {})))
        
        # Validation des entités analytiques
        results.extend(self._validate_analytical_entities(config.get('analytical_entities', {})))
        
        return results
    
    async def validate_cross_compatibility(self, 
                                         intentions_config: Dict[str, Any], 
                                         entities_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la compatibilité croisée entre intentions et entités"""
        results = []
        
        # Extraire les champs mentionnés dans les intentions
        mentioned_fields = self._extract_fields_from_intentions(intentions_config)
        
        # Extraire les champs disponibles dans les entités
        available_fields = self._extract_available_fields(entities_config)
        
        # Vérifier la compatibilité
        for field in mentioned_fields:
            if field not in available_fields and field not in self.known_elasticsearch_fields:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    category="cross_compatibility",
                    message=f"Field '{field}' mentioned in intentions but not defined in entities or Elasticsearch schema",
                    field_path=f"intentions→entities.{field}",
                    suggestion=f"Add '{field}' to entities config or verify field name"
                ))
        
        return results
    
    async def validate_search_service_compatibility(self, entities_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la compatibilité avec search_service"""
        results = []
        
        # Vérifier les champs définis vs champs réels
        search_fields = entities_config.get('search_service_fields', {})
        
        for field_category, fields in search_fields.items():
            if isinstance(fields, list):
                for field in fields:
                    if field not in self.known_elasticsearch_fields:
                        results.append(ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.CRITICAL,
                            category="search_service_compatibility",
                            message=f"Field '{field}' not found in Elasticsearch schema",
                            field_path=f"search_service_fields.{field_category}.{field}",
                            suggestion=f"Verify field exists in enrichment_service models or remove from config"
                        ))
        
        # Vérifier les agrégations valides
        results.extend(self._validate_aggregation_syntax(entities_config))
        
        return results
    
    def _validate_intentions_structure(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la structure de base des intentions"""
        results = []
        
        required_fields = ['intent_groups', 'classification_strategies', 'query_strategies']
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.CRITICAL,
                    category="structure",
                    message=f"Missing required field '{field}' in intentions config",
                    field_path=field
                ))
        
        # Validation metadata
        if 'metadata' in config:
            metadata = config['metadata']
            if 'version' not in metadata:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    category="metadata",
                    message="Missing version in metadata",
                    field_path="metadata.version"
                ))
        
        return results
    
    def _validate_intent_groups(self, groups: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les groupes d'intentions"""
        results = []
        
        required_groups = ['TRANSACTION_SEARCH', 'ANALYSIS_INSIGHTS', 'ACCOUNT_BALANCE']
        for group in required_groups:
            if group not in groups:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    category="intent_groups",
                    message=f"Recommended intent group '{group}' missing",
                    field_path=f"intent_groups.{group}",
                    suggestion=f"Consider adding {group} for complete financial coverage"
                ))
        
        for group_name, group_config in groups.items():
            # Vérifier structure du groupe
            if 'description' not in group_config:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.INFO,
                    category="intent_groups",
                    message=f"Intent group '{group_name}' missing description",
                    field_path=f"intent_groups.{group_name}.description"
                ))
            
            if 'patterns_generiques' not in group_config:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.CRITICAL,
                    category="intent_groups",
                    message=f"Intent group '{group_name}' missing patterns_generiques",
                    field_path=f"intent_groups.{group_name}.patterns_generiques"
                ))
        
        return results
    
    def _validate_few_shot_examples(self, groups: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les exemples few-shot"""
        results = []
        
        for group_name, group_config in groups.items():
            patterns = group_config.get('patterns_generiques', {})
            
            for pattern_name, pattern_config in patterns.items():
                examples = pattern_config.get('few_shot_examples', [])
                
                if not examples:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        category="few_shot_examples",
                        message=f"No few-shot examples for pattern {group_name}.{pattern_name}",
                        field_path=f"intent_groups.{group_name}.patterns_generiques.{pattern_name}.few_shot_examples",
                        suggestion="Add at least 2-3 examples for effective few-shot learning"
                    ))
                    continue
                
                if len(examples) < 2:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.INFO,
                        category="few_shot_examples",
                        message=f"Only {len(examples)} examples for pattern {group_name}.{pattern_name}",
                        field_path=f"intent_groups.{group_name}.patterns_generiques.{pattern_name}.few_shot_examples",
                        suggestion="Consider adding more examples (3-5) for better accuracy"
                    ))
                
                # Valider chaque exemple
                for i, example in enumerate(examples):
                    self._validate_single_example(results, example, f"{group_name}.{pattern_name}[{i}]")
        
        return results
    
    def _validate_single_example(self, results: List[ValidationResult], example: Dict[str, Any], path: str):
        """Valide un exemple few-shot individuel"""
        required_fields = ['input', 'output']
        
        for field in required_fields:
            if field not in example:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.CRITICAL,
                    category="few_shot_examples",
                    message=f"Example missing required field '{field}'",
                    field_path=f"{path}.{field}"
                ))
        
        # Valider la structure de l'output
        if 'output' in example:
            output = example['output']
            required_output_fields = ['intent_group', 'intent_subtype', 'entities', 'query_strategy']
            
            for field in required_output_fields:
                if field not in output:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.CRITICAL,
                        category="few_shot_examples",
                        message=f"Example output missing required field '{field}'",
                        field_path=f"{path}.output.{field}"
                    ))
    
    def _validate_entities_structure(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la structure de base des entités"""
        results = []
        
        required_sections = ['search_service_fields', 'temporal_entities', 'monetary_entities', 'textual_entities']
        for section in required_sections:
            if section not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    category="structure", 
                    message=f"Missing recommended section '{section}' in entities config",
                    field_path=section,
                    suggestion=f"Add {section} section for complete entity coverage"
                ))
        
        return results
    
    def _validate_search_service_fields(self, fields_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la configuration des champs search_service"""
        results = []
        
        expected_categories = ['temporal_fields', 'monetary_fields', 'text_fields', 'enum_fields']
        for category in expected_categories:
            if category not in fields_config:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.INFO,
                    category="search_service_fields",
                    message=f"Field category '{category}' not defined",
                    field_path=f"search_service_fields.{category}"
                ))
        
        # Vérifier que tous les champs listés existent réellement
        for category, fields in fields_config.items():
            if isinstance(fields, list):
                for field in fields:
                    if field not in self.known_elasticsearch_fields:
                        results.append(ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.CRITICAL,
                            category="search_service_fields",
                            message=f"Unknown Elasticsearch field '{field}' in category '{category}'",
                            field_path=f"search_service_fields.{category}.{field}",
                            suggestion="Verify field exists in enrichment_service models"
                        ))
        
        return results
    
    def _validate_temporal_entities(self, temporal_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les entités temporelles"""
        results = []
        
        for entity_name, entity_config in temporal_config.items():
            # Vérifier target_field
            target_field = entity_config.get('target_field')
            if target_field and target_field not in ['date', 'timestamp', 'date_str']:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    category="temporal_entities",
                    message=f"Temporal entity '{entity_name}' uses non-standard target_field '{target_field}'",
                    field_path=f"temporal_entities.{entity_name}.target_field",
                    suggestion="Use 'date' as primary temporal field"
                ))
            
            # Valider les exemples temporels
            examples = entity_config.get('few_shot_examples', [])
            for i, example in enumerate(examples):
                if 'output' in example:
                    self._validate_temporal_example_format(results, example['output'], f"{entity_name}[{i}]")
        
        return results
    
    def _validate_temporal_example_format(self, results: List[ValidationResult], output: Any, path: str):
        """Valide le format des exemples temporels"""
        # Si l'output est une string simple (ex: granularité), pas de validation de date
        if isinstance(output, str):
            return
            
        # Si l'output n'est pas un dict, passer
        if not isinstance(output, dict):
            return
            
        # Vérifier format ISO des dates
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
        
        for key, value in output.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey in ['gte', 'lte'] and isinstance(subvalue, str):
                        if not date_pattern.match(subvalue):
                            results.append(ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.WARNING,
                                category="temporal_format",
                                message=f"Date format should be ISO 8601 with timezone: {subvalue}",
                                field_path=f"temporal_entities.{path}.{key}.{subkey}",
                                suggestion="Use format YYYY-MM-DDTHH:MM:SSZ"
                            ))
    
    def _validate_monetary_entities(self, monetary_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les entités monétaires"""
        results = []
        
        for entity_name, entity_config in monetary_config.items():
            target_fields = entity_config.get('target_fields', [])
            
            # Vérifier que amount_abs est privilégié pour les filtres
            if 'amount_abs' not in target_fields and entity_config.get('type') == 'monetary_range':
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.INFO,
                    category="monetary_entities",
                    message=f"Monetary entity '{entity_name}' should include 'amount_abs' for positive range filters",
                    field_path=f"monetary_entities.{entity_name}.target_fields",
                    suggestion="Add 'amount_abs' to target_fields for consistent positive filtering"
                ))
        
        return results
    
    def _validate_textual_entities(self, textual_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les entités textuelles"""
        results = []
        
        for entity_name, entity_config in textual_config.items():
            target_fields = entity_config.get('target_fields', [])
            
            # Vérifier que les champs cibles existent
            for field in target_fields:
                if field not in self.known_elasticsearch_fields:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.CRITICAL,
                        category="textual_entities",
                        message=f"Target field '{field}' in entity '{entity_name}' doesn't exist in Elasticsearch schema",
                        field_path=f"textual_entities.{entity_name}.target_fields",
                        suggestion=f"Remove '{field}' or verify field name in enrichment_service models"
                    ))
        
        return results
    
    def _validate_analytical_entities(self, analytical_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les entités analytiques"""
        results = []
        
        for entity_name, entity_config in analytical_config.items():
            examples = entity_config.get('few_shot_examples', [])
            
            for i, example in enumerate(examples):
                if 'output' in example and 'aggregations' in example['output']:
                    self._validate_aggregation_structure(results, example['output']['aggregations'], f"{entity_name}[{i}]")
        
        return results
    
    def _validate_aggregation_structure(self, results: List[ValidationResult], aggs: Dict[str, Any], path: str):
        """Valide la structure des agrégations Elasticsearch"""
        for agg_name, agg_config in aggs.items():
            if isinstance(agg_config, dict):
                # Vérifier les types d'agrégation valides
                agg_types = {'terms', 'sum', 'avg', 'date_histogram', 'range', 'filter'}
                found_types = set(agg_config.keys()) & agg_types
                
                if not found_types:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.CRITICAL,
                        category="aggregations",
                        message=f"No valid aggregation type found in '{agg_name}'",
                        field_path=f"analytical_entities.{path}.aggregations.{agg_name}",
                        suggestion=f"Use one of: {', '.join(agg_types)}"
                    ))
                
                # Vérifier les champs utilisés dans les agrégations
                self._validate_aggregation_fields(results, agg_config, f"{path}.{agg_name}")
    
    def _validate_aggregation_fields(self, results: List[ValidationResult], agg_config: Dict[str, Any], path: str):
        """Valide les champs utilisés dans les agrégations"""
        for key, value in agg_config.items():
            if isinstance(value, dict) and 'field' in value:
                field = value['field']
                base_field = field.replace('.keyword', '')  # Retirer le suffixe .keyword
                
                if base_field not in self.known_elasticsearch_fields:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.CRITICAL,
                        category="aggregation_fields",
                        message=f"Aggregation field '{field}' doesn't exist in Elasticsearch schema",
                        field_path=f"analytical_entities.{path}.{key}.field",
                        suggestion=f"Verify field exists or use correct field name"
                    ))
    
    def _validate_query_strategies(self, strategies: Dict[str, Any]) -> List[ValidationResult]:
        """Valide les stratégies de requête"""
        results = []
        
        expected_strategies = ['simple_strategies', 'aggregation_strategies']
        for strategy_type in expected_strategies:
            if strategy_type not in strategies:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.INFO,
                    category="query_strategies",
                    message=f"Strategy type '{strategy_type}' not defined",
                    field_path=f"query_strategies.{strategy_type}"
                ))
        
        return results
    
    def _validate_aggregation_syntax(self, entities_config: Dict[str, Any]) -> List[ValidationResult]:
        """Valide la syntaxe des agrégations"""
        results = []
        # Implementation détaillée selon besoin
        return results
    
    def _extract_fields_from_intentions(self, intentions_config: Dict[str, Any]) -> Set[str]:
        """Extrait tous les champs mentionnés dans les intentions"""
        fields = set()
        
        intent_groups = intentions_config.get('intent_groups', {})
        for group_config in intent_groups.values():
            patterns = group_config.get('patterns_generiques', {})
            for pattern_config in patterns.values():
                examples = pattern_config.get('few_shot_examples', [])
                for example in examples:
                    if 'output' in example and 'entities' in example['output']:
                        fields.update(self._extract_fields_from_entities(example['output']['entities']))
        
        return fields
    
    def _extract_fields_from_entities(self, entities: Dict[str, Any]) -> Set[str]:
        """Extrait les champs d'un dictionnaire d'entités"""
        fields = set()
        
        def extract_recursive(obj):
            if isinstance(obj, dict):
                fields.update(obj.keys())
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(entities)
        return fields
    
    def _extract_available_fields(self, entities_config: Dict[str, Any]) -> Set[str]:
        """Extrait tous les champs disponibles dans la configuration entités"""
        fields = set()
        
        search_fields = entities_config.get('search_service_fields', {})
        for field_list in search_fields.values():
            if isinstance(field_list, list):
                fields.update(field_list)
        
        return fields

# Instance globale
validator = ConfigValidator()

async def validate_full_configuration(
    intentions_config: Dict[str, Any],
    entities_config: Dict[str, Any]
) -> Tuple[bool, List[ValidationResult]]:
    """Valide la configuration complète"""
    all_results = []
    
    # Validation des intentions
    all_results.extend(await validator.validate_intentions_config(intentions_config))
    
    # Validation des entités
    all_results.extend(await validator.validate_entities_config(entities_config))
    
    # Validation croisée
    all_results.extend(await validator.validate_cross_compatibility(intentions_config, entities_config))
    
    # Validation search_service
    all_results.extend(await validator.validate_search_service_compatibility(entities_config))
    
    # Déterminer si la configuration est globalement valide
    has_critical_errors = any(result.level == ValidationLevel.CRITICAL for result in all_results)
    is_valid = not has_critical_errors
    
    return is_valid, all_results