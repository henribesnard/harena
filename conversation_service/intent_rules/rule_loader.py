"""
🔧 Rule Loader - Chargeur de règles de détection d'intentions

Ce module charge et valide les fichiers de configuration JSON des règles
d'intentions financières, conversationnelles et d'extraction d'entités.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Catégories d'intentions supportées"""
    SEARCH = "SEARCH"
    ANALYZE = "ANALYZE"
    CONVERSATIONAL = "CONVERSATIONAL"


class RuleMatch(NamedTuple):
    """Résultat d'un match de règle"""
    intent: str
    score: float
    pattern_matched: str
    entities: Dict
    method: str = "rule_based"


@dataclass
class PatternRule:
    """Règle de pattern avec métadonnées"""
    regex: re.Pattern
    weight: float
    entity_extract: Optional[Dict] = None
    extract_group: Optional[int] = None
    extract_groups: Optional[List[int]] = None
    normalize: Optional[str] = None


@dataclass
class IntentRule:
    """Règle d'intention complète"""
    intent: str
    description: str
    intent_category: IntentCategory
    confidence: float
    priority: int
    patterns: List[PatternRule]
    exact_matches: set
    search_parameters: Optional[Dict] = None
    default_filters: Optional[List[Dict]] = None
    examples: Optional[List[str]] = None
    no_search_needed: bool = False
    
    def __post_init__(self):
        """Validation post-initialisation"""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.priority < 1:
            raise ValueError(f"Priority must be >= 1, got {self.priority}")


@dataclass 
class EntityPattern:
    """Pattern d'extraction d'entité"""
    regex: re.Pattern
    entity_type: str
    extract_group: Optional[int] = None
    extract_groups: Optional[List[int]] = None
    normalize: Optional[str] = None
    weight: float = 1.0


class RuleLoader:
    """Chargeur et gestionnaire des règles de détection"""
    
    def __init__(self, rules_directory: Optional[Path] = None):
        """
        Initialise le chargeur de règles
        
        Args:
            rules_directory: Répertoire contenant les fichiers de règles
        """
        if rules_directory is None:
            rules_directory = Path(__file__).parent
            
        self.rules_dir = Path(rules_directory)
        
        # Stockage des règles chargées
        self.financial_rules: Dict[str, IntentRule] = {}
        self.conversational_rules: Dict[str, IntentRule] = {}
        self.entity_patterns: Dict[str, List[EntityPattern]] = {}
        
        # Métadonnées
        self.version_info: Dict = {}
        self.global_settings: Dict = {}
        
        # Chargement initial
        self._load_all_rules()
    
    def _load_all_rules(self) -> None:
        """Charge tous les fichiers de règles"""
        try:
            # Chargement patterns financiers
            self._load_financial_patterns()
            
            # Chargement patterns conversationnels
            self._load_conversational_patterns()
            
            # Chargement patterns d'entités
            self._load_entity_patterns()
            
            logger.info(f"Loaded {len(self.financial_rules)} financial rules")
            logger.info(f"Loaded {len(self.conversational_rules)} conversational rules")
            logger.info(f"Loaded {len(self.entity_patterns)} entity pattern types")
            
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            raise
    
    def _load_financial_patterns(self) -> None:
        """Charge les patterns financiers depuis financial_patterns.json"""
        file_path = self.rules_dir / "financial_patterns.json"
        
        if not file_path.exists():
            logger.warning(f"Financial patterns file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.version_info['financial'] = {
                'version': config.get('version', 'unknown'),
                'last_updated': config.get('last_updated', 'unknown')
            }
            
            self.global_settings.update(config.get('global_settings', {}))
            
            # Parsing des intentions
            for intent_name, intent_config in config.get('intents', {}).items():
                rule = self._parse_intent_rule(intent_name, intent_config)
                self.financial_rules[intent_name] = rule
                
        except Exception as e:
            logger.error(f"Error loading financial patterns: {e}")
            raise
    
    def _load_conversational_patterns(self) -> None:
        """Charge les patterns conversationnels depuis conversational_patterns.json"""
        file_path = self.rules_dir / "conversational_patterns.json"
        
        if not file_path.exists():
            logger.warning(f"Conversational patterns file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.version_info['conversational'] = {
                'version': config.get('version', 'unknown'),
                'last_updated': config.get('last_updated', 'unknown')
            }
            
            # Parsing des intentions conversationnelles
            for intent_name, intent_config in config.get('intents', {}).items():
                rule = self._parse_intent_rule(intent_name, intent_config)
                self.conversational_rules[intent_name] = rule
                
        except Exception as e:
            logger.error(f"Error loading conversational patterns: {e}")
            raise
    
    def _load_entity_patterns(self) -> None:
        """Charge les patterns d'entités depuis entity_patterns.json"""
        file_path = self.rules_dir / "entity_patterns.json"
        
        if not file_path.exists():
            logger.warning(f"Entity patterns file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.version_info['entities'] = {
                'version': config.get('version', 'unknown'),
                'last_updated': config.get('last_updated', 'unknown')
            }
            
            # Parsing des patterns d'entités
            for entity_type, entity_config in config.get('entity_types', {}).items():
                patterns = self._parse_entity_patterns(entity_type, entity_config)
                self.entity_patterns[entity_type] = patterns
                
        except Exception as e:
            logger.error(f"Error loading entity patterns: {e}")
            raise
    
    def _parse_intent_rule(self, intent_name: str, config: Dict) -> IntentRule:
        """Parse une règle d'intention depuis sa configuration"""
        try:
            # Compilation des patterns regex
            pattern_rules = []
            for pattern_config in config.get('patterns', []):
                regex_pattern = re.compile(
                    pattern_config['regex'], 
                    re.IGNORECASE if not pattern_config.get('case_sensitive', False) else 0
                )
                
                pattern_rule = PatternRule(
                    regex=regex_pattern,
                    weight=pattern_config.get('weight', 1.0),
                    entity_extract=pattern_config.get('entity_extract'),
                    extract_group=pattern_config.get('extract_group'),
                    extract_groups=pattern_config.get('extract_groups'),
                    normalize=pattern_config.get('normalize')
                )
                pattern_rules.append(pattern_rule)
            
            # Création de la règle d'intention
            rule = IntentRule(
                intent=intent_name,
                description=config.get('description', ''),
                intent_category=IntentCategory(config.get('intent_category', 'SEARCH')),
                confidence=config.get('confidence', 0.8),
                priority=config.get('priority', 5),
                patterns=pattern_rules,
                exact_matches=set(config.get('exact_matches', [])),
                search_parameters=config.get('search_parameters'),
                default_filters=config.get('default_filters'),
                examples=config.get('examples'),
                no_search_needed=config.get('no_search_needed', False)
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Error parsing intent rule {intent_name}: {e}")
            raise
    
    def _parse_entity_patterns(self, entity_type: str, config: Dict) -> List[EntityPattern]:
        """Parse les patterns d'une entité depuis sa configuration"""
        patterns = []
        
        try:
            # Patterns regex
            for pattern_config in config.get('patterns', []):
                regex_pattern = re.compile(
                    pattern_config['regex'],
                    re.IGNORECASE if not pattern_config.get('case_sensitive', False) else 0
                )
                
                pattern = EntityPattern(
                    regex=regex_pattern,
                    entity_type=entity_type,
                    extract_group=pattern_config.get('extract_group'),
                    extract_groups=pattern_config.get('extract_groups'),
                    normalize=pattern_config.get('normalize'),
                    weight=pattern_config.get('weight', 1.0)
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error parsing entity patterns for {entity_type}: {e}")
            raise
    
    def get_financial_rules(self) -> Dict[str, IntentRule]:
        """Retourne les règles financières chargées"""
        return self.financial_rules.copy()
    
    def get_conversational_rules(self) -> Dict[str, IntentRule]:
        """Retourne les règles conversationnelles chargées"""
        return self.conversational_rules.copy()
    
    def get_entity_patterns(self, entity_type: Optional[str] = None) -> Union[Dict[str, List[EntityPattern]], List[EntityPattern]]:
        """
        Retourne les patterns d'entités
        
        Args:
            entity_type: Type d'entité spécifique (optionnel)
            
        Returns:
            Patterns pour le type demandé ou tous les patterns
        """
        if entity_type:
            return self.entity_patterns.get(entity_type, [])
        return self.entity_patterns.copy()
    
    def get_all_rules(self) -> Dict[str, IntentRule]:
        """Retourne toutes les règles (financières + conversationnelles)"""
        all_rules = {}
        all_rules.update(self.financial_rules)
        all_rules.update(self.conversational_rules)
        return all_rules
    
    def get_rules_by_category(self, category: IntentCategory) -> Dict[str, IntentRule]:
        """Retourne les règles d'une catégorie spécifique"""
        all_rules = self.get_all_rules()
        return {
            name: rule for name, rule in all_rules.items()
            if rule.intent_category == category
        }
    
    def get_rules_by_priority(self, min_priority: int = 1, max_priority: int = 10) -> List[IntentRule]:
        """Retourne les règles triées par priorité dans une plage donnée"""
        all_rules = self.get_all_rules()
        filtered_rules = [
            rule for rule in all_rules.values()
            if min_priority <= rule.priority <= max_priority
        ]
        return sorted(filtered_rules, key=lambda r: r.priority)
    
    def validate_rules(self) -> Dict[str, List[str]]:
        """
        Valide toutes les règles chargées
        
        Returns:
            Dictionnaire des erreurs trouvées par type de règle
        """
        errors = {
            'financial': [],
            'conversational': [],
            'entities': []
        }
        
        # Validation règles financières
        for intent_name, rule in self.financial_rules.items():
            rule_errors = self._validate_intent_rule(intent_name, rule)
            errors['financial'].extend(rule_errors)
        
        # Validation règles conversationnelles
        for intent_name, rule in self.conversational_rules.items():
            rule_errors = self._validate_intent_rule(intent_name, rule)
            errors['conversational'].extend(rule_errors)
        
        # Validation patterns entités
        for entity_type, patterns in self.entity_patterns.items():
            entity_errors = self._validate_entity_patterns(entity_type, patterns)
            errors['entities'].extend(entity_errors)
        
        return errors
    
    def _validate_intent_rule(self, intent_name: str, rule: IntentRule) -> List[str]:
        """Valide une règle d'intention"""
        errors = []
        
        # Vérification patterns vides
        if not rule.patterns and not rule.exact_matches:
            errors.append(f"{intent_name}: No patterns or exact matches defined")
        
        # Vérification regex patterns
        for i, pattern in enumerate(rule.patterns):
            try:
                # Test regex avec exemple
                pattern.regex.search("test")
            except Exception as e:
                errors.append(f"{intent_name}: Invalid regex pattern {i}: {e}")
        
        # Vérification cohérence confidence/priority
        if rule.confidence > 0.95 and rule.priority > 3:
            errors.append(f"{intent_name}: High confidence ({rule.confidence}) with low priority ({rule.priority})")
        
        return errors
    
    def _validate_entity_patterns(self, entity_type: str, patterns: List[EntityPattern]) -> List[str]:
        """Valide les patterns d'une entité"""
        errors = []
        
        if not patterns:
            errors.append(f"{entity_type}: No patterns defined")
            return errors
        
        for i, pattern in enumerate(patterns):
            try:
                # Test regex
                pattern.regex.search("test")
            except Exception as e:
                errors.append(f"{entity_type}: Invalid regex pattern {i}: {e}")
        
        return errors
    
    def reload_rules(self) -> None:
        """Recharge tous les fichiers de règles"""
        logger.info("Reloading all rule files...")
        
        # Clear existing rules
        self.financial_rules.clear()
        self.conversational_rules.clear()
        self.entity_patterns.clear()
        self.version_info.clear()
        self.global_settings.clear()
        
        # Reload all rules
        self._load_all_rules()
        
        logger.info("Rules reloaded successfully")
    
    def get_version_info(self) -> Dict:
        """Retourne les informations de version des règles"""
        return self.version_info.copy()
    
    def get_global_settings(self) -> Dict:
        """Retourne les paramètres globaux"""
        return self.global_settings.copy()
    
    def export_rules_summary(self) -> Dict:
        """Exporte un résumé des règles chargées pour debug"""
        return {
            'version_info': self.version_info,
            'financial_rules_count': len(self.financial_rules),
            'conversational_rules_count': len(self.conversational_rules),
            'entity_types_count': len(self.entity_patterns),
            'financial_rules': {
                name: {
                    'category': rule.intent_category.value,
                    'confidence': rule.confidence,
                    'priority': rule.priority,
                    'patterns_count': len(rule.patterns),
                    'exact_matches_count': len(rule.exact_matches),
                    'examples': rule.examples[:3] if rule.examples else []
                }
                for name, rule in self.financial_rules.items()
            },
            'conversational_rules': {
                name: {
                    'category': rule.intent_category.value,
                    'confidence': rule.confidence,
                    'priority': rule.priority,
                    'patterns_count': len(rule.patterns),
                    'exact_matches_count': len(rule.exact_matches)
                }
                for name, rule in self.conversational_rules.items()
            },
            'entity_patterns': {
                entity_type: len(patterns)
                for entity_type, patterns in self.entity_patterns.items()
            }
        }


# Factory function pour faciliter l'utilisation
def create_rule_loader(rules_directory: Optional[Path] = None) -> RuleLoader:
    """
    Factory function pour créer un RuleLoader
    
    Args:
        rules_directory: Répertoire des règles (optionnel)
        
    Returns:
        Instance RuleLoader configurée
    """
    return RuleLoader(rules_directory)


# Validation à l'import
if __name__ == "__main__":
    # Test du chargeur de règles
    loader = create_rule_loader()
    
    print("=== RULE LOADER TEST ===")
    print(f"Financial rules: {len(loader.financial_rules)}")
    print(f"Conversational rules: {len(loader.conversational_rules)}")
    print(f"Entity patterns: {len(loader.entity_patterns)}")
    
    # Validation
    errors = loader.validate_rules()
    if any(errors.values()):
        print("\n=== VALIDATION ERRORS ===")
        for rule_type, error_list in errors.items():
            if error_list:
                print(f"{rule_type.upper()}:")
                for error in error_list:
                    print(f"  - {error}")
    else:
        print("\n✅ All rules validated successfully!")
    
    # Export summary
    summary = loader.export_rules_summary()
    print(f"\n=== RULES SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))