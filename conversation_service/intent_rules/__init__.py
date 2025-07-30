"""
🔧 Intent Rules Package - Module de détection d'intentions hybride

Ce package implémente un système de détection d'intentions à deux niveaux :
- Niveau 0 : Règles et patterns configurables (ultra-rapide, 0 coût)
- Niveau 1 : Fallback IA via DeepSeek (intelligent, coût optimisé)

Architecture :
- rule_loader.py : Chargement et validation des fichiers de configuration ✅
- pattern_matcher.py : Matching patterns et extraction d'entités ✅
- rule_engine.py : Moteur de règles complet avec scoring ✅
- intent_detector.py : Interface principale de détection (à venir)

Fichiers de configuration :
- financial_patterns.json : Intentions financières (SEARCH, ANALYZE) ✅
- conversational_patterns.json : Intentions conversationnelles ✅
- entity_patterns.json : Patterns d'extraction d'entités ✅
"""

from typing import Dict, List, Optional, Union, NamedTuple
from pathlib import Path

# Version du package
__version__ = "1.0.0"
__author__ = "Harena Financial AI Team"
__description__ = "Système hybride de détection d'intentions financières"

# Exports principaux - Rule Loader
from .rule_loader import (
    RuleLoader,
    IntentRule,
    PatternRule,
    EntityPattern,
    IntentCategory,
    RuleMatch as LoaderRuleMatch,
    create_rule_loader
)

# Exports Pattern Matcher
from .pattern_matcher import (
    PatternMatcher,
    EntityMatch,
    ExtractionResult,
    EntityNormalizer,
    create_pattern_matcher
)

# Exports Rule Engine
from .rule_engine import (
    RuleEngine,
    RuleMatch,
    MatchingContext,
    ExactMatcher,
    PatternMatchScorer,
    create_rule_engine
)

# Exports des structures de données
__all__ = [
    # Classes principales
    "RuleLoader",
    "PatternMatcher", 
    "RuleEngine",
    
    # Structures de données rule_loader
    "IntentRule",
    "PatternRule", 
    "EntityPattern",
    
    # Structures de données pattern_matcher
    "EntityMatch",
    "ExtractionResult",
    "EntityNormalizer",
    
    # Structures de données rule_engine
    "RuleMatch",
    "MatchingContext",
    "ExactMatcher",
    "PatternMatchScorer",
    
    # Enums et types
    "IntentCategory",
    "LoaderRuleMatch",
    
    # Factory functions
    "create_rule_loader",
    "create_pattern_matcher", 
    "create_rule_engine",
    
    # Constantes
    "__version__",
    "__author__",
    "__description__"
]

# Configuration par défaut du package
DEFAULT_RULES_DIR = Path(__file__).parent
DEFAULT_CONFIG = {
    "cache_size": 1000,
    "confidence_threshold": 0.7,
    "enable_ai_fallback": True,
    "max_patterns_per_intent": 50,
    "validation_strict": True,
    "performance_monitoring": True
}

# Initialisation paresseuse des composants
_default_loader: Optional[RuleLoader] = None
_default_pattern_matcher: Optional[PatternMatcher] = None
_default_rule_engine: Optional[RuleEngine] = None

def get_default_loader() -> RuleLoader:
    """
    Retourne le RuleLoader par défaut du package
    
    Returns:
        Instance RuleLoader configurée avec les règles par défaut
    """
    global _default_loader
    
    if _default_loader is None:
        _default_loader = create_rule_loader(DEFAULT_RULES_DIR)
    
    return _default_loader

def get_default_pattern_matcher() -> PatternMatcher:
    """
    Retourne the PatternMatcher par défaut du package
    
    Returns:
        Instance PatternMatcher configurée avec les patterns par défaut
    """
    global _default_pattern_matcher
    
    if _default_pattern_matcher is None:
        loader = get_default_loader()
        _default_pattern_matcher = create_pattern_matcher(loader)
    
    return _default_pattern_matcher

def get_default_rule_engine() -> RuleEngine:
    """
    Retourne le RuleEngine par défaut du package
    
    Returns:
        Instance RuleEngine configurée et prête à l'emploi
    """
    global _default_rule_engine
    
    if _default_rule_engine is None:
        loader = get_default_loader()
        pattern_matcher = get_default_pattern_matcher()
        _default_rule_engine = create_rule_engine(loader, pattern_matcher)
    
    return _default_rule_engine

def reload_default_rules() -> None:
    """
    Recharge les règles par défaut du package
    
    Utile pour le développement et les mises à jour à chaud
    """
    global _default_loader, _default_pattern_matcher, _default_rule_engine
    
    if _default_loader is not None:
        _default_loader.reload_rules()
    
    if _default_rule_engine is not None:
        _default_rule_engine.reload_rules()
    
    # Force la recréation des composants
    _default_pattern_matcher = None
    
    print("✅ Default rules reloaded successfully")

def get_package_info() -> Dict:
    """
    Retourne les informations complètes du package
    
    Returns:
        Dictionnaire avec version, statistiques, configuration
    """
    try:
        loader = get_default_loader()
        engine = get_default_rule_engine()
        
        return {
            "package": {
                "name": "intent_rules",
                "version": __version__,
                "author": __author__,
                "description": __description__
            },
            "rules": {
                "financial_count": len(loader.financial_rules),
                "conversational_count": len(loader.conversational_rules), 
                "entity_types_count": len(loader.entity_patterns),
                "version_info": loader.get_version_info()
            },
            "performance": engine.get_performance_stats(),
            "config": DEFAULT_CONFIG
        }
    except Exception as e:
        return {
            "package": {
                "name": "intent_rules",
                "version": __version__,
                "status": "error",
                "error": str(e)
            }
        }

def quick_detect_intent(text: str, confidence_threshold: float = 0.7) -> Optional[RuleMatch]:
    """
    Fonction utilitaire pour détection rapide d'intention
    
    Args:
        text: Texte à analyser
        confidence_threshold: Seuil de confidence minimum
        
    Returns:
        RuleMatch si intention détectée, None sinon
    """
    try:
        engine = get_default_rule_engine()
        return engine.match_intent(text, confidence_threshold=confidence_threshold)
    except Exception as e:
        import logging
        logging.error(f"Error in quick_detect_intent: {e}")
        return None

def extract_entities_quick(text: str, target_entities: Optional[List[str]] = None) -> ExtractionResult:
    """
    Fonction utilitaire pour extraction rapide d'entités
    
    Args:
        text: Texte à analyser
        target_entities: Types d'entités spécifiques (optionnel)
        
    Returns:
        ExtractionResult avec entités trouvées
    """
    try:
        matcher = get_default_pattern_matcher()
        return matcher.extract_entities(text, target_entities)
    except Exception as e:
        import logging
        logging.error(f"Error in extract_entities_quick: {e}")
        return ExtractionResult(
            entities={},
            total_matches=0,
            extraction_time_ms=0.0,
            cache_hit=False
        )

def get_performance_report() -> Dict:
    """
    Génère un rapport de performance complet du package
    
    Returns:
        Rapport détaillé avec métriques et recommandations
    """
    try:
        engine = get_default_rule_engine()
        stats = engine.get_performance_stats()
        
        # Analyse des performances
        cache_hit_rate = stats["cache_stats"]["hit_rate_percent"]
        avg_exact_time = stats["matcher_stats"]["exact_matcher"]["avg_execution_time_ms"]
        avg_pattern_time = stats["matcher_stats"]["pattern_matcher"]["avg_execution_time_ms"]
        
        # Recommandations
        recommendations = []
        
        if cache_hit_rate < 60:
            recommendations.append("Consider increasing cache size for better performance")
        
        if avg_pattern_time > 20:
            recommendations.append("Pattern matching is slow, consider optimizing regex patterns")
        
        if stats["engine_info"]["total_rules"] > 50:
            recommendations.append("Large number of rules may impact performance")
        
        return {
            "performance_stats": stats,
            "analysis": {
                "cache_efficiency": "excellent" if cache_hit_rate > 80 else "good" if cache_hit_rate > 60 else "needs_improvement",
                "exact_match_speed": "excellent" if avg_exact_time < 2 else "good" if avg_exact_time < 5 else "slow",
                "pattern_match_speed": "excellent" if avg_pattern_time < 10 else "good" if avg_pattern_time < 20 else "slow"
            },
            "recommendations": recommendations,
            "summary": f"Processed {stats['cache_stats']['hits'] + stats['cache_stats']['misses']} requests with {cache_hit_rate:.1f}% cache hit rate"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed_to_generate_report"
        }

# Validation à l'import
def _validate_package():
    """Validation du package à l'import avec gestion d'erreurs robuste"""
    try:
        # Test chargement des règles par défaut
        loader = get_default_loader()
        
        # Validation basique
        errors = loader.validate_rules()
        total_errors = sum(len(error_list) for error_list in errors.values())
        
        if total_errors > 0:
            import warnings
            warnings.warn(
                f"Intent rules package loaded with {total_errors} validation errors. "
                f"Use get_package_info() for details.",
                UserWarning
            )
        
        # Test création du rule engine
        engine = get_default_rule_engine()
        
        # Log succès
        import logging
        logging.info(f"Intent rules package loaded successfully: "
                    f"{len(loader.all_rules)} rules, "
                    f"{len(loader.entity_patterns)} entity types")
        
        return True
        
    except Exception as e:
        import warnings
        warnings.warn(
            f"Intent rules package failed to load properly: {e}. "
            f"Some features may not work correctly.",
            ImportWarning
        )
        return False

# Validation à l'import (peut être désactivée via variable d'environnement)
import os
if os.getenv("INTENT_RULES_SKIP_VALIDATION") != "1":
    _validation_success = _validate_package()
else:
    _validation_success = True
    print("⚠️ Intent rules validation skipped via environment variable")

# Ajout d'informations de debug si nécessaire
if os.getenv("INTENT_RULES_DEBUG") == "1":
    def debug_info():
        """Affiche les informations de debug du package"""
        print("🔍 Intent Rules Debug Information")
        print("=" * 50)
        
        try:
            info = get_package_info()
            print(f"📦 Package: {info['package']['name']} v{info['package']['version']}")
            print(f"📊 Rules: {info['rules']['financial_count']} financial, {info['rules']['conversational_count']} conversational")
            print(f"🔧 Entities: {info['rules']['entity_types_count']} types")
            
            if 'performance' in info:
                perf = info['performance']
                print(f"💾 Cache: {perf['cache_stats']['hit_rate_percent']:.1f}% hit rate")
                print(f"⚡ Performance: {perf['matcher_stats']['exact_matcher']['avg_execution_time_ms']:.2f}ms exact, {perf['matcher_stats']['pattern_matcher']['avg_execution_time_ms']:.2f}ms pattern")
            
        except Exception as e:
            print(f"❌ Error getting debug info: {e}")
        
        print("=" * 50)
    
    # Affichage automatique en mode debug
    debug_info()
    
    # Export de la fonction debug
    __all__.append("debug_info")

# Nettoyage des imports temporaires - LAISSÉ À LA FIN
# del os  # ← COMMENTÉ pour éviter l'erreur

# Définition d'un alias pour compatibilité
IntentDetector = RuleEngine  # Alias pour transition
__all__.append("IntentDetector")