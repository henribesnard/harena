"""
🎭 Prompts Package - Conversation Service

Ce package contient tous les prompts optimisés pour DeepSeek dans le contexte
d'agents AutoGen financiers. Chaque module expose des prompts spécialisés
et des fonctions de formatage pour maximiser la précision des réponses.

Modules:
    intent_prompts: Prompts fallback détection d'intention
    search_prompts: Prompts génération requêtes Search Service  
    response_prompts: Prompts génération réponses contextuelles
    orchestrator_prompts: Prompts orchestrateur principal
"""

# Import des prompts système principaux
from .intent_prompts import (
    INTENT_FALLBACK_SYSTEM_PROMPT,
    INTENT_FALLBACK_USER_TEMPLATE,
    INTENT_EXAMPLES_FEW_SHOT,
    format_intent_prompt
)

from .search_prompts import (
    SEARCH_GENERATION_SYSTEM_PROMPT,
    SEARCH_GENERATION_TEMPLATE,
    SEARCH_EXAMPLES_FEW_SHOT,
    format_search_prompt
)

from .response_prompts import (
    RESPONSE_GENERATION_SYSTEM_PROMPT,
    RESPONSE_GENERATION_TEMPLATE,
    RESPONSE_EXAMPLES_FEW_SHOT,
    format_response_prompt
)

from .orchestrator_prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_WORKFLOW_TEMPLATE,
    format_orchestrator_prompt
)

from .intent_templates import (
    IntentPromptTemplate,
    INTENT_PROMPT_TEMPLATES,
    get_intent_prompt_template,
    build_intent_prompt,
)

# Import des utilitaires de formatage
from .intent_prompts import build_context_summary
from .response_prompts import truncate_search_results
from .orchestrator_prompts import build_workflow_state

__all__ = [
    # Intent Detection
    "INTENT_FALLBACK_SYSTEM_PROMPT",
    "INTENT_FALLBACK_USER_TEMPLATE", 
    "INTENT_EXAMPLES_FEW_SHOT",
    "format_intent_prompt",
    "build_context_summary",

    # Intent Templates
    "IntentPromptTemplate",
    "INTENT_PROMPT_TEMPLATES",
    "get_intent_prompt_template",
    "build_intent_prompt",
    
    # Search Generation
    "SEARCH_GENERATION_SYSTEM_PROMPT",
    "SEARCH_GENERATION_TEMPLATE",
    "SEARCH_EXAMPLES_FEW_SHOT", 
    "format_search_prompt",
    
    # Response Generation
    "RESPONSE_GENERATION_SYSTEM_PROMPT",
    "RESPONSE_GENERATION_TEMPLATE",
    "RESPONSE_EXAMPLES_FEW_SHOT",
    "format_response_prompt",
    "truncate_search_results",
    
    # Orchestrator
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "ORCHESTRATOR_WORKFLOW_TEMPLATE",
    "format_orchestrator_prompt",
    "build_workflow_state"
]

# Métadonnées du package
__version__ = "1.0.0"
__author__ = "Conversation Service Team"
__description__ = "Prompts optimisés DeepSeek pour agents AutoGen financiers"