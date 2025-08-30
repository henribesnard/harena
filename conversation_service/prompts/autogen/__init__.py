"""
Prompts et extensions AutoGen pour la collaboration multi-agents
"""

from .collaboration_extensions import (
    get_intent_classification_with_team_collaboration,
    get_entity_extraction_with_team_collaboration,
    get_adaptive_entity_prompt,
    get_team_collaboration_guidelines,
    get_entity_focus_mapping,
    get_entity_hints_for_intent
)

from .team_orchestration import (
    MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
    TEAM_ORCHESTRATION_FALLBACK_PROMPT,
    SINGLE_AGENT_FALLBACK_PROMPTS,
    get_orchestration_prompt_for_context,
    get_workflow_completion_message,
    get_workflow_error_message
)

__all__ = [
    # Extensions collaboration prompts existants
    "get_intent_classification_with_team_collaboration",
    "get_entity_extraction_with_team_collaboration",
    "get_adaptive_entity_prompt",
    "get_team_collaboration_guidelines", 
    "get_entity_focus_mapping",
    "get_entity_hints_for_intent",
    
    # Orchestration équipe
    "MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT",
    "TEAM_ORCHESTRATION_FALLBACK_PROMPT", 
    "SINGLE_AGENT_FALLBACK_PROMPTS",
    "get_orchestration_prompt_for_context",
    "get_workflow_completion_message",
    "get_workflow_error_message"
]