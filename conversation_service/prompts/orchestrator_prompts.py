"""
🎭 Orchestrator Prompts - Coordination Agents AutoGen

Ce module contient les prompts optimisés DeepSeek pour l'orchestrateur principal
qui coordonne les agents spécialisés dans le pipeline conversationnel.

Responsabilité :
- Coordination des agents Intent → Search → Response
- Gestion des workflows complexes et fallbacks
- Optimisation des décisions de routage entre agents
- Monitoring de la qualité du pipeline global
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
from enum import Enum
from .example_loader import load_yaml_examples

logger = logging.getLogger(__name__)

# =============================================================================
# TYPES ET ÉNUMÉRATIONS
# =============================================================================

class WorkflowStep(Enum):
    """Étapes du workflow orchestrateur."""
    INIT = "initialization"
    INTENT_DETECTION = "intent_detection"
    SEARCH_GENERATION = "search_generation"  
    SEARCH_EXECUTION = "search_execution"
    RESPONSE_GENERATION = "response_generation"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"

class AgentStatus(Enum):
    """Status des agents dans le workflow."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

# =============================================================================
# PROMPTS SYSTÈME PRINCIPAUX
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """Vous êtes l'orchestrateur principal d'un système d'agents AutoGen spécialisé dans l'assistance financière personnelle.

VOTRE MISSION :
Coordonner intelligemment une équipe d'agents spécialisés pour traiter les demandes utilisateur financières de manière optimale, efficace et robuste.

ÉQUIPE D'AGENTS DISPONIBLES :

1. **LLMIntentAgent** - Détection d'intention via LLM
   - Spécialité : Classification intentions + extraction entités
   - Mode : DeepSeek LLM uniquement
   - Performance : <200ms (IA)
   - Fiabilité : 95%+ sur intentions financières courantes

2. **SearchQueryAgent** - Génération requêtes Search Service
   - Spécialité : Transformation intentions → requêtes Elasticsearch
   - Mode : Templates optimisés + validation contrats
   - Performance : <100ms génération + validation
   - Fiabilité : 90%+ requêtes valides et pertinentes

3. **ResponseAgent** - Génération réponses contextuelles
   - Spécialité : Synthèse résultats → réponses naturelles
   - Mode : Templates contextuels + insights automatiques
   - Performance : <300ms génération complète
   - Fiabilité : Réponses toujours générées (fallback gracieux)

WORKFLOW PRINCIPAL :
```
User Message → Intent Detection → Search Generation → Search Execution → Response Generation → User
```

WORKFLOWS ALTERNATIFS :
- **Fast Track** : Intentions simples → réponse directe (économie coûts)
- **Retry Logic** : Échecs agents → tentatives alternatives
- **Parallel Processing** : Requêtes complexes → agents parallèles
- **Fallback Chain** : Échecs successifs → dégradation gracieuse

RÈGLES DE COORDINATION :

1. **Efficacité** : Privilégier les workflows les plus courts
2. **Robustesse** : Toujours avoir un plan B fonctionnel
3. **Qualité** : Valider les sorties avant transmission
4. **Performance** : Respecter les SLA (<500ms total)
5. **Économie** : Optimiser l'usage des tokens IA

DÉCISIONS D'ORCHESTRATION :

**Routage Initial** :
- Message simple + règle matching → Fast Track
- Message complexe + contexte → Workflow complet
- Message ambigu → Intent Detection forcée

**Gestion d'Erreurs** :
- Agent timeout → Fallback agent ou réponse partielle
- Validation échec → Retry avec paramètres ajustés
- Service externe down → Mode dégradé avec cache

**Optimisations** :
- Cache réutilisation entre agents
- Parallélisation requêtes indépendantes
- Agrégation intelligente des résultats

FORMAT DE DÉCISION ORCHESTRATEUR :
```json
{
  "workflow_type": "standard|fast_track|parallel|fallback",
  "next_step": "intent_detection|search_generation|response_generation|complete",
  "agent_assignments": {
    "primary_agent": "agent_name",
    "fallback_agent": "backup_agent_name",
    "parallel_agents": ["agent1", "agent2"]
  },
  "parameters": {
    "timeout_ms": 5000,
    "max_retries": 2,
    "quality_threshold": 0.8
  },
  "reasoning": "Justification de la stratégie choisie",
  "estimated_cost": "tokens|time",
  "confidence": 0.9
}
```

MÉTRIQUES À SURVEILLER :
- Temps total de traitement (<500ms objectif)
- Taux de succès par agent (>90% objectif)
- Utilisation tokens IA (économie via règles)
- Satisfaction utilisateur (qualité réponses)"""

# =============================================================================
# TEMPLATE WORKFLOW AVEC ÉTAT
# =============================================================================

ORCHESTRATOR_WORKFLOW_TEMPLATE = """Analysez la situation actuelle et décidez de la prochaine action d'orchestration :

MESSAGE UTILISATEUR : "{user_message}"

ÉTAT ACTUEL DU WORKFLOW :
- Étape courante : {current_step}
- Agents exécutés : {completed_agents}
- Résultats disponibles : {available_results}
- Temps écoulé : {elapsed_time}ms

CONTEXTE CONVERSATIONNEL :
{context_section}

HISTORIQUE DES AGENTS :
{agent_history}

CONTRAINTES ACTUELLES :
- Budget tokens restant : {token_budget}
- Timeout global : {global_timeout}ms
- Qualité minimum requise : {quality_threshold}

OBJECTIF : Déterminer la meilleure stratégie pour compléter efficacement la demande utilisateur.

Répondez avec votre décision d'orchestration au format JSON spécifié."""

# =============================================================================
# FONCTIONS DE FORMATAGE
# =============================================================================

def format_orchestrator_prompt(
    current_step: WorkflowStep,
    context: Dict[str, Any],
    user_message: str = "",
    agent_history: List[Dict[str, Any]] = None,
    constraints: Dict[str, Any] = None
) -> str:
    """
    Formate le prompt complet pour l'orchestrateur.
    
    Args:
        current_step: Étape actuelle du workflow
        context: Contexte global de la conversation
        user_message: Message utilisateur original
        agent_history: Historique des exécutions d'agents
        constraints: Contraintes de temps/budget/qualité
        
    Returns:
        Prompt formaté prêt pour DeepSeek
        
    Example:
        >>> prompt = format_orchestrator_prompt(
        ...     WorkflowStep.INTENT_DETECTION,
        ...     {"conversation_id": "123"},
        ...     "Mes achats Amazon"
        ... )
    """
    if not isinstance(current_step, WorkflowStep):
        raise ValueError("current_step doit être un WorkflowStep")
    
    if not isinstance(context, dict):
        raise ValueError("context doit être un dictionnaire")
    
    # Extraction des informations du contexte
    completed_agents = context.get("completed_agents", [])
    available_results = list(context.get("results", {}).keys())
    elapsed_time = context.get("elapsed_time_ms", 0)
    conversation_context = context.get("conversation_history", "")
    
    # Formatage de l'historique des agents
    agent_history_formatted = format_agent_history(agent_history or [])
    
    # Formatage de la section contexte
    context_section = ""
    if conversation_context:
        if isinstance(conversation_context, str):
            context_section = conversation_context
        else:
            context_section = str(conversation_context)[:500] + "..."
    
    # Contraintes par défaut
    default_constraints = {
        "token_budget": 2000,
        "global_timeout": 5000,
        "quality_threshold": 0.8
    }
    
    if constraints:
        default_constraints.update(constraints)
    
    # Formatage du prompt
    prompt = ORCHESTRATOR_WORKFLOW_TEMPLATE.format(
        user_message=user_message or context.get("user_message", ""),
        current_step=current_step.value,
        completed_agents=", ".join(completed_agents) if completed_agents else "Aucun",
        available_results=", ".join(available_results) if available_results else "Aucun",
        elapsed_time=elapsed_time,
        context_section=context_section,
        agent_history=agent_history_formatted,
        token_budget=default_constraints["token_budget"],
        global_timeout=default_constraints["global_timeout"],
        quality_threshold=default_constraints["quality_threshold"]
    )
    return f"{prompt}\n\n{ORCHESTRATOR_DECISION_EXAMPLES}"

def format_agent_history(agent_history: List[Dict[str, Any]]) -> str:
    """
    Formate l'historique des exécutions d'agents pour le prompt.
    
    Args:
        agent_history: Liste des exécutions d'agents
        
    Returns:
        Historique formaté lisible
        
    Example:
        >>> history = [{"agent": "IntentAgent", "status": "success", "duration": 150}]
        >>> formatted = format_agent_history(history)
    """
    if not agent_history:
        return "Aucun agent exécuté pour l'instant."
    
    formatted_entries = []
    
    for entry in agent_history[-5:]:  # 5 dernières exécutions max
        agent_name = entry.get("agent_name", "Agent inconnu")
        status = entry.get("status", "unknown")
        duration = entry.get("duration_ms", 0)
        error_msg = entry.get("error_message", "")
        
        # Formatage du statut avec emoji
        status_emoji = {
            "success": "✅",
            "failed": "❌", 
            "timeout": "⏰",
            "running": "🔄",
            "pending": "⏳"
        }.get(status, "❓")
        
        entry_line = f"{status_emoji} {agent_name}: {status} ({duration}ms)"
        
        if error_msg and status == "failed":
            entry_line += f" - {error_msg[:50]}..."
        
        formatted_entries.append(entry_line)
    
    return "\n".join(formatted_entries)

def build_workflow_state(
    current_step: WorkflowStep,
    agent_results: Dict[str, Any],
    start_time: datetime,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Construit l'état complet du workflow pour l'orchestrateur.
    
    Args:
        current_step: Étape actuelle
        agent_results: Résultats des agents exécutés
        start_time: Heure de début du workflow
        metadata: Métadonnées additionnelles
        
    Returns:
        État formaté du workflow
        
    Example:
        >>> state = build_workflow_state(
        ...     WorkflowStep.SEARCH_GENERATION,
        ...     {"intent_agent": {"intent": "transaction_query"}},
        ...     datetime.now()
        ... )
    """
    current_time = datetime.now()
    elapsed_ms = int((current_time - start_time).total_seconds() * 1000)
    
    # Construction de l'état de base
    workflow_state = {
        "current_step": current_step.value,
        "elapsed_time_ms": elapsed_ms,
        "completed_agents": list(agent_results.keys()),
        "results": agent_results.copy(),
        "status": "running" if current_step != WorkflowStep.COMPLETE else "complete"
    }
    
    # Ajout des métadonnées
    if metadata:
        workflow_state.update(metadata)
    
    # Calcul des métriques de performance
    agent_count = len(agent_results)
    avg_time_per_agent = elapsed_ms / max(agent_count, 1)
    
    workflow_state["performance_metrics"] = {
        "agents_executed": agent_count,
        "average_time_per_agent": avg_time_per_agent,
        "total_elapsed": elapsed_ms,
        "estimated_completion": _estimate_completion_time(current_step, elapsed_ms)
    }
    
    # État de santé du workflow
    workflow_state["health_status"] = _assess_workflow_health(current_step, elapsed_ms, agent_results)
    
    return workflow_state

def parse_orchestrator_decision(response: str) -> Dict[str, Any]:
    """
    Parse la décision JSON de l'orchestrateur DeepSeek.
    
    Args:
        response: Réponse brute de DeepSeek
        
    Returns:
        Décision parsée et validée
        
    Raises:
        ValueError: Si la décision est invalide
        
    Example:
        >>> decision = parse_orchestrator_decision(json_response)
        >>> print(decision["next_step"])
    """
    if not response or not response.strip():
        raise ValueError("Réponse orchestrateur vide")
    
    try:
        # Extraction du JSON depuis la réponse
        response_clean = response.strip()
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("Aucun JSON trouvé dans la réponse orchestrateur")
        
        json_str = response_clean[json_start:json_end]
        decision = json.loads(json_str)
        
        # Validation des champs obligatoires
        required_fields = ["workflow_type", "next_step", "reasoning"]
        for field in required_fields:
            if field not in decision:
                raise ValueError(f"Champ obligatoire manquant: {field}")
        
        # Validation des valeurs
        valid_workflows = ["standard", "fast_track", "parallel", "fallback"]
        if decision["workflow_type"] not in valid_workflows:
            logger.warning(f"Workflow type non standard: {decision['workflow_type']}")
            decision["workflow_type"] = "standard"  # Fallback
        
        valid_steps = [step.value for step in WorkflowStep]
        if decision["next_step"] not in valid_steps:
            raise ValueError(f"Étape invalide: {decision['next_step']}")
        
        # Valeurs par défaut
        if "parameters" not in decision:
            decision["parameters"] = {}
        
        default_params = {
            "timeout_ms": 5000,
            "max_retries": 2,
            "quality_threshold": 0.8
        }
        
        for key, value in default_params.items():
            if key not in decision["parameters"]:
                decision["parameters"][key] = value
        
        if "confidence" not in decision:
            decision["confidence"] = 0.7  # Confiance par défaut
        
        # Validation de la confiance
        confidence = decision["confidence"]
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            logger.warning(f"Confiance invalide: {confidence}")
            decision["confidence"] = 0.5
        
        return decision
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Erreur parsing décision orchestrateur: {e}")
        logger.error(f"Réponse brute: {response}")
        
        # Fallback gracieux - décision standard
        return {
            "workflow_type": "standard",
            "next_step": "intent_detection",
            "agent_assignments": {
                "primary_agent": "llm_intent_agent"
            },
            "parameters": {
                "timeout_ms": 5000,
                "max_retries": 2,
                "quality_threshold": 0.8
            },
            "reasoning": f"Décision fallback suite à erreur parsing: {str(e)}",
            "confidence": 0.3
        }

def _estimate_completion_time(current_step: WorkflowStep, elapsed_ms: int) -> int:
    """
    Estime le temps total de completion du workflow.
    
    Args:
        current_step: Étape actuelle
        elapsed_ms: Temps déjà écoulé
        
    Returns:
        Estimation du temps total en ms
    """
    # Temps moyen par étape (basé sur les performances observées)
    step_durations = {
        WorkflowStep.INIT: 10,
        WorkflowStep.INTENT_DETECTION: 150,  # Règles rapides ou IA
        WorkflowStep.SEARCH_GENERATION: 100,
        WorkflowStep.SEARCH_EXECUTION: 200,  # Dépend du Search Service
        WorkflowStep.RESPONSE_GENERATION: 250,
        WorkflowStep.VALIDATION: 50,
        WorkflowStep.COMPLETE: 0
    }
    
    # Étapes restantes
    all_steps = list(WorkflowStep)
    current_index = all_steps.index(current_step)
    remaining_steps = all_steps[current_index + 1:]
    
    # Calcul du temps restant estimé
    remaining_time = sum(step_durations.get(step, 100) for step in remaining_steps)
    
    return elapsed_ms + remaining_time

def _assess_workflow_health(
    current_step: WorkflowStep, 
    elapsed_ms: int, 
    agent_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Évalue la santé du workflow en cours.
    
    Args:
        current_step: Étape actuelle
        elapsed_ms: Temps écoulé
        agent_results: Résultats des agents
        
    Returns:
        Status de santé avec métriques
    """
    health_status = {
        "overall": "healthy",
        "warnings": [],
        "errors": [],
        "performance_score": 1.0
    }
    
    # Vérification du timing
    expected_time = _estimate_completion_time(WorkflowStep.INIT, 0)
    if elapsed_ms > expected_time * 1.5:
        health_status["warnings"].append("Workflow plus lent que prévu")
        health_status["performance_score"] *= 0.8
    
    if elapsed_ms > 5000:  # SLA de 5 secondes
        health_status["errors"].append("Dépassement du SLA global")
        health_status["overall"] = "degraded"
        health_status["performance_score"] *= 0.5
    
    # Vérification des échecs d'agents
    failed_agents = []
    for agent_name, result in agent_results.items():
        if isinstance(result, dict) and result.get("status") == "failed":
            failed_agents.append(agent_name)
    
    if failed_agents:
        health_status["warnings"].append(f"Agents en échec: {', '.join(failed_agents)}")
        health_status["performance_score"] *= 0.7
    
    # Vérification de la qualité des résultats
    low_quality_results = []
    for agent_name, result in agent_results.items():
        if isinstance(result, dict):
            confidence = result.get("confidence", 1.0)
            if confidence < 0.6:
                low_quality_results.append(agent_name)
    
    if low_quality_results:
        health_status["warnings"].append(f"Qualité faible: {', '.join(low_quality_results)}")
        health_status["performance_score"] *= 0.9
    
    # Détermination du status global
    if health_status["errors"]:
        health_status["overall"] = "critical"
    elif len(health_status["warnings"]) >= 2:
        health_status["overall"] = "degraded"
    elif health_status["warnings"]:
        health_status["overall"] = "warning"
    
    return health_status

def create_workflow_decision(
    workflow_type: str,
    next_step: WorkflowStep,
    primary_agent: str,
    reasoning: str,
    confidence: float = 0.8,
    **kwargs
) -> Dict[str, Any]:
    """
    Crée une décision d'orchestrateur structurée.
    
    Args:
        workflow_type: Type de workflow ("standard", "fast_track", etc.)
        next_step: Prochaine étape à exécuter
        primary_agent: Agent principal pour la prochaine étape
        reasoning: Justification de la décision
        confidence: Niveau de confiance (0.0-1.0)
        **kwargs: Paramètres additionnels
        
    Returns:
        Décision formatée prête à l'emploi
        
    Example:
        >>> decision = create_workflow_decision(
        ...     "standard",
        ...     WorkflowStep.INTENT_DETECTION,
        ...     "llm_intent_agent",
        ...     "Message complexe nécessitant détection IA"
        ... )
    """
    decision = {
        "workflow_type": workflow_type,
        "next_step": next_step.value if isinstance(next_step, WorkflowStep) else next_step,
        "agent_assignments": {
            "primary_agent": primary_agent
        },
        "parameters": {
            "timeout_ms": kwargs.get("timeout_ms", 5000),
            "max_retries": kwargs.get("max_retries", 2),
            "quality_threshold": kwargs.get("quality_threshold", 0.8)
        },
        "reasoning": reasoning,
        "confidence": max(0.0, min(1.0, confidence))
    }
    
    # Ajout des agents de fallback si spécifiés
    if "fallback_agent" in kwargs:
        decision["agent_assignments"]["fallback_agent"] = kwargs["fallback_agent"]
    
    if "parallel_agents" in kwargs:
        decision["agent_assignments"]["parallel_agents"] = kwargs["parallel_agents"]
    
    # Estimation du coût en tokens
    if "estimated_tokens" in kwargs:
        decision["estimated_cost"] = f"{kwargs['estimated_tokens']} tokens"
    
    return decision

def validate_workflow_constraints(
    decision: Dict[str, Any],
    constraints: Dict[str, Any]
) -> List[str]:
    """
    Valide qu'une décision respecte les contraintes du workflow.
    
    Args:
        decision: Décision d'orchestrateur à valider
        constraints: Contraintes à respecter
        
    Returns:
        Liste des violations de contraintes
        
    Example:
        >>> violations = validate_workflow_constraints(decision, {"max_timeout": 3000})
    """
    violations = []
    
    # Validation du timeout
    max_timeout = constraints.get("max_timeout_ms", 10000)
    decision_timeout = decision.get("parameters", {}).get("timeout_ms", 5000)
    
    if decision_timeout > max_timeout:
        violations.append(f"Timeout trop élevé: {decision_timeout}ms > {max_timeout}ms")
    
    # Validation du budget tokens
    max_tokens = constraints.get("max_tokens", 2000)
    estimated_cost = decision.get("estimated_cost", "0 tokens")
    
    if "tokens" in estimated_cost:
        try:
            tokens = int(estimated_cost.split()[0])
            if tokens > max_tokens:
                violations.append(f"Budget tokens dépassé: {tokens} > {max_tokens}")
        except (ValueError, IndexError):
            pass  # Ignore les erreurs de parsing
    
    # Validation de la qualité minimum
    min_quality = constraints.get("min_quality_threshold", 0.6)
    decision_quality = decision.get("parameters", {}).get("quality_threshold", 0.8)
    
    if decision_quality < min_quality:
        violations.append(f"Seuil qualité trop bas: {decision_quality} < {min_quality}")
    
    # Validation des agents disponibles
    available_agents = constraints.get("available_agents", [])
    if available_agents:
        primary_agent = decision.get("agent_assignments", {}).get("primary_agent")
        if primary_agent and primary_agent not in available_agents:
            violations.append(f"Agent non disponible: {primary_agent}")
    
    return violations

# =============================================================================
# EXEMPLES DE DÉCISIONS ORCHESTRATEUR
# =============================================================================

ORCHESTRATOR_DECISION_EXAMPLES = load_yaml_examples(
    'orchestrator_agent_examples.yaml', 'EXEMPLES DE DÉCISIONS ORCHESTRATEUR :'
)

# =============================================================================
# CONSTANTES UTILES
# =============================================================================

WORKFLOW_TYPES = {
    "standard": "Workflow complet avec tous les agents",
    "fast_track": "Raccourci pour requêtes simples",
    "parallel": "Exécution parallèle pour requêtes complexes",
    "fallback": "Mode dégradé après échecs"
}

AGENT_CAPABILITIES = {
    "llm_intent_agent": {
        "speciality": "Intent detection + entity extraction",
        "avg_duration_ms": 150,
        "success_rate": 0.95,
        "cost_tokens": 100
    },
    "search_query_agent": {
        "speciality": "Search query generation",
        "avg_duration_ms": 100,
        "success_rate": 0.90,
        "cost_tokens": 80
    },
    "response_agent": {
        "speciality": "Response generation",
        "avg_duration_ms": 250,
        "success_rate": 0.98,
        "cost_tokens": 200
    }
}

PERFORMANCE_THRESHOLDS = {
    "excellent": {"total_time_ms": 300, "token_usage": 150},
    "good": {"total_time_ms": 500, "token_usage": 300},
    "acceptable": {"total_time_ms": 1000, "token_usage": 500},
    "poor": {"total_time_ms": 2000, "token_usage": 1000}
}

# Export des éléments principaux
__all__ = [
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "ORCHESTRATOR_WORKFLOW_TEMPLATE",
    "ORCHESTRATOR_DECISION_EXAMPLES",
    "WorkflowStep",
    "AgentStatus",
    "format_orchestrator_prompt",
    "format_agent_history",
    "build_workflow_state",
    "parse_orchestrator_decision",
    "create_workflow_decision",
    "validate_workflow_constraints",
    "WORKFLOW_TYPES",
    "AGENT_CAPABILITIES",
    "PERFORMANCE_THRESHOLDS"
]