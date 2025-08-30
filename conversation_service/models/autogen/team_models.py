"""
Modèles Pydantic Phase 2 : AutoGen Team Models
Modèles spécialisés équipes multi-agents AutoGen
Orchestration, coordination, et métriques équipes
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, Literal, Set
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentRole(str, Enum):
    """Rôles agents dans équipe AutoGen"""
    INTENT_CLASSIFIER = "intent_classifier"
    ENTITY_EXTRACTOR = "entity_extractor"
    ORCHESTRATOR = "orchestrator"
    VALIDATOR = "validator"
    FALLBACK_HANDLER = "fallback_handler"


class AgentStatus(str, Enum):
    """Statuts agents"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    INITIALIZING = "initializing"


class WorkflowStage(str, Enum):
    """Étapes workflow équipe"""
    INITIALIZATION = "initialization"
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    CROSS_VALIDATION = "cross_validation"
    RESULT_SYNTHESIS = "result_synthesis"
    FINALIZATION = "finalization"
    ERROR_HANDLING = "error_handling"


class TeamCommunicationMode(str, Enum):
    """Modes communication équipe"""
    SEQUENTIAL = "sequential"          # Agents en séquence
    PARALLEL = "parallel"              # Agents en parallèle
    ROUND_ROBIN = "round_robin"        # Tour de rôle
    CONSENSUS_BASED = "consensus_based"  # Basé consensus
    HIERARCHICAL = "hierarchical"      # Hiérarchique


class AgentPerformanceMetrics(BaseModel):
    """Métriques performance agent individuel"""
    
    agent_id: str = Field(..., description="ID unique agent")
    agent_role: AgentRole = Field(..., description="Rôle agent")
    agent_name: str = Field(..., description="Nom agent")
    
    # Métriques temporelles
    average_response_time_ms: float = Field(default=0.0, description="Temps réponse moyen ms")
    last_response_time_ms: int = Field(default=0, description="Dernier temps réponse ms")
    total_processing_time_ms: int = Field(default=0, description="Temps traitement total ms")
    
    # Métriques qualité
    success_rate: float = Field(default=0.0, description="Taux succès", ge=0.0, le=1.0)
    average_confidence: float = Field(default=0.0, description="Confiance moyenne", ge=0.0, le=1.0)
    error_rate: float = Field(default=0.0, description="Taux erreur", ge=0.0, le=1.0)
    
    # Compteurs activité
    requests_processed: int = Field(default=0, description="Requêtes traitées")
    successful_requests: int = Field(default=0, description="Requêtes réussies")
    failed_requests: int = Field(default=0, description="Requêtes échouées")
    
    # Statut santé
    current_status: AgentStatus = Field(default=AgentStatus.HEALTHY)
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    consecutive_failures: int = Field(default=0, description="Échecs consécutifs")
    
    # Spécialisations par rôle
    role_specific_metrics: Dict[str, Any] = Field(default_factory=dict, description="Métriques spécialisées rôle")
    
    @model_validator(mode='after')
    def calculate_derived_metrics(cls, values):
        """Calcul métriques dérivées"""
        processed = values.get('requests_processed', 0)
        successful = values.get('successful_requests', 0)
        failed = values.get('failed_requests', 0)
        
        if processed > 0:
            values['success_rate'] = successful / processed
            values['error_rate'] = failed / processed
        
        # Mise à jour statut basé sur performance
        error_rate = values.get('error_rate', 0)
        consecutive_failures = values.get('consecutive_failures', 0)
        
        if consecutive_failures >= 5:
            values['current_status'] = AgentStatus.ERROR
        elif error_rate > 0.5:
            values['current_status'] = AgentStatus.DEGRADED
        elif error_rate > 0.8:
            values['current_status'] = AgentStatus.UNAVAILABLE
        else:
            values['current_status'] = AgentStatus.HEALTHY
            
        return values
    
    def update_performance(self, response_time_ms: int, success: bool, confidence: float = None):
        """Mise à jour métriques performance"""
        self.last_response_time_ms = response_time_ms
        self.total_processing_time_ms += response_time_ms
        self.requests_processed += 1
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
        
        # Recalcul moyenne temps réponse
        self.average_response_time_ms = self.total_processing_time_ms / self.requests_processed
        
        # Mise à jour confiance moyenne
        if confidence is not None:
            current_avg = self.average_confidence
            count = self.requests_processed
            self.average_confidence = ((current_avg * (count - 1)) + confidence) / count
    
    def is_healthy(self) -> bool:
        """Agent en bonne santé"""
        return (
            self.current_status in [AgentStatus.HEALTHY, AgentStatus.DEGRADED] and
            self.consecutive_failures < 3 and
            self.error_rate < 0.7
        )


class TeamWorkflowExecution(BaseModel):
    """Exécution workflow équipe AutoGen"""
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_message: str = Field(..., description="Message utilisateur traité")
    
    # Configuration workflow
    communication_mode: TeamCommunicationMode = Field(default=TeamCommunicationMode.SEQUENTIAL)
    max_iterations: int = Field(default=5, description="Itérations max", ge=1, le=20)
    timeout_seconds: int = Field(default=30, description="Timeout workflow", ge=5, le=300)
    
    # Progression workflow
    current_stage: WorkflowStage = Field(default=WorkflowStage.INITIALIZATION)
    completed_stages: List[WorkflowStage] = Field(default_factory=list)
    iteration_count: int = Field(default=0, description="Itérations actuelles")
    
    # Participants workflow
    active_agents: Set[str] = Field(default_factory=set, description="Agents actifs")
    agent_sequence: List[str] = Field(default_factory=list, description="Séquence agents")
    current_speaker: Optional[str] = Field(None, description="Agent parlant actuel")
    
    # Résultats intermédiaires
    stage_results: Dict[WorkflowStage, Dict[str, Any]] = Field(default_factory=dict)
    agent_contributions: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    consensus_votes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Statut exécution
    is_completed: bool = Field(default=False)
    is_successful: bool = Field(default=False)
    termination_reason: Optional[str] = Field(None)
    error_details: Optional[Dict[str, Any]] = Field(None)
    
    def start_stage(self, stage: WorkflowStage, agent_id: Optional[str] = None):
        """Démarrer nouvelle étape workflow"""
        self.current_stage = stage
        if agent_id:
            self.current_speaker = agent_id
            self.active_agents.add(agent_id)
    
    def complete_stage(self, stage: WorkflowStage, results: Dict[str, Any]):
        """Marquer étape comme complétée"""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.stage_results[stage] = results
    
    def add_agent_contribution(self, agent_id: str, contribution: Dict[str, Any]):
        """Ajouter contribution agent"""
        if agent_id not in self.agent_contributions:
            self.agent_contributions[agent_id] = []
        self.agent_contributions[agent_id].append(contribution)
    
    def record_consensus_vote(self, agent_id: str, vote_data: Dict[str, Any]):
        """Enregistrer vote consensus"""
        self.consensus_votes[agent_id] = vote_data
    
    def complete_workflow(self, success: bool, reason: str = None):
        """Marquer workflow comme terminé"""
        self.is_completed = True
        self.is_successful = success
        self.termination_reason = reason
        if not success and reason:
            self.error_details = {"reason": reason, "stage": self.current_stage}
    
    def get_workflow_duration(self) -> timedelta:
        """Durée exécution workflow"""
        return datetime.utcnow() - self.execution_timestamp


class TeamCoordinationContext(BaseModel):
    """Contexte coordination équipe multi-agents"""
    
    coordination_id: str = Field(default_factory=lambda: str(uuid4()))
    team_session_id: str = Field(..., description="ID session équipe")
    
    # État équipe
    team_composition: Dict[AgentRole, str] = Field(..., description="Composition équipe role->agent_id")
    team_leader: Optional[str] = Field(None, description="Agent leader équipe")
    backup_agents: Dict[AgentRole, List[str]] = Field(default_factory=dict, description="Agents backup")
    
    # Configuration coordination
    consensus_threshold: float = Field(default=0.7, description="Seuil consensus", ge=0.5, le=1.0)
    max_coordination_rounds: int = Field(default=3, description="Rounds coordination max", ge=1, le=10)
    enable_cross_validation: bool = Field(default=True, description="Validation croisée activée")
    
    # État coordination
    shared_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte partagé agents")
    coordination_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historique coordination")
    conflict_resolution_log: List[Dict[str, Any]] = Field(default_factory=list, description="Log résolution conflits")
    
    # Métriques coordination
    coordination_efficiency: float = Field(default=0.0, description="Efficacité coordination", ge=0.0, le=1.0)
    consensus_achievement_rate: float = Field(default=0.0, description="Taux atteinte consensus", ge=0.0, le=1.0)
    average_coordination_time_ms: float = Field(default=0.0, description="Temps coordination moyen ms")
    
    def update_shared_context(self, key: str, value: Any, agent_id: str):
        """Mise à jour contexte partagé"""
        if key not in self.shared_context:
            self.shared_context[key] = {}
        
        self.shared_context[key].update({
            "value": value,
            "updated_by": agent_id,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Log coordination
        self.coordination_history.append({
            "action": "context_update",
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_conflict_resolution(self, conflict_type: str, agents_involved: List[str], resolution: str):
        """Enregistrer résolution conflit"""
        self.conflict_resolution_log.append({
            "conflict_type": conflict_type,
            "agents_involved": agents_involved,
            "resolution": resolution,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_agent_by_role(self, role: AgentRole) -> Optional[str]:
        """Récupérer agent par rôle"""
        return self.team_composition.get(role)
    
    def has_healthy_agent_for_role(self, role: AgentRole, agent_performances: Dict[str, AgentPerformanceMetrics]) -> bool:
        """Vérifier agent sain disponible pour rôle"""
        agent_id = self.get_agent_by_role(role)
        if not agent_id:
            return False
        
        agent_perf = agent_performances.get(agent_id)
        return agent_perf and agent_perf.is_healthy()


class MultiAgentTeamState(BaseModel):
    """État global équipe multi-agents AutoGen"""
    
    team_id: str = Field(default_factory=lambda: str(uuid4()))
    team_name: str = Field(..., description="Nom équipe")
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # État agents
    agent_performances: Dict[str, AgentPerformanceMetrics] = Field(default_factory=dict)
    team_coordination: TeamCoordinationContext = Field(..., description="Contexte coordination")
    active_workflows: Dict[str, TeamWorkflowExecution] = Field(default_factory=dict)
    
    # Configuration équipe
    default_communication_mode: TeamCommunicationMode = Field(default=TeamCommunicationMode.ROUND_ROBIN)
    auto_scaling_enabled: bool = Field(default=False, description="Scaling automatique")
    fallback_enabled: bool = Field(default=True, description="Fallback activé")
    
    # Métriques équipe globales
    total_requests_processed: int = Field(default=0, description="Requêtes totales traitées")
    successful_team_executions: int = Field(default=0, description="Exécutions équipe réussies")
    failed_team_executions: int = Field(default=0, description="Exécutions équipe échouées")
    average_team_response_time_ms: float = Field(default=0.0, description="Temps réponse équipe moyen ms")
    
    # État santé équipe
    team_health_score: float = Field(default=1.0, description="Score santé équipe", ge=0.0, le=1.0)
    last_health_assessment: datetime = Field(default_factory=datetime.utcnow)
    critical_issues: List[str] = Field(default_factory=list, description="Issues critiques")
    
    @model_validator(mode='after') 
    def initialize_team_coordination(cls, values):
        """Initialiser coordination si pas fournie"""
        if 'team_coordination' not in values:
            team_id = values.get('team_id', str(uuid4()))
            values['team_coordination'] = TeamCoordinationContext(
                team_session_id=team_id,
                team_composition={}
            )
        return values
    
    def add_agent(self, agent_id: str, role: AgentRole, agent_name: str):
        """Ajouter agent à l'équipe"""
        # Créer métriques performance agent
        self.agent_performances[agent_id] = AgentPerformanceMetrics(
            agent_id=agent_id,
            agent_role=role,
            agent_name=agent_name
        )
        
        # Ajouter à composition équipe
        self.team_coordination.team_composition[role] = agent_id
    
    def start_workflow(self, workflow: TeamWorkflowExecution):
        """Démarrer nouveau workflow"""
        self.active_workflows[workflow.execution_id] = workflow
    
    def complete_workflow(self, execution_id: str, success: bool):
        """Marquer workflow comme terminé"""
        if execution_id in self.active_workflows:
            workflow = self.active_workflows[execution_id]
            workflow.complete_workflow(success)
            
            # Mise à jour métriques équipe
            self.total_requests_processed += 1
            if success:
                self.successful_team_executions += 1
            else:
                self.failed_team_executions += 1
            
            # Mise à jour temps réponse moyen
            workflow_duration = workflow.get_workflow_duration()
            duration_ms = workflow_duration.total_seconds() * 1000
            
            total_requests = self.total_requests_processed
            current_avg = self.average_team_response_time_ms
            self.average_team_response_time_ms = ((current_avg * (total_requests - 1)) + duration_ms) / total_requests
    
    def assess_team_health(self) -> float:
        """Évaluer santé équipe"""
        if not self.agent_performances:
            return 0.0
        
        # Santé agents individuels
        healthy_agents = sum(1 for perf in self.agent_performances.values() if perf.is_healthy())
        total_agents = len(self.agent_performances)
        agent_health_ratio = healthy_agents / total_agents if total_agents > 0 else 0
        
        # Performance équipe
        team_success_rate = 0.0
        if self.total_requests_processed > 0:
            team_success_rate = self.successful_team_executions / self.total_requests_processed
        
        # Score combiné
        health_score = (agent_health_ratio * 0.6) + (team_success_rate * 0.4)
        
        self.team_health_score = health_score
        self.last_health_assessment = datetime.utcnow()
        
        # Issues critiques
        self.critical_issues.clear()
        unhealthy_agents = [agent_id for agent_id, perf in self.agent_performances.items() 
                          if not perf.is_healthy()]
        if unhealthy_agents:
            self.critical_issues.append(f"Agents unhealthy: {', '.join(unhealthy_agents)}")
        
        if team_success_rate < 0.5:
            self.critical_issues.append(f"Low team success rate: {team_success_rate:.2%}")
        
        return health_score
    
    def is_team_operational(self) -> bool:
        """Équipe opérationnelle"""
        return (
            self.team_health_score >= 0.5 and
            len(self.critical_issues) == 0 and
            len([p for p in self.agent_performances.values() if p.is_healthy()]) >= 2
        )
    
    def get_team_summary(self) -> Dict[str, Any]:
        """Résumé état équipe"""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "agents_count": len(self.agent_performances),
            "healthy_agents": len([p for p in self.agent_performances.values() if p.is_healthy()]),
            "active_workflows": len(self.active_workflows),
            "team_health_score": self.team_health_score,
            "is_operational": self.is_team_operational(),
            "success_rate": (self.successful_team_executions / self.total_requests_processed 
                           if self.total_requests_processed > 0 else 0),
            "average_response_time_ms": self.average_team_response_time_ms,
            "critical_issues_count": len(self.critical_issues)
        }


# Factory Functions

def create_financial_team_state(team_name: str = "Financial Multi-Agent Team") -> MultiAgentTeamState:
    """Factory création état équipe financière"""
    
    team_state = MultiAgentTeamState(
        team_name=team_name,
        team_coordination=TeamCoordinationContext(
            team_session_id=str(uuid4()),
            team_composition={},
            consensus_threshold=0.75,
            enable_cross_validation=True
        )
    )
    
    # Ajouter agents standards
    team_state.add_agent("intent_classifier_001", AgentRole.INTENT_CLASSIFIER, "Financial Intent Classifier")
    team_state.add_agent("entity_extractor_001", AgentRole.ENTITY_EXTRACTOR, "Financial Entity Extractor")
    team_state.add_agent("orchestrator_001", AgentRole.ORCHESTRATOR, "Team Orchestrator")
    
    return team_state


def create_workflow_execution(
    user_message: str, 
    communication_mode: TeamCommunicationMode = TeamCommunicationMode.ROUND_ROBIN
) -> TeamWorkflowExecution:
    """Factory création exécution workflow"""
    
    return TeamWorkflowExecution(
        user_message=user_message,
        communication_mode=communication_mode,
        agent_sequence=["intent_classifier_001", "entity_extractor_001"],
        max_iterations=5,
        timeout_seconds=30
    )