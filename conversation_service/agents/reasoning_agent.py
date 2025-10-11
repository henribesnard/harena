"""
Reasoning Agent - Complex Query Decomposition
Architecture v3.0 - Phase 2

Responsabilité: Décomposition et orchestration de questions complexes multi-étapes
- Analyse intention complexe et décomposition en sous-tâches
- Plan d'exécution séquentiel ou parallèle
- Gestion état inter-étapes
- Chain-of-Thought reasoning avec LLM
- Composition résultats multi-requêtes
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types de tâches supportées"""
    QUERY_SEARCH = "query_search"  # Requête search_service
    COMPUTE_METRIC = "compute_metric"  # Calcul métrique (Analytics Agent)
    COMPARE_PERIODS = "compare_periods"  # Comparaison temporelle
    AGGREGATE_RESULTS = "aggregate_results"  # Agrégation résultats
    GENERATE_INSIGHT = "generate_insight"  # Génération insight


class ExecutionMode(Enum):
    """Modes d'exécution des tâches"""
    SEQUENTIAL = "sequential"  # Une par une, ordre strict
    PARALLEL = "parallel"  # Toutes en parallèle
    MIXED = "mixed"  # Certaines parallèles, d'autres séquentielles


@dataclass
class ReasoningTask:
    """Tâche individuelle dans le plan d'exécution"""
    task_id: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # IDs tâches dépendantes
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Plan d'exécution complet pour une question complexe"""
    plan_id: str
    user_question: str
    reasoning: str  # Explication du raisonnement
    tasks: List[ReasoningTask]
    execution_mode: ExecutionMode
    estimated_time_seconds: int


@dataclass
class ReasoningResult:
    """Résultat final du raisonnement"""
    success: bool
    plan: ExecutionPlan
    final_answer: str
    intermediate_results: Dict[str, Any]
    total_execution_time_ms: int
    tasks_completed: int
    tasks_failed: int


class ReasoningAgent:
    """
    Agent de raisonnement pour questions complexes

    Utilise Chain-of-Thought avec LLM pour:
    1. Analyser question complexe
    2. Décomposer en sous-tâches exécutables
    3. Orchestrer exécution (séquentiel/parallèle)
    4. Composer résultat final

    Exemple:
    Question: "Compare mes dépenses alimentaires ce mois vs mois dernier"

    Plan généré:
    1. [PARALLEL] Récupérer transactions alimentaires mois actuel
    2. [PARALLEL] Récupérer transactions alimentaires mois précédent
    3. [SEQUENTIAL] Comparer les deux périodes (dépend de 1 et 2)
    4. [SEQUENTIAL] Générer insights de comparaison
    """

    def __init__(
        self,
        llm_manager,  # LLMProviderManager
        analytics_agent,  # AnalyticsAgent
        query_executor  # QueryExecutor
    ):
        self.llm_manager = llm_manager
        self.analytics_agent = analytics_agent
        self.query_executor = query_executor

        # Statistiques
        self.stats = {
            "plans_generated": 0,
            "tasks_executed": 0,
            "complex_queries_solved": 0,
            "avg_tasks_per_plan": 0.0
        }

        logger.info("ReasoningAgent initialized")

    async def reason_and_execute(
        self,
        user_question: str,
        user_id: int,
        context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """
        Analyse question complexe, génère plan, et exécute

        Args:
            user_question: Question utilisateur complexe
            user_id: ID utilisateur
            context: Contexte additionnel (historique, profil)

        Returns:
            ReasoningResult avec réponse finale et résultats intermédiaires
        """
        start_time = datetime.now()

        try:
            # ÉTAPE 1: Générer plan d'exécution avec LLM
            logger.info(f"Generating execution plan for: {user_question}")
            plan = await self._generate_execution_plan(user_question, context)

            if not plan.tasks:
                return ReasoningResult(
                    success=False,
                    plan=plan,
                    final_answer="Je n'arrive pas à décomposer cette question en étapes exécutables.",
                    intermediate_results={},
                    total_execution_time_ms=self._get_elapsed_ms(start_time),
                    tasks_completed=0,
                    tasks_failed=0
                )

            # ÉTAPE 2: Exécuter le plan
            logger.info(f"Executing plan {plan.plan_id} with {len(plan.tasks)} tasks")
            execution_result = await self._execute_plan(plan, user_id)

            # ÉTAPE 3: Composer réponse finale
            final_answer = await self._compose_final_answer(
                user_question, plan, execution_result
            )

            # Mise à jour statistiques
            self.stats["plans_generated"] += 1
            self.stats["tasks_executed"] += execution_result["tasks_completed"]
            self.stats["complex_queries_solved"] += 1

            # Moyenne mobile tâches par plan
            current_avg = self.stats["avg_tasks_per_plan"]
            total_plans = self.stats["plans_generated"]
            self.stats["avg_tasks_per_plan"] = (
                (current_avg * (total_plans - 1) + len(plan.tasks)) / total_plans
            )

            return ReasoningResult(
                success=execution_result["success"],
                plan=plan,
                final_answer=final_answer,
                intermediate_results=execution_result["results"],
                total_execution_time_ms=self._get_elapsed_ms(start_time),
                tasks_completed=execution_result["tasks_completed"],
                tasks_failed=execution_result["tasks_failed"]
            )

        except Exception as e:
            logger.error(f"Error in reasoning: {str(e)}")
            return ReasoningResult(
                success=False,
                plan=ExecutionPlan(
                    plan_id="error",
                    user_question=user_question,
                    reasoning="Error occurred",
                    tasks=[],
                    execution_mode=ExecutionMode.SEQUENTIAL,
                    estimated_time_seconds=0
                ),
                final_answer=f"Une erreur est survenue lors du raisonnement: {str(e)}",
                intermediate_results={},
                total_execution_time_ms=self._get_elapsed_ms(start_time),
                tasks_completed=0,
                tasks_failed=1
            )

    async def _generate_execution_plan(
        self,
        user_question: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Génère un plan d'exécution avec LLM (Chain-of-Thought)
        """

        system_prompt = """Tu es un expert en raisonnement et décomposition de problèmes financiers complexes.

Ton rôle est d'analyser une question utilisateur complexe et de la décomposer en étapes exécutables.

TYPES DE TÂCHES DISPONIBLES:
1. query_search: Rechercher des transactions avec filtres
2. compute_metric: Calculer une métrique (total, moyenne, etc.)
3. compare_periods: Comparer deux périodes temporelles
4. aggregate_results: Agréger plusieurs résultats
5. generate_insight: Générer un insight

RÈGLES:
- Identifier les sous-questions implicites
- Décomposer en étapes simples et exécutables
- Spécifier les dépendances entre tâches
- Indiquer si tâches peuvent être parallélisées
- Être précis sur les paramètres de chaque tâche

FORMAT DE RÉPONSE (JSON strict):
{
  "reasoning": "Explication du raisonnement étape par étape",
  "tasks": [
    {
      "task_id": "task_1",
      "task_type": "query_search",
      "description": "Description claire",
      "parameters": {
        "filters": {...},
        "period": "..."
      },
      "dependencies": []
    },
    ...
  ],
  "execution_mode": "parallel" ou "sequential" ou "mixed"
}

EXEMPLES:

Question: "Compare mes dépenses alimentaires ce mois vs mois dernier"
Raisonnement:
1. Besoin de 2 requêtes parallèles (ce mois + mois dernier)
2. Puis comparaison séquentielle des résultats
3. Génération insight final

Plan:
- task_1: Récupérer transactions alimentaires mois actuel [PARALLEL]
- task_2: Récupérer transactions alimentaires mois précédent [PARALLEL]
- task_3: Comparer les périodes (dépend task_1, task_2) [SEQUENTIAL]
- task_4: Générer insight comparaison [SEQUENTIAL]
"""

        user_prompt = f"""QUESTION UTILISATEUR: "{user_question}"

CONTEXTE:
{json.dumps(context, ensure_ascii=False, indent=2) if context else "Aucun contexte"}

GÉNÈRE UN PLAN D'EXÉCUTION:"""

        try:
            # Appel LLM pour génération du plan
            from .llm import LLMRequest

            llm_request = LLMRequest(
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
                system_prompt=system_prompt,
                temperature=0.1,  # Faible pour raisonnement déterministe
                max_tokens=2000,
                response_format={"type": "json_object"},
                user_id=0,
                conversation_id=None
            )

            llm_response = await self.llm_manager.generate(llm_request)

            if llm_response.error:
                logger.error(f"LLM error generating plan: {llm_response.error}")
                return self._fallback_simple_plan(user_question)

            # Parse JSON response
            plan_data = json.loads(llm_response.content)

            # Construction du plan
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = ReasoningTask(
                    task_id=task_data["task_id"],
                    task_type=TaskType(task_data["task_type"]),
                    description=task_data["description"],
                    parameters=task_data.get("parameters", {}),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)

            execution_mode = ExecutionMode(
                plan_data.get("execution_mode", "sequential")
            )

            plan = ExecutionPlan(
                plan_id=f"plan_{datetime.now().timestamp()}",
                user_question=user_question,
                reasoning=plan_data.get("reasoning", "Plan generated"),
                tasks=tasks,
                execution_mode=execution_mode,
                estimated_time_seconds=len(tasks) * 2  # Estimation simple
            )

            logger.info(f"Generated plan with {len(tasks)} tasks ({execution_mode.value} mode)")

            return plan

        except Exception as e:
            logger.error(f"Error generating plan: {str(e)}")
            return self._fallback_simple_plan(user_question)

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Exécute le plan avec gestion dépendances et parallélisation
        """

        results = {}
        tasks_completed = 0
        tasks_failed = 0

        try:
            if plan.execution_mode == ExecutionMode.SEQUENTIAL:
                # Exécution séquentielle stricte
                for task in plan.tasks:
                    result = await self._execute_task(task, results, user_id)
                    if result["success"]:
                        tasks_completed += 1
                        results[task.task_id] = result["data"]
                        task.status = "completed"
                        task.result = result["data"]
                    else:
                        tasks_failed += 1
                        task.status = "failed"
                        task.error = result["error"]

            elif plan.execution_mode == ExecutionMode.PARALLEL:
                # Exécution parallèle (pas de dépendances)
                parallel_tasks = [
                    self._execute_task(task, results, user_id)
                    for task in plan.tasks
                ]
                parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

                for i, task in enumerate(plan.tasks):
                    result = parallel_results[i]
                    if isinstance(result, Exception):
                        tasks_failed += 1
                        task.status = "failed"
                        task.error = str(result)
                    elif result["success"]:
                        tasks_completed += 1
                        results[task.task_id] = result["data"]
                        task.status = "completed"
                        task.result = result["data"]
                    else:
                        tasks_failed += 1
                        task.status = "failed"
                        task.error = result["error"]

            else:  # MIXED mode
                # Exécuter par vagues selon dépendances
                executed_tasks = set()

                while len(executed_tasks) < len(plan.tasks):
                    # Trouver tâches exécutables (dépendances satisfaites)
                    ready_tasks = [
                        task for task in plan.tasks
                        if task.task_id not in executed_tasks
                        and all(dep in executed_tasks for dep in task.dependencies)
                    ]

                    if not ready_tasks:
                        logger.warning("No tasks ready to execute (circular dependency?)")
                        break

                    # Exécuter la vague en parallèle
                    wave_tasks = [
                        self._execute_task(task, results, user_id)
                        for task in ready_tasks
                    ]
                    wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

                    for i, task in enumerate(ready_tasks):
                        executed_tasks.add(task.task_id)
                        result = wave_results[i]

                        if isinstance(result, Exception):
                            tasks_failed += 1
                            task.status = "failed"
                            task.error = str(result)
                        elif result["success"]:
                            tasks_completed += 1
                            results[task.task_id] = result["data"]
                            task.status = "completed"
                            task.result = result["data"]
                        else:
                            tasks_failed += 1
                            task.status = "failed"
                            task.error = result["error"]

            return {
                "success": tasks_failed == 0,
                "results": results,
                "tasks_completed": tasks_completed,
                "tasks_failed": tasks_failed
            }

        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}")
            return {
                "success": False,
                "results": results,
                "tasks_completed": tasks_completed,
                "tasks_failed": len(plan.tasks) - tasks_completed
            }

    async def _execute_task(
        self,
        task: ReasoningTask,
        previous_results: Dict[str, Any],
        user_id: int
    ) -> Dict[str, Any]:
        """
        Exécute une tâche individuelle selon son type
        """

        logger.info(f"Executing task {task.task_id}: {task.description}")

        try:
            task.status = "running"

            if task.task_type == TaskType.QUERY_SEARCH:
                # Exécuter requête search_service
                result = await self._execute_query_search(task, user_id)

            elif task.task_type == TaskType.COMPUTE_METRIC:
                # Calculer métrique avec Analytics Agent
                result = await self._execute_compute_metric(task, previous_results)

            elif task.task_type == TaskType.COMPARE_PERIODS:
                # Comparer deux périodes
                result = await self._execute_compare_periods(task, previous_results)

            elif task.task_type == TaskType.AGGREGATE_RESULTS:
                # Agréger résultats
                result = await self._execute_aggregate(task, previous_results)

            elif task.task_type == TaskType.GENERATE_INSIGHT:
                # Générer insight
                result = await self._execute_generate_insight(task, previous_results)

            else:
                result = {
                    "success": False,
                    "error": f"Unknown task type: {task.task_type}"
                }

            return result

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_query_search(
        self,
        task: ReasoningTask,
        user_id: int
    ) -> Dict[str, Any]:
        """Exécute une requête de recherche"""

        # Utiliser query_executor pour rechercher
        # (implémentation simplifiée, à adapter selon votre QueryExecutor)

        return {
            "success": True,
            "data": {
                "transactions": [],  # Résultats de la requête
                "total_hits": 0
            }
        }

    async def _execute_compute_metric(
        self,
        task: ReasoningTask,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcule une métrique sur des résultats précédents"""

        # Récupérer données des tâches dépendantes
        transactions = []
        for dep_id in task.dependencies:
            if dep_id in previous_results:
                transactions.extend(previous_results[dep_id].get("transactions", []))

        # Calculer métrique demandée
        metric_type = task.parameters.get("metric", "sum")

        if metric_type == "sum":
            total = sum(abs(float(tx.get("amount", 0))) for tx in transactions)
            return {"success": True, "data": {"total": total}}

        elif metric_type == "average":
            if not transactions:
                return {"success": True, "data": {"average": 0}}
            total = sum(abs(float(tx.get("amount", 0))) for tx in transactions)
            return {"success": True, "data": {"average": total / len(transactions)}}

        return {"success": False, "error": f"Unknown metric: {metric_type}"}

    async def _execute_compare_periods(
        self,
        task: ReasoningTask,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare deux périodes avec Analytics Agent"""

        # Récupérer transactions des deux périodes
        period_1_id = task.parameters.get("period_1_task_id")
        period_2_id = task.parameters.get("period_2_task_id")

        transactions_1 = previous_results.get(period_1_id, {}).get("transactions", [])
        transactions_2 = previous_results.get(period_2_id, {}).get("transactions", [])

        # Comparer avec Analytics Agent
        from .analytics_agent import ComparisonPeriod

        comparison = await self.analytics_agent.compare_periods(
            current_transactions=transactions_1,
            previous_transactions=transactions_2,
            comparison_type=ComparisonPeriod.MONTH_OVER_MONTH,
            current_label="Période actuelle",
            previous_label="Période précédente"
        )

        return {
            "success": True,
            "data": {
                "comparison": comparison.__dict__
            }
        }

    async def _execute_aggregate(
        self,
        task: ReasoningTask,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Agrège plusieurs résultats"""

        aggregated = {}

        for dep_id in task.dependencies:
            if dep_id in previous_results:
                aggregated[dep_id] = previous_results[dep_id]

        return {
            "success": True,
            "data": aggregated
        }

    async def _execute_generate_insight(
        self,
        task: ReasoningTask,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère un insight basé sur les résultats"""

        # Composer insight à partir des résultats
        insight = {
            "title": task.description,
            "data": previous_results
        }

        return {
            "success": True,
            "data": insight
        }

    async def _compose_final_answer(
        self,
        user_question: str,
        plan: ExecutionPlan,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Compose la réponse finale en langage naturel avec LLM
        """

        # Utiliser LLM pour générer réponse naturelle
        system_prompt = "Tu es un assistant financier qui explique des résultats d'analyse en langage simple et clair."

        user_prompt = f"""QUESTION UTILISATEUR: "{user_question}"

PLAN EXÉCUTÉ:
{plan.reasoning}

RÉSULTATS DES TÂCHES:
{json.dumps(execution_result["results"], ensure_ascii=False, indent=2, default=str)}

GÉNÈRE UNE RÉPONSE NATURELLE ET CLAIRE:"""

        try:
            from .llm import LLMRequest

            llm_request = LLMRequest(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=500,
                user_id=0,
                conversation_id=None
            )

            llm_response = await self.llm_manager.generate(llm_request)

            if llm_response.error:
                return "J'ai exécuté votre demande mais j'ai des difficultés à formuler la réponse."

            return llm_response.content

        except Exception as e:
            logger.error(f"Error composing final answer: {str(e)}")
            return f"Votre demande a été traitée mais je rencontre une erreur de génération de réponse."

    def _fallback_simple_plan(self, user_question: str) -> ExecutionPlan:
        """Plan de fallback si génération LLM échoue"""

        simple_task = ReasoningTask(
            task_id="fallback_task",
            task_type=TaskType.QUERY_SEARCH,
            description="Recherche simple",
            parameters={}
        )

        return ExecutionPlan(
            plan_id="fallback_plan",
            user_question=user_question,
            reasoning="Fallback to simple query",
            tasks=[simple_task],
            execution_mode=ExecutionMode.SEQUENTIAL,
            estimated_time_seconds=5
        )

    def _get_elapsed_ms(self, start_time: datetime) -> int:
        """Calcule temps écoulé en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)

    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de l'agent"""
        return self.stats
