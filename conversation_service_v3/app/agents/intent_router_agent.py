"""
Intent Router Agent - Détermine si une recherche financière est nécessaire
"""
import logging
import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..models import UserQuery, AgentResponse, AgentRole
from ..models.intent import IntentCategory, IntentClassification
from ..core.llm_factory import create_llm_from_settings
from ..config.settings import settings

logger = logging.getLogger(__name__)


class IntentRouterAgent:
    """
    Agent de routage des intentions

    Responsabilités:
    - Classifier l'intention de l'utilisateur
    - Déterminer si une recherche financière est nécessaire
    - Générer des réponses conversationnelles directes
    """

    def __init__(self, llm_model: str = None, temperature: float = 0.1):
        # Use factory to create LLM (supports both OpenAI and DeepSeek)
        self.llm = create_llm_from_settings(
            settings,
            model=llm_model,  # If None, uses settings.LLM_MODEL
            temperature=temperature
        )
        self.parser = JsonOutputParser()

        # Définition du persona
        self.persona = """Je suis Harena, votre assistant financier personnel intelligent.

**Mes capacités:**
- 📊 Analyser vos transactions bancaires
- 💰 Calculer vos dépenses par catégorie, marchand ou période
- 📈 Générer des statistiques et insights sur vos finances
- 🔍 Rechercher des transactions spécifiques
- 💡 Vous aider à mieux comprendre vos habitudes de dépenses

**Ce que je peux faire:**
- "Combien j'ai dépensé en courses ce mois-ci ?"
- "Montre-moi mes transactions chez Carrefour"
- "Quelle est ma plus grosse dépense en loisirs ?"
- "Analyse mes dépenses de la semaine dernière"

**Architecture:**
- Version 3.0 avec agents LangChain autonomes
- Auto-correction des requêtes
- Compréhension en langage naturel"""

        # Prompt de classification
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en classification d'intentions pour assistant financier.

**Catégories:**

1. RECHERCHE FINANCIÈRE STANDARD → requires_search: true
   - FINANCIAL_QUERY: Recherche simple de transactions
   - FINANCIAL_ANALYSIS: Analyse basique de dépenses
   - FINANCIAL_STATS: Statistiques simples

2. ANALYSES AVANCÉES → requires_search: true
   - COMPARATIVE_ANALYSIS: Comparaisons (périodes, catégories)
     * "compare janvier vs février"
     * "charges fixes vs variables"

   - TREND_ANALYSIS: Analyse de tendances temporelles
     * "quelle est la tendance de mes revenus"
     * "évolution de mes dépenses cette année"

   - PREDICTIVE_ANALYSIS: Prévisions budgétaires
     * "prévois mon budget pour le mois prochain"
     * "estime mes dépenses futures"

   - OPTIMIZATION_RECOMMENDATION: Recommandations d'optimisation
     * "quel type de dépenses réduire"
     * "comment augmenter mon épargne"

   - BUDGET_ANALYSIS: Analyse budgétaire globale
     * "analyse mon budget des 3 derniers mois"
     * "quel est mon taux d'épargne"

3. CONVERSATIONNELLES → requires_search: false
   - GREETING/FAREWELL/GRATITUDE: Salutations, remerciements
   - SERVICE_INFO/CAPABILITY_QUERY/HELP_REQUEST: Questions sur Harena

4. HORS SCOPE → requires_search: false
   - SMALL_TALK/OUT_OF_SCOPE/UNCLEAR: Discussion générale, hors domaine

**Exemples de classification:**

"compare mes dépenses de janvier à février"
→ COMPARATIVE_ANALYSIS (comparaison temporelle)

"quelle est la tendance de mes revenus cette année"
→ TREND_ANALYSIS (analyse de tendance)

"prévois mon budget pour le mois prochain"
→ PREDICTIVE_ANALYSIS (prévision)

"quel type de dépenses je peux réduire pour augmenter mon épargne"
→ OPTIMIZATION_RECOMMENDATION (recommandation d'optimisation)

"quel est mon taux d'épargne"
→ BUDGET_ANALYSIS (analyse budgétaire globale)

"compare mes charges fixes à mes charges variables"
→ COMPARATIVE_ANALYSIS (comparaison de catégories)

"combien j'ai dépensé en courses ce mois"
→ FINANCIAL_QUERY (recherche simple)

**Format JSON:**
{{
  "category": "COMPARATIVE_ANALYSIS",
  "confidence": 0.95,
  "requires_search": true,
  "reasoning": "Demande de comparaison de deux périodes",
  "suggested_response": null
}}

**IMPORTANT:**
- requires_search: true pour FINANCIAL_* et toutes les analyses avancées
- Privilégier catégories analytiques pour questions complexes
- Fournir suggested_response pour intentions conversationnelles uniquement
"""),
            ("user", "Message: {user_message}\nContexte: {context}")
        ])

        self.chain = self.prompt | self.llm | self.parser

        self.stats = {
            "classifications": 0,
            "searches_triggered": 0,
            "conversational_responses": 0
        }

        logger.info(f"IntentRouterAgent initialized")

    async def classify_intent(self, user_query: UserQuery) -> AgentResponse:
        """Classifie l'intention de l'utilisateur"""
        try:
            logger.info(f"Classifying: {user_query.message[:100]}")

            context_str = ""
            if user_query.context:
                context_str = "\n".join([
                    f"{turn['role']}: {turn['content']}"
                    for turn in user_query.context[-3:]
                ])

            result = await self.chain.ainvoke({
                "user_message": user_query.message,
                "context": context_str or "Aucun"
            })

            category_str = result.get("category", "UNCLEAR")
            try:
                category = IntentCategory[category_str]
            except KeyError:
                category = IntentCategory.UNCLEAR

            classification = IntentClassification(
                category=category,
                confidence=result.get("confidence", 0.0),
                requires_search=result.get("requires_search", False),
                reasoning=result.get("reasoning", ""),
                suggested_response=result.get("suggested_response")
            )

            self.stats["classifications"] += 1
            if classification.requires_search:
                self.stats["searches_triggered"] += 1
            else:
                self.stats["conversational_responses"] += 1

            logger.info(
                f"Intent: {classification.category.value}, "
                f"search={classification.requires_search}"
            )

            return AgentResponse(
                success=True,
                data=classification,
                agent_role=AgentRole.QUERY_ANALYZER,
                metadata={"raw_result": result}
            )

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return AgentResponse(
                success=False,
                data=None,
                agent_role=AgentRole.QUERY_ANALYZER,
                error=str(e)
            )

    def get_persona_response(self, intent_category: IntentCategory) -> str:
        """Génère réponse persona pour intentions métacognitives"""

        if intent_category == IntentCategory.SERVICE_INFO:
            return self.persona

        elif intent_category == IntentCategory.CAPABILITY_QUERY:
            return """**Voici ce que je peux faire pour vous :**

📊 **Recherche de Transactions**
- Trouver des transactions par marchand, catégorie, montant ou date
- Exemple: "Montre-moi mes achats chez Carrefour"

💰 **Analyse de Dépenses**
- Calculer vos dépenses par catégorie ou période
- Exemple: "Combien j'ai dépensé en restaurants ce mois-ci ?"

📈 **Statistiques et Insights**
- Identifier votre plus grosse dépense, vos tendances
- Exemple: "Quelle est ma plus grosse dépense en loisirs ?"

🔍 **Recherche Avancée**
- Filtrer par montant, type d'opération, période
- Exemple: "Mes dépenses de plus de 100€ la semaine dernière"

N'hésitez pas à me poser vos questions en langage naturel !"""

        elif intent_category == IntentCategory.HELP_REQUEST:
            return """**Comment utiliser Harena :**

1. **Posez vos questions en langage naturel**
   - "Combien j'ai dépensé en courses ?"
   - "Montre-moi mes transactions Carrefour"

2. **Soyez spécifique si nécessaire**
   - Mentionnez périodes: "ce mois-ci", "la semaine dernière"
   - Précisez montants: "plus de 50€", "entre 10 et 100€"
   - Indiquez catégories: "alimentation", "loisirs", "transport"

3. **Demandez des analyses**
   - "Ma plus grosse dépense en loisirs ?"
   - "Évolution de mes dépenses alimentaires"

**Besoin d'aide ?** Essayez de me demander quelque chose !"""

        return "Comment puis-je vous aider avec vos finances ?"

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques de l'agent"""
        search_rate = 0.0
        if self.stats["classifications"] > 0:
            search_rate = self.stats["searches_triggered"] / self.stats["classifications"]

        return {
            "agent": "intent_router",
            "model": self.llm.model_name,
            **self.stats,
            "search_trigger_rate": search_rate
        }
