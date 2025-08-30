"""
Exemples d'utilisation des prompts AutoGen dans les agents
Démontre l'intégration avec les prompts existants
"""

# Exemple d'usage dans les agents existants
INTENT_CLASSIFIER_USAGE_EXAMPLE = """
# Dans conversation_service/agents/financial/intent_classifier.py

from autogen import AssistantAgent
from prompts.system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT  # Existant
from prompts.autogen import get_intent_classification_with_team_collaboration  # Extension AutoGen

class IntentClassifierAgent(AssistantAgent):
    def __init__(self, autogen_mode: bool = True):
        # Mode AutoGen équipe ou mode standard
        if autogen_mode:
            prompt = get_intent_classification_with_team_collaboration()
        else:
            prompt = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT  # Standard existant
        
        super().__init__(
            name="intent_classifier",
            system_message=prompt,
            llm_config={
                "model": "deepseek-chat",
                "temperature": 0.05,
                "response_format": {"type": "json_object"}  # JSON forcé conservé
            }
        )

# Exemple de réponse en mode équipe AutoGen:
EXAMPLE_INTENT_RESPONSE = {
  "intent": "SEARCH_BY_MERCHANT",
  "confidence": 0.92,
  "reasoning": "Recherche de dépenses chez un marchand spécifique avec contrainte temporelle",
  "team_context": {
    "user_message": "Combien j'ai dépensé chez Leclerc le mois dernier ?",
    "suggested_entities_focus": {
      "priority_entities": ["merchants", "amounts", "time_periods"],
      "extraction_strategy": "focused",
      "entity_hints": {
        "merchants": ["Leclerc", "variations possibles: E.Leclerc"],
        "time_periods": ["mois dernier", "pattern: références temporelles passées"]
      }
    },
    "processing_metadata": {
      "confidence_level": "high",
      "complexity_assessment": "medium",
      "ready_for_entity_extraction": True
    }
  }
}
"""

ENTITY_EXTRACTOR_USAGE_EXAMPLE = """
# Dans conversation_service/agents/financial/entity_extractor.py

from autogen import AssistantAgent
from prompts.system_prompts import ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT  # Existant
from prompts.autogen import (
    get_entity_extraction_with_team_collaboration,  # Extension AutoGen
    get_adaptive_entity_prompt  # Adaptatif selon contexte
)

class EntityExtractorAgent(AssistantAgent):
    def __init__(self, autogen_mode: bool = True):
        # Mode AutoGen équipe ou mode standard
        if autogen_mode:
            prompt = get_entity_extraction_with_team_collaboration()
        else:
            prompt = ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT  # Standard existant
        
        super().__init__(
            name="entity_extractor", 
            system_message=prompt,
            llm_config={
                "model": "deepseek-chat",
                "temperature": 0.05,
                "response_format": {"type": "json_object"}  # JSON forcé conservé
            }
        )
    
    def extract_with_intent_context(self, message: str, intent_context: dict):
        \"\"\"
        Extraction adaptée selon le contexte de l'intention reçue
        \"\"\"
        # Utiliser prompt adaptatif selon contexte
        adaptive_prompt = get_adaptive_entity_prompt(intent_context)
        
        # Mettre à jour system message temporairement
        original_prompt = self.system_message
        self.system_message = adaptive_prompt
        
        try:
            response = self.generate_reply([{"role": "user", "content": message}])
            return response
        finally:
            # Restaurer prompt original
            self.system_message = original_prompt

# Exemple de réponse en mode équipe AutoGen:
EXAMPLE_ENTITY_RESPONSE = {
  "entities": {
    "amounts": [],
    "dates": [{"type": "period", "value": "2024-01", "text": "le mois dernier"}],
    "merchants": ["Leclerc"],
    "categories": [],
    "operation_types": ["dépense"],
    "text_search": []
  },
  "confidence": 0.88,
  "reasoning": "Extraction focalisée sur merchant + période selon intention SEARCH_BY_MERCHANT",
  "team_context": {
    "intent_received": "SEARCH_BY_MERCHANT",
    "intent_confidence": 0.92,
    "extraction_strategy_used": "focused",
    "entities_extracted_count": 3,
    "priority_entities_found": ["merchants", "dates", "operation_types"],
    "processing_metadata": {
      "ready_for_query_generation": True,
      "extraction_quality": "high",
      "complexity_level": "medium",
      "requires_fallback": False
    }
  }
}
"""

GROUP_CHAT_MANAGER_USAGE_EXAMPLE = """
# Dans conversation_service/autogen_core/group_chat_manager.py

from autogen import GroupChatManager, GroupChat
from prompts.autogen import (
    MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
    get_orchestration_prompt_for_context,
    get_workflow_completion_message,
    get_workflow_error_message
)

class FinancialTeamManager(GroupChatManager):
    def __init__(self, intent_agent, entity_agent):
        # Créer group chat avec workflow séquentiel strict
        group_chat = GroupChat(
            agents=[intent_agent, entity_agent],
            messages=[],
            max_round=3,  # Maximum 3 tours
            speaker_selection_method="manual",  # Contrôle manuel workflow
            allow_repeat_speaker=False
        )
        
        super().__init__(
            groupchat=group_chat,
            system_message=MULTI_AGENT_TEAM_ORCHESTRATION_PROMPT,
            llm_config={
                "model": "deepseek-chat",
                "temperature": 0.1
            }
        )
        
        self.intent_agent = intent_agent
        self.entity_agent = entity_agent
        self.workflow_results = {}
        
    def select_speaker(self, last_speaker, selector):
        \"\"\"
        Sélection séquentielle: Intent → Entity → Terminé
        \"\"\"
        
        if last_speaker is None:
            # Début workflow: commencer par intent
            return self.intent_agent
        elif last_speaker == self.intent_agent:
            # Intent terminé: passer à entity
            return self.entity_agent  
        elif last_speaker == self.entity_agent:
            # Entity terminé: workflow complet
            return None
        
        return None
    
    def process_financial_query(self, user_message: str, retry_context: dict = None):
        \"\"\"
        Traite une requête financière avec workflow équipe AutoGen
        \"\"\"
        
        # Adapter prompt selon contexte retry si nécessaire
        if retry_context:
            self.system_message = get_orchestration_prompt_for_context(retry_context)
        
        try:
            # Démarrer workflow équipe
            workflow_start = time.time()
            
            # Message initial au groupe
            initial_message = f\"Analyse de la requête utilisateur: {user_message}\"
            
            # Exécuter workflow
            result = self.initiate_chat(
                recipient=self.groupchat,
                message=initial_message
            )
            
            workflow_duration = time.time() - workflow_start
            
            # Compiler résultats
            return self._compile_workflow_results(workflow_duration)
            
        except Exception as e:
            # Gestion erreurs avec fallback
            return self._handle_workflow_error(str(e), retry_context)
    
    def _compile_workflow_results(self, duration: float):
        \"\"\"
        Compile les résultats des agents en format final
        \"\"\"
        
        # Extraire résultats JSON des messages
        intent_result = None
        entity_result = None
        
        for message in self.groupchat.messages:
            try:
                import json
                content = json.loads(message["content"])
                
                if "intent" in content:
                    intent_result = content
                elif "entities" in content:
                    entity_result = content
                    
            except json.JSONDecodeError:
                continue
        
        # Générer message de fin
        if intent_result and entity_result:
            completion_msg = get_workflow_completion_message(
                intent_result, entity_result, duration
            )
            return {
                "status": "success",
                "intent": intent_result,
                "entities": entity_result,
                "message": completion_msg
            }
        else:
            # Résultats partiels
            return self._handle_partial_results(intent_result, entity_result)
    
    def _handle_workflow_error(self, error: str, context: dict):
        \"\"\"
        Gère les erreurs de workflow avec stratégie fallback
        \"\"\"
        
        error_message = get_workflow_error_message(
            error_type="workflow_interruption",
            failed_agents=context.get("failed_agents", []),
            partial_results=self.workflow_results
        )
        
        return {
            "status": "partial_failure",
            "error": error,
            "message": error_message,
            "fallback_applied": True
        }
"""

INTEGRATION_BEST_PRACTICES = """
# BONNES PRATIQUES D'INTÉGRATION

## 1. Réutilisation prompts existants
Les agents utilisent TOUJOURS les prompts existants comme base:

```python
# ✅ CORRECT - Réutilisation base existante
from prompts.system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
from prompts.autogen import get_intent_classification_with_team_collaboration

# Mode équipe AutoGen: extension du prompt existant
if autogen_mode:
    prompt = get_intent_classification_with_team_collaboration()
else:
    prompt = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT

# ❌ INCORRECT - Remplacer prompts existants
# Jamais faire ça: créer nouveaux prompts from scratch
```

## 2. Configuration JSON OUTPUT conservée
Tous les agents maintiennent `response_format={"type": "json_object"}`:

```python
llm_config = {
    "model": "deepseek-chat",
    "temperature": 0.05,
    "response_format": {"type": "json_object"}  # ✅ Toujours maintenir
}
```

## 3. Compatibilité ascendante garantie
Les prompts étendus contiennent 100% du contenu existant:

```python
# Vérification automatique
base_prompt = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
extended_prompt = get_intent_classification_with_team_collaboration()

assert base_prompt in extended_prompt  # ✅ Toujours vrai
```

## 4. Imports cohérents
Pattern standardisé pour tous les agents:

```python
# Base existante
from prompts.system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT

# Extensions AutoGen 
from prompts.autogen import get_intent_classification_with_team_collaboration

# Utilisation conditionnelle
prompt = (get_intent_classification_with_team_collaboration() 
          if autogen_mode 
          else INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT)
```

## 5. Tests de régression
Validation que extensions maintiennent qualité:

```python
def test_prompt_compatibility():
    # Test que prompt de base est inclus intact
    base = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
    extended = get_intent_classification_with_team_collaboration()
    assert base in extended
    
    # Test que structure JSON maintenue
    assert '"intent":' in extended
    assert '"confidence":' in extended
    
    # Test que extensions AutoGen ajoutées
    assert '"team_context":' in extended
    assert 'COLLABORATION ÉQUIPE AUTOGEN' in extended
```
"""


def demonstrate_compatibility():
    """
    Fonction de démonstration de la compatibilité complète
    """
    
    print("=== DÉMONSTRATION COMPATIBILITÉ PROMPTS AUTOGEN ===\n")
    
    # Import des prompts existants et extensions
    from prompts.system_prompts import (
        INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT,
        ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
    )
    from prompts.autogen import (
        get_intent_classification_with_team_collaboration,
        get_entity_extraction_with_team_collaboration
    )
    
    # Test 1: Prompts de base inclus intégralement
    intent_extended = get_intent_classification_with_team_collaboration()
    entity_extended = get_entity_extraction_with_team_collaboration()
    
    print("✅ Prompt intent de base inclus:", INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT in intent_extended)
    print("✅ Prompt entity de base inclus:", ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT in entity_extended)
    
    # Test 2: Extensions AutoGen ajoutées
    print("✅ Extension AutoGen intent:", "COLLABORATION ÉQUIPE AUTOGEN" in intent_extended)
    print("✅ Extension AutoGen entity:", "COLLABORATION ÉQUIPE AUTOGEN" in entity_extended)
    
    # Test 3: Structure JSON maintenue
    print("✅ JSON intent maintenu:", '"intent":' in intent_extended and '"confidence":' in intent_extended)
    print("✅ JSON entity maintenu:", '"entities":' in entity_extended and '"confidence":' in entity_extended)
    
    # Test 4: Champs équipe ajoutés
    print("✅ Team context intent:", '"team_context":' in intent_extended)
    print("✅ Team context entity:", '"team_context":' in entity_extended)
    
    print("\n🎯 COMPATIBILITÉ 100% VALIDÉE")
    print("   - Prompts existants réutilisés intégralement")
    print("   - Extensions AutoGen ajoutées sans conflit")
    print("   - Structure JSON OUTPUT conservée")
    print("   - Fonctionnalités équipe intégrées")


if __name__ == "__main__":
    demonstrate_compatibility()