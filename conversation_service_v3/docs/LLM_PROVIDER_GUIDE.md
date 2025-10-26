# Guide d'utilisation des LLM Providers

Ce guide explique comment utiliser différents LLM providers (OpenAI, DeepSeek) avec conversation_service_v3 et comment benchmarker leurs performances.

## Table des matières

1. [Configuration](#configuration)
2. [Providers supportés](#providers-supportés)
3. [Basculer entre providers](#basculer-entre-providers)
4. [Benchmarking](#benchmarking)
5. [Limitations connues](#limitations-connues)

---

## Configuration

### Variables d'environnement

Configurez votre fichier `.env` avec les variables suivantes :

```bash
# Choix du provider
LLM_PROVIDER=openai  # ou "deepseek"

# Configuration OpenAI
OPENAI_API_KEY=sk-xxxxx

# Configuration DeepSeek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Modèles (selon le provider)
LLM_MODEL=gpt-4o-mini              # ou "deepseek-chat"
LLM_RESPONSE_MODEL=gpt-4o          # ou "deepseek-chat"
LLM_TEMPERATURE=0.1
```

---

## Providers supportés

### 1. OpenAI

**Modèles recommandés :**
- `gpt-4o-mini` : Rapide et économique (agents d'analyse)
- `gpt-4o` : Qualité optimale (génération de réponse)
- `gpt-4-turbo` : Bon compromis vitesse/qualité

**Avantages :**
- ✅ Excellent function calling
- ✅ Très bon multi-turn
- ✅ Latence faible
- ✅ Qualité constante

**Configuration :**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxx
LLM_MODEL=gpt-4o-mini
LLM_RESPONSE_MODEL=gpt-4o
```

### 2. DeepSeek

**Modèles disponibles :**
- `deepseek-chat` : Modèle standard (recommandé)
- `deepseek-reasoner` : Mode "thinking" (plus lent, raisonnement explicite)

**Avantages :**
- ✅ Compatible API OpenAI
- ✅ Support function calling
- ✅ Contexte 128K tokens
- ✅ Coût potentiellement plus bas

**Limitations :**
- ⚠️ Moins optimal pour multi-turn function calling
- ⚠️ Peut nécessiter des ajustements de prompts

**Configuration :**
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_RESPONSE_MODEL=deepseek-chat
```

---

## Basculer entre providers

### Méthode 1 : Modifier le .env

1. Ouvrir le fichier `.env`
2. Changer `LLM_PROVIDER=deepseek`
3. Vérifier que `DEEPSEEK_API_KEY` est configurée
4. Ajuster `LLM_MODEL=deepseek-chat`
5. Redémarrer le service

```bash
# Dans docker-compose
docker-compose restart harena_conversation_service_v3
```

### Méthode 2 : Variables d'environnement Docker

Vous pouvez override dans `docker-compose.yml` :

```yaml
services:
  harena_conversation_service_v3:
    environment:
      - LLM_PROVIDER=deepseek
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - LLM_MODEL=deepseek-chat
```

---

## Benchmarking

### Objectif

Comparer OpenAI et DeepSeek sur des questions analytiques pour évaluer :
- **Latence** : Temps de réponse
- **Qualité** : Précision des analyses
- **Fiabilité** : Taux de succès des function calls

### Script de benchmark

Créez un fichier `test_llm_benchmark.py` :

```python
"""
Benchmark OpenAI vs DeepSeek pour questions analytiques
"""

import time
import asyncio
from datetime import datetime
from app.config.settings import settings
from app.agents.intent_router_agent import IntentRouterAgent
from app.agents.query_analyzer_agent import QueryAnalyzerAgent
from app.agents.elasticsearch_builder_agent import ElasticsearchBuilderAgent
from app.models import UserQuery

# Questions de test (analytiques)
TEST_QUERIES = [
    "Combien j'ai dépensé en restaurants ce mois ?",
    "Quelle est ma plus grosse dépense en loisirs ?",
    "Compare mes dépenses de janvier à février",
    "Quel est mon taux d'épargne cette année ?",
    "Analyse mes achats de plus de 100 euros",
]

async def benchmark_query(query_text: str, provider: str):
    """Benchmark une seule query"""

    # Configurer le provider
    original_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = provider

    # Initialiser les agents
    intent_agent = IntentRouterAgent()
    query_agent = QueryAnalyzerAgent()
    builder_agent = ElasticsearchBuilderAgent()

    # Mesurer le temps
    start = time.time()

    try:
        # 1. Intent classification
        user_query = UserQuery(user_id=1, message=query_text)
        intent_response = await intent_agent.classify_intent(user_query)

        if not intent_response.success:
            return {"error": "Intent classification failed", "time": time.time() - start}

        # 2. Query analysis
        analysis_response = await query_agent.analyze(
            user_query,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

        if not analysis_response.success:
            return {"error": "Query analysis failed", "time": time.time() - start}

        # 3. Build Elasticsearch query
        es_response = await builder_agent.build_query(
            analysis_response.data,
            user_id=1,
            user_query=query_text
        )

        if not es_response.success:
            return {"error": "ES query build failed", "time": time.time() - start}

        elapsed = time.time() - start

        return {
            "success": True,
            "time": elapsed,
            "intent": intent_response.data.category.value,
            "confidence": analysis_response.data.confidence
        }

    except Exception as e:
        return {"error": str(e), "time": time.time() - start}

    finally:
        settings.LLM_PROVIDER = original_provider

async def run_benchmark():
    """Execute le benchmark complet"""

    print("🔬 BENCHMARK OpenAI vs DeepSeek")
    print("=" * 60)

    providers = ["openai", "deepseek"]
    results = {provider: [] for provider in providers}

    for query in TEST_QUERIES:
        print(f"\n📝 Query: {query}")

        for provider in providers:
            print(f"  Testing {provider}...", end=" ")
            result = await benchmark_query(query, provider)
            results[provider].append(result)

            if result.get("success"):
                print(f"✅ {result['time']:.2f}s")
            else:
                print(f"❌ {result.get('error')}")

    # Calcul des stats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)

    for provider in providers:
        successful = [r for r in results[provider] if r.get("success")]
        total = len(results[provider])

        print(f"\n{provider.upper()}:")
        print(f"  Succès: {len(successful)}/{total} ({len(successful)/total*100:.1f}%)")

        if successful:
            avg_time = sum(r["time"] for r in successful) / len(successful)
            min_time = min(r["time"] for r in successful)
            max_time = max(r["time"] for r in successful)

            print(f"  Latence moyenne: {avg_time:.2f}s")
            print(f"  Latence min: {min_time:.2f}s")
            print(f"  Latence max: {max_time:.2f}s")

            avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
            print(f"  Confiance moyenne: {avg_confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

### Exécution du benchmark

```bash
# Dans le container Docker
docker exec -it harena_conversation_service_v3 python test_llm_benchmark.py

# Ou en local
python conversation_service_v3/test_llm_benchmark.py
```

### Métriques à observer

1. **Latence moyenne** : Temps de traitement complet
2. **Taux de succès** : % de queries traitées correctement
3. **Confiance moyenne** : Score de confiance du LLM
4. **Erreurs** : Types d'erreurs rencontrées

### Exemple de résultats attendus

```
📊 RÉSULTATS
============================================================

OPENAI:
  Succès: 5/5 (100.0%)
  Latence moyenne: 2.34s
  Latence min: 1.82s
  Latence max: 3.01s
  Confiance moyenne: 0.94

DEEPSEEK:
  Succès: 5/5 (100.0%)
  Latence moyenne: 2.89s
  Latence min: 2.12s
  Latence max: 3.67s
  Confiance moyenne: 0.91
```

---

## Limitations connues

### DeepSeek

1. **Multi-turn function calling** : Moins performant que OpenAI pour les conversations complexes nécessitant plusieurs appels de fonction séquentiels.

   **Impact** : Faible dans notre architecture car la plupart des queries sont single-turn.

2. **Variations de prompt** : Peut nécessiter des ajustements de prompts pour obtenir la même qualité qu'OpenAI.

   **Solution** : Tester et ajuster les prompts si nécessaire.

3. **Latence** : Peut être légèrement plus lent qu'OpenAI (dépend de la charge serveur).

### Recommandations

- **Production** : OpenAI pour garantir la meilleure qualité
- **Développement/Test** : DeepSeek pour réduire les coûts
- **Hybride** : OpenAI pour ResponseGeneratorAgent, DeepSeek pour les autres agents

---

## Architecture technique

### LLMFactory

La factory (`app/core/llm_factory.py`) gère la création des instances LLM :

```python
from app.core.llm_factory import create_llm_from_settings
from app.config.settings import settings

# Crée automatiquement le bon LLM selon settings.LLM_PROVIDER
llm = create_llm_from_settings(settings, temperature=0.1)
```

### Compatibilité API

DeepSeek utilise l'API OpenAI, donc le code reste identique :

```python
# Fonctionne avec les deux providers
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",           # ou "gpt-4o-mini"
    base_url="https://api.deepseek.com",  # seulement pour DeepSeek
    api_key="sk-xxxx",
    temperature=0.1
)
```

---

## Support

Pour toute question ou problème :

1. Vérifier les logs : `docker logs harena_conversation_service_v3`
2. Vérifier la configuration : `.env` et `settings.py`
3. Tester les clés API : `curl https://api.deepseek.com/v1/models -H "Authorization: Bearer $DEEPSEEK_API_KEY"`

---

**Dernière mise à jour** : 2025-10-26
**Auteur** : Claude Code
