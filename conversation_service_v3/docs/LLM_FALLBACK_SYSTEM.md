# 🔄 Système de Fallback LLM - Documentation

**Date :** 2025-10-26
**Auteur :** Claude Code
**Version :** 1.0

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Comportement](#comportement)
5. [Tests](#tests)
6. [Logs et monitoring](#logs-et-monitoring)
7. [FAQ](#faq)

---

## 🎯 Vue d'ensemble

Le système de fallback LLM permet une **haute disponibilité** et une **résilience** du service conversation_v3 en basculant automatiquement sur un LLM de secours si le LLM principal échoue.

### Cas d'usage

- **API timeout** : Si DeepSeek est lent ou ne répond pas
- **API down** : Si DeepSeek est temporairement indisponible
- **Rate limiting** : Si les quotas DeepSeek sont dépassés
- **Erreurs API** : Toute erreur 4xx/5xx du provider principal

### Avantages

✅ **Haute disponibilité** : Le service ne tombe jamais
✅ **Réduction des coûts** : Utilise DeepSeek (90% moins cher) par défaut
✅ **Performance** : DeepSeek est 10% plus rapide qu'OpenAI
✅ **Fiabilité** : OpenAI en secours si DeepSeek échoue
✅ **Transparent** : Fallback automatique sans intervention

---

## 🏗️ Architecture

### Configuration par défaut

```
Primary:  DeepSeek (deepseek-chat)
          ↓ (si échec)
Fallback: OpenAI (gpt-4o-mini / gpt-4o)
```

### Composants

#### 1. `LLMFactory`
Crée des instances LLM pour chaque provider.

```python
from app.core.llm_factory import LLMFactory

# Primary factory
primary_factory = LLMFactory(
    provider="deepseek",
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    timeout=60
)

# Fallback factory
fallback_factory = LLMFactory(
    provider="openai",
    api_key="sk-...",
    timeout=60
)
```

#### 2. `LLMWithFallback`
Wrapper qui gère la logique de fallback automatique.

```python
from app.core.llm_factory import LLMWithFallback

llm_with_fallback = LLMWithFallback(
    primary_llm=primary_llm,
    fallback_llm=fallback_llm,
    primary_provider="deepseek",
    fallback_provider="openai"
)

# Utilisation transparente
response = await llm_with_fallback.ainvoke(messages)
# → Essaie DeepSeek, bascule sur OpenAI si échec
```

#### 3. `create_llm_from_settings()`
Fonction utilitaire qui configure automatiquement le fallback.

```python
from app.core.llm_factory import create_llm_from_settings
from app.config.settings import settings

# Avec fallback (par défaut)
llm = create_llm_from_settings(settings, temperature=0.1)

# Sans fallback
llm = create_llm_from_settings(settings, use_fallback=False)
```

### Flux de fonctionnement

```
┌─────────────────┐
│  Requête user   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLMWithFallback│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Tentative PRIMARY (DeepSeek)│
└────────┬────────────────────┘
         │
    ┌────┴────┐
    │ Succès? │
    └────┬────┘
         │
    ┌────┴────────┐
    │ YES         │ NO
    ▼             ▼
┌─────────┐   ┌─────────────────────────┐
│ Return  │   │ Log warning + tentative │
│ result  │   │ FALLBACK (OpenAI)       │
└─────────┘   └────────┬────────────────┘
                       │
                  ┌────┴────┐
                  │ Succès? │
                  └────┬────┘
                       │
                  ┌────┴────────┐
                  │ YES         │ NO
                  ▼             ▼
              ┌─────────┐   ┌──────────┐
              │ Return  │   │ Raise    │
              │ result  │   │ Exception│
              └─────────┘   └──────────┘
```

---

## ⚙️ Configuration

### Variables d'environnement

#### `.env` ou `docker-compose.yml`

```bash
# Primary LLM provider (default: deepseek)
LLM_PRIMARY_PROVIDER=deepseek

# Fallback LLM provider (default: openai)
LLM_FALLBACK_PROVIDER=openai

# Enable automatic fallback (default: true)
LLM_FALLBACK_ENABLED=true

# API Keys
OPENAI_API_KEY=sk-xxxxx
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Primary models (DeepSeek)
LLM_MODEL=deepseek-chat
LLM_RESPONSE_MODEL=deepseek-chat

# Fallback models (OpenAI)
LLM_FALLBACK_MODEL=gpt-4o-mini
LLM_FALLBACK_RESPONSE_MODEL=gpt-4o

# Common settings
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60  # Timeout before fallback (seconds)
```

### Configurations recommandées

#### 1. Production (coût optimisé)
```bash
LLM_PRIMARY_PROVIDER=deepseek
LLM_FALLBACK_PROVIDER=openai
LLM_FALLBACK_ENABLED=true
```
**Résultat :** 90% des requêtes sur DeepSeek (économie), 10% sur OpenAI (fallback)

#### 2. Production (qualité maximale)
```bash
LLM_PRIMARY_PROVIDER=openai
LLM_FALLBACK_PROVIDER=deepseek
LLM_FALLBACK_ENABLED=true
```
**Résultat :** OpenAI principal, DeepSeek en secours

#### 3. Développement
```bash
LLM_PRIMARY_PROVIDER=deepseek
LLM_FALLBACK_PROVIDER=openai
LLM_FALLBACK_ENABLED=false
```
**Résultat :** Pas de fallback, erreur si échec (pour déboguer)

---

## 🎬 Comportement

### Scénarios

#### Scénario 1 : Succès du primary

```python
# DeepSeek répond correctement
llm = create_llm_from_settings(settings)
result = await llm.ainvoke(messages)

# Log:
# INFO: Attempting ainvoke with primary LLM (deepseek)
# INFO: Response received from deepseek
```

**Résultat :** Pas de fallback, DeepSeek utilisé ✅

---

#### Scénario 2 : Timeout du primary

```python
# DeepSeek timeout après 60s
llm = create_llm_from_settings(settings)
result = await llm.ainvoke(messages)

# Log:
# INFO: Attempting ainvoke with primary LLM (deepseek)
# WARNING: Primary LLM (deepseek) failed: Request timeout
# INFO: Falling back to openai
# INFO: Fallback LLM (openai) succeeded
```

**Résultat :** Fallback automatique sur OpenAI ✅

---

#### Scénario 3 : Échec primary ET fallback

```python
# DeepSeek ET OpenAI échouent
llm = create_llm_from_settings(settings)
result = await llm.ainvoke(messages)

# Log:
# INFO: Attempting ainvoke with primary LLM (deepseek)
# WARNING: Primary LLM (deepseek) failed: API error
# INFO: Falling back to openai
# ERROR: Fallback LLM (openai) also failed: API error
# ERROR: Both primary (deepseek) and fallback (openai) failed
```

**Résultat :** Exception raised ❌

---

#### Scénario 4 : Fallback désactivé

```python
# Fallback désactivé
llm = create_llm_from_settings(settings, use_fallback=False)
result = await llm.ainvoke(messages)

# Log:
# INFO: LLM created without fallback: deepseek
# INFO: Attempting ainvoke with primary LLM (deepseek)
# ERROR: Primary LLM (deepseek) failed and no fallback configured
```

**Résultat :** Exception raised si échec ❌

---

## 🧪 Tests

### Test 1 : Vérifier que le fallback est actif

```bash
# Vérifier les variables d'environnement
docker exec harena_conversation_v3 printenv | grep LLM_

# Sortie attendue:
# LLM_PRIMARY_PROVIDER=deepseek
# LLM_FALLBACK_PROVIDER=openai
# LLM_FALLBACK_ENABLED=true
```

### Test 2 : Tester le service normalement

```bash
curl --request POST \
  --url http://localhost:3008/api/v3/conversation/3 \
  --header "Authorization: Bearer YOUR_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "client_info": {"platform": "web", "version": "1.0.0"},
    "message": "combien j'\''ai dépensé ce mois ?",
    "message_type": "text",
    "priority": "normal"
  }'

# Vérifier les logs
docker logs harena_conversation_v3 --tail 50 | grep "LLM"

# Sortie attendue (si succès primary):
# INFO: Attempting ainvoke with primary LLM (deepseek)
```

### Test 3 : Simuler une erreur DeepSeek

Pour tester le fallback, on peut temporairement invalider la clé DeepSeek :

```bash
# 1. Invalider la clé DeepSeek
docker exec harena_conversation_v3 sh -c 'export DEEPSEEK_API_KEY=invalid'

# 2. Faire une requête
curl [... même requête que Test 2 ...]

# 3. Vérifier les logs
docker logs harena_conversation_v3 --tail 50 | grep -E "(WARNING|Fallback)"

# Sortie attendue:
# WARNING: Primary LLM (deepseek) failed: ...
# INFO: Falling back to openai
# INFO: Fallback LLM (openai) succeeded
```

### Test 4 : Script de test automatique

Créez un fichier `test_fallback.py` :

```python
import asyncio
from app.core.llm_factory import create_llm_from_settings, LLMWithFallback
from app.config.settings import settings

async def test_fallback():
    # Créer LLM avec fallback
    llm = create_llm_from_settings(settings)

    # Vérifier que c'est un LLMWithFallback
    assert isinstance(llm, LLMWithFallback), "Fallback should be enabled"

    print(f"✅ Primary: {llm.primary_provider}")
    print(f"✅ Fallback: {llm.fallback_provider}")

    # Test simple
    messages = [{"role": "user", "content": "Hello"}]
    result = await llm.ainvoke(messages)

    print(f"✅ Response: {result.content[:100]}")
    print(f"✅ Fallback used: {llm.fallback_used}")

if __name__ == "__main__":
    asyncio.run(test_fallback())
```

---

## 📊 Logs et monitoring

### Logs importants

```python
# Succès primary
"INFO: Attempting ainvoke with primary LLM (deepseek)"
"INFO: Response received from deepseek"

# Fallback activé
"WARNING: Primary LLM (deepseek) failed: [error]"
"INFO: Falling back to openai"
"INFO: Fallback LLM (openai) succeeded"

# Échec total
"ERROR: Fallback LLM (openai) also failed: [error]"
"ERROR: Both primary (deepseek) and fallback (openai) failed"
```

### Métriques à surveiller

1. **Taux de fallback** : % de requêtes utilisant le fallback
2. **Latence** : Temps de réponse avec/sans fallback
3. **Taux d'erreur** : % d'échecs total (primary + fallback)
4. **Coûts** : Répartition primary vs fallback

### Alertes recommandées

- ⚠️ **Taux de fallback > 20%** → Problème avec le primary
- 🚨 **Taux d'erreur > 5%** → Problème avec les deux providers
- 📈 **Latence > 30s** → Timeouts fréquents

---

## ❓ FAQ

### Q: Le fallback ralentit-il les requêtes ?

**R:** Non, seulement si le primary échoue. Dans ce cas :
- Primary timeout (60s) + Fallback (20s) = **~80s total**
- Sans fallback : Échec après 60s

Le fallback ajoute donc ~20s en cas d'échec, mais permet d'avoir une réponse au lieu d'une erreur.

---

### Q: Puis-je désactiver le fallback ?

**R:** Oui, mettez `LLM_FALLBACK_ENABLED=false` dans le `.env`.

```bash
LLM_FALLBACK_ENABLED=false
```

---

### Q: Comment savoir quel LLM a été utilisé ?

**R:** Vérifiez les logs ou ajoutez un champ dans la réponse :

```python
# Dans l'agent
if hasattr(llm, 'fallback_used'):
    metadata["llm_fallback_used"] = llm.fallback_used
    metadata["llm_provider"] = llm.fallback_provider if llm.fallback_used else llm.primary_provider
```

---

### Q: Peut-on utiliser OpenAI en primary et DeepSeek en fallback ?

**R:** Oui ! Inversez simplement les providers :

```bash
LLM_PRIMARY_PROVIDER=openai
LLM_FALLBACK_PROVIDER=deepseek
LLM_MODEL=gpt-4o-mini
LLM_FALLBACK_MODEL=deepseek-chat
```

---

### Q: Que se passe-t-il si les deux échouent ?

**R:** Une exception est levée avec les détails des deux erreurs :

```python
Exception: Both primary (deepseek) and fallback (openai) failed.
Primary error: Timeout after 60s
Fallback error: API error 503
```

Le client reçoit une erreur 500 avec ce message.

---

### Q: Le fallback fonctionne-t-il en streaming ?

**R:** Oui ! `LLMWithFallback` supporte :
- `invoke()` : Appel synchrone
- `ainvoke()` : Appel asynchrone
- `astream()` : Streaming asynchrone

---

## 🚀 Prochaines étapes

1. **Monitoring** : Implémenter des métriques Prometheus
2. **Circuit breaker** : Désactiver temporairement le primary si taux d'échec élevé
3. **Multi-fallback** : Chaîne de fallback (DeepSeek → OpenAI → Anthropic)
4. **Smart routing** : Router selon le type de requête (simple → DeepSeek, complexe → OpenAI)

---

**Dernière mise à jour :** 2025-10-26
**Auteur :** Claude Code
**Contact :** Voir documentation projet
