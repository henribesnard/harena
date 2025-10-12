# Sprint 1.4 - Tests et Déploiement Production - Plan Détaillé

**Date**: 2025-01-12
**Version cible**: v3.2.8 (post-validation production)
**Durée estimée**: 2 sessions (Phase 1: 1 session, Phase 2: 1 session)

---

## 📋 Vue d'Ensemble

Sprint 1.4 valide l'ensemble du workflow Phase 1 avec des questions utilisateur réelles, puis déploie en production après validation.

**Approche en 2 Phases** :
- **Phase 1** : Validation intensive avec vraies questions → Go/No-Go decision
- **Phase 2** : Déploiement production + monitoring (si Phase 1 OK)

---

## 🎯 Objectifs

### Objectif Principal
Valider que le workflow complet (Analytics + User Profiles + Visualizations) fonctionne correctement avec des questions utilisateur réelles avant déploiement production.

### Objectifs Secondaires
1. Détecter les bugs/edge cases avec vraies données
2. Mesurer les performances end-to-end
3. Valider la qualité des réponses LLM
4. Documenter les résultats de validation
5. Déployer en production si validation OK

---

## 📦 Phase 1 : Validation avec Questions Réelles

### T4.1 : Tests Unitaires Visualisations ✅ (Déjà fait)

**Statut** : Déjà complété dans Sprint 1.3
- 12 tests E2E visualisations passent
- Pas de travail supplémentaire nécessaire

---

### T4.2 : Tests avec Vraies Questions Utilisateur

**Objectif** : Tester le workflow complet avec 10-15 questions utilisateur réalistes

**Questions à Tester** (exemples) :

#### Catégorie 1 : Transaction Search Simple
1. "Mes dépenses du mois dernier"
2. "Transactions de cette semaine"
3. "Combien j'ai dépensé aujourd'hui ?"

**Validation attendue** :
- ✅ Intent classifié : `transaction_search.simple`
- ✅ Search service retourne transactions
- ✅ Response Generator génère réponse + insights
- ✅ Visualisations : 3 KPI Cards (Total, Count, Average)
- ✅ Pas d'erreur 500

#### Catégorie 2 : By Category
4. "Mes dépenses alimentaires ce mois-ci"
5. "Combien j'ai dépensé en transport ?"
6. "Répartition de mes dépenses par catégorie"

**Validation attendue** :
- ✅ Intent : `transaction_search.by_category`
- ✅ Filtrage par catégorie fonctionnel
- ✅ Visualisations : KPI Cards + Pie Chart
- ✅ Pie Chart affiche top 5 catégories + "Autres"

#### Catégorie 3 : By Merchant
7. "Mes dépenses chez Carrefour"
8. "Combien j'ai payé à Netflix ce mois ?"
9. "Transactions SNCF du dernier mois"

**Validation attendue** :
- ✅ Intent : `transaction_search.by_merchant`
- ✅ Filtrage par merchant fonctionnel
- ✅ Visualisations : KPI Cards + Bar Chart

#### Catégorie 4 : Analytics & Insights
10. "Quelles sont mes dépenses inhabituelles ?"
11. "Mes tendances de dépenses"
12. "Comparaison avec le mois dernier"

**Validation attendue** :
- ✅ Analytics Agent détecte anomalies
- ✅ Insights générés (unusual transactions, trends)
- ✅ Visualisations appropriées (Bar/Line charts)

#### Catégorie 5 : Edge Cases
13. "Mes transactions du 1er janvier 1970" (date invalide)
14. "Dépenses chez XYZ_MERCHANT_INEXISTANT"
15. Question ambiguë : "Mes trucs"

**Validation attendue** :
- ✅ Pas de crash
- ✅ Message d'erreur clair
- ✅ Fallback gracieux

**Script de Test** :
```python
# scripts/test_sprint_1.4_validation.py

import asyncio
from conversation_service.api.dependencies import get_context_manager

test_questions = [
    {"user_id": 1, "message": "Mes dépenses du mois dernier"},
    {"user_id": 1, "message": "Mes dépenses alimentaires ce mois-ci"},
    # ... 15 questions
]

async def test_workflow(question):
    context_manager = get_context_manager()
    result = await context_manager.process_conversation(
        user_id=question["user_id"],
        message=question["message"]
    )

    # Validation
    assert result.success is True
    assert result.response_text is not None
    assert len(result.data_visualizations) > 0  # Au moins 1 viz

    return {
        "question": question["message"],
        "intent": result.intent_group,
        "success": result.success,
        "visualizations_count": len(result.data_visualizations),
        "insights_count": len(result.insights),
        "processing_time_ms": result.processing_time_ms
    }

# Run all tests
results = await asyncio.gather(*[test_workflow(q) for q in test_questions])
```

---

### T4.3 : Validation End-to-End Workflow Complet

**Objectif** : Valider le flow complet avec monitoring des performances

**Workflow à Valider** :

```
User Question
    ↓
Intent Classifier
    ↓
Search Service (fetch transactions + aggregations)
    ↓
User Profile Service (fetch profile + metrics)
    ↓
Response Generator
    ├── Analytics Agent (insights)
    ├── VisualizationService (charts)
    └── LLM (response text)
    ↓
ResponseGenerationResult
```

**Métriques à Mesurer** :

1. **Latence** :
   - Intent classification : < 100ms
   - Search service : < 500ms
   - User profile fetch : < 100ms
   - Response generation : < 3000ms
   - **Total E2E : < 4000ms** (objectif)

2. **Taux de Succès** :
   - Objectif : **95%+** success rate
   - Max 1 erreur sur 20 questions

3. **Qualité des Visualisations** :
   - 100% des DATA_PRESENTATION responses ont visualisations
   - KPI Cards présents pour tous les intents
   - Charts appropriés selon intent type

4. **Qualité des Insights** :
   - Analytics Agent détecte anomalies (si présentes)
   - Insights pertinents générés
   - Pas de faux positifs excessifs

**Script de Monitoring** :
```python
# scripts/test_performance_monitoring.py

import time
from statistics import mean, stdev

async def measure_performance(questions):
    results = []

    for question in questions:
        start = time.time()
        result = await context_manager.process_conversation(
            user_id=1,
            message=question
        )
        elapsed_ms = (time.time() - start) * 1000

        results.append({
            "question": question,
            "latency_ms": elapsed_ms,
            "success": result.success,
            "viz_count": len(result.data_visualizations),
            "insights_count": len(result.insights)
        })

    # Statistiques
    latencies = [r["latency_ms"] for r in results]
    success_rate = sum(1 for r in results if r["success"]) / len(results) * 100

    print(f"Latency - Mean: {mean(latencies):.2f}ms, Stdev: {stdev(latencies):.2f}ms")
    print(f"Success Rate: {success_rate:.1f}%")

    return results
```

---

### T4.4 : Documentation Résultats Validation

**Objectif** : Documenter les résultats des tests et décider Go/No-Go

**Rapport de Validation** (template) :

```markdown
# Sprint 1.4 - Phase 1 - Rapport de Validation

## Résumé Exécutif
- Tests effectués : 15 questions réelles
- Taux de succès : XX%
- Latence moyenne : XXms
- Decision : GO / NO-GO

## Résultats par Catégorie

### Transaction Search Simple (3 questions)
| Question | Intent | Success | Viz Count | Latency |
|----------|--------|---------|-----------|---------|
| "Mes dépenses du mois dernier" | transaction_search.simple | ✅ | 3 | 2500ms |
| ... | ... | ... | ... | ... |

### By Category (3 questions)
...

### Anomalies Détectées
- Question X : Erreur 500 (cause: ...)
- Question Y : Latence excessive (5000ms)
- Question Z : Visualisations manquantes

## Métriques Globales
- Latency P50: XXms
- Latency P95: XXms
- Latency P99: XXms
- Success Rate: XX%
- Visualization Coverage: XX%

## Recommandations
- [ ] GO pour production (si >95% success, <4000ms P95)
- [ ] NO-GO - Corrections nécessaires
```

**Critères Go/No-Go** :

**✅ GO pour Phase 2 (Production)** si :
- Success rate ≥ 95%
- Latency P95 ≤ 4000ms
- 100% des DATA_PRESENTATION ont visualisations
- Pas de bugs critiques

**❌ NO-GO** si :
- Success rate < 95%
- Latency P95 > 5000ms
- Bugs critiques détectés
- Visualisations manquantes

---

## 📦 Phase 2 : Déploiement Production

**Condition** : Phase 1 validée avec GO decision

---

### T4.5 : Préparation Deployment Scripts

**Objectif** : Préparer les scripts de déploiement pour serveur production

**Scripts à Créer** :

#### 1. `deploy_v3.2.7_to_production.sh`
```bash
#!/bin/bash
# Deployment script for v3.2.7

set -e

ELASTIC_IP="52.210.228.191"
INSTANCE_ID="i-0147cf4bfd894e87d"
TAG="v3.2.7"

echo "=== Deploying v3.2.7 to Production ==="

# 1. SSH into instance
echo "Step 1: Connecting to production server..."

# 2. Pull latest code
echo "Step 2: Pulling v3.2.7 from GitHub..."
ssh ec2-user@$ELASTIC_IP << EOF
  cd /home/ec2-user/harena
  git fetch --tags
  git checkout $TAG
  source env/bin/activate
  pip install -r requirements.txt
EOF

# 3. Run migrations if needed
echo "Step 3: Running database migrations..."
# (if any)

# 4. Restart services
echo "Step 4: Restarting services..."
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "sudo systemctl restart conversation-service",
    "sudo systemctl restart search-service",
    "sudo systemctl status conversation-service"
  ]' \
  --output text --query 'Command.CommandId'

echo "=== Deployment Complete ==="
echo "Check health: curl http://$ELASTIC_IP:8001/api/v1/health"
```

#### 2. `rollback_to_v3.2.6.2.sh`
```bash
#!/bin/bash
# Rollback script to v3.2.6.2

set -e

ELASTIC_IP="52.210.228.191"
TAG="v3.2.6.2"

echo "=== ROLLBACK to v3.2.6.2 ==="

ssh ec2-user@$ELASTIC_IP << EOF
  cd /home/ec2-user/harena
  git checkout $TAG
  source env/bin/activate
  sudo systemctl restart conversation-service
EOF

echo "=== Rollback Complete ==="
```

#### 3. `verify_deployment.sh`
```bash
#!/bin/bash
# Verification script post-deployment

ELASTIC_IP="52.210.228.191"

echo "=== Verifying Deployment ==="

# Health check
echo "1. Health Check:"
curl -s "http://$ELASTIC_IP:8001/api/v1/health" | jq .

# Test question
echo "2. Test Question:"
curl -s -X POST "http://$ELASTIC_IP:8001/api/v1/conversation/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "message": "Mes dépenses du mois"}' | jq .

# Check logs
echo "3. Recent Logs:"
ssh ec2-user@$ELASTIC_IP "sudo journalctl -u conversation-service -n 20 --no-pager"
```

---

### T4.6 : Déploiement sur Serveur Production

**Objectif** : Déployer v3.2.7 sur le serveur de production

**Étapes de Déploiement** :

1. **Backup actuel** :
   ```bash
   # Backup database
   ssh ec2-user@52.210.228.191 "pg_dump harena > backup_pre_v3.2.7.sql"

   # Tag current version
   git tag v3.2.6.2-backup
   ```

2. **Déploiement** :
   ```bash
   ./scripts/deploy_v3.2.7_to_production.sh
   ```

3. **Vérification** :
   ```bash
   ./scripts/verify_deployment.sh
   ```

4. **Monitoring** (premières 30 minutes) :
   - Watch logs en temps réel
   - Monitor error rate
   - Tester 5-10 questions manuellement

**Checklist Déploiement** :
- [ ] Backup database effectué
- [ ] Code v3.2.7 déployé sur serveur
- [ ] Services redémarrés (conversation-service, search-service)
- [ ] Health check OK (200 status)
- [ ] Test question fonctionne
- [ ] Logs ne montrent pas d'erreurs
- [ ] Visualisations générées correctement

---

### T4.7 : Monitoring et Rollback Plan

**Objectif** : Monitorer la production et préparer rollback si nécessaire

**Monitoring Continu (24h post-deployment)** :

#### Métriques à Surveiller :
1. **Error Rate** : < 5% (objectif)
2. **Latency P95** : < 4000ms
3. **Success Rate** : > 95%
4. **Crash Rate** : 0%

#### Alertes :
- Error rate > 10% → Investigate immédiatement
- Latency P95 > 5000ms → Performance issue
- Service down → Rollback immédiat

**Rollback Plan** :

**Trigger Rollback si** :
- Error rate > 20%
- Service crash repeatedly
- Critical bug détecté
- Data corruption

**Rollback Procedure** :
```bash
# 1. Execute rollback script
./scripts/rollback_to_v3.2.6.2.sh

# 2. Verify rollback
curl http://52.210.228.191:8001/api/v1/health

# 3. Restore database if needed
ssh ec2-user@52.210.228.191 "psql harena < backup_pre_v3.2.7.sql"

# 4. Notify team
echo "ROLLBACK executed - investigating v3.2.7 issues"
```

**Post-Deployment Report** (template) :
```markdown
# v3.2.7 Production Deployment Report

## Deployment Info
- Date: YYYY-MM-DD HH:MM
- Version: v3.2.7
- Deployer: [name]
- Duration: XXm

## 24h Post-Deployment Metrics
- Total Requests: XXXX
- Success Rate: XX%
- Error Rate: XX%
- Latency P50/P95/P99: XXms / XXms / XXms
- Visualizations Generated: XXXX

## Issues Detected
- None / [list issues]

## Status
- ✅ Stable in production
- ❌ Rolled back (reason: ...)
```

---

## 🎯 Acceptance Criteria

### Phase 1 : Validation
- [ ] T4.1: Tests unitaires (déjà ✅)
- [ ] T4.2: 15 questions testées avec succès
- [ ] T4.3: Workflow E2E validé (latency, success rate)
- [ ] T4.4: Rapport de validation documenté
- [ ] **GO/NO-GO decision** : GO pour Phase 2

### Phase 2 : Déploiement
- [ ] T4.5: Scripts déploiement créés et testés
- [ ] T4.6: v3.2.7 déployé sur production
- [ ] T4.7: Monitoring 24h sans erreurs critiques
- [ ] Rapport post-deployment rédigé

---

## 📊 Livrables

### Phase 1
1. Script de test : `scripts/test_sprint_1.4_validation.py`
2. Script monitoring : `scripts/test_performance_monitoring.py`
3. Rapport validation : `SPRINT_1.4_PHASE1_VALIDATION_REPORT.md`

### Phase 2
1. Scripts déploiement : `scripts/deploy_v3.2.7_to_production.sh`
2. Script rollback : `scripts/rollback_to_v3.2.6.2.sh`
3. Script vérification : `scripts/verify_deployment.sh`
4. Rapport déploiement : `SPRINT_1.4_PHASE2_DEPLOYMENT_REPORT.md`

---

## ⚠️ Risques et Mitigation

### Risques Phase 1
| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Tests échouent | High | Medium | Fix bugs avant Phase 2 |
| Latence excessive | Medium | Low | Optimize queries |
| Visualisations manquantes | Medium | Low | Debug VisualizationService |

### Risques Phase 2
| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Crash production | Critical | Low | Rollback immédiat |
| Performance dégradée | High | Medium | Monitoring + rollback |
| Database migration fail | Critical | Very Low | Backup avant déploiement |

---

## 🚀 Timeline

### Phase 1 (Session 1 - ~2-3h)
- T4.2: Tests questions réelles (1h)
- T4.3: Validation E2E workflow (1h)
- T4.4: Documentation résultats (30min)
- **GO/NO-GO Decision**

### Phase 2 (Session 2 - ~2h)
- T4.5: Préparation scripts (30min)
- T4.6: Déploiement production (30min)
- T4.7: Monitoring initial (1h)

---

## 📝 Notes

- **Phase 1 doit être 100% validée avant Phase 2**
- Si NO-GO, corriger les issues et re-tester Phase 1
- Déploiement production uniquement si tous les critères GO sont remplis
- Rollback plan doit être prêt AVANT déploiement

---

**Prêt à démarrer Sprint 1.4 - Phase 1 ?**
