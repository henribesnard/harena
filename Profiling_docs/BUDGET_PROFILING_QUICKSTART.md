# Budget Profiling Service - Guide de démarrage rapide

## 🚀 Démarrage avec Docker

### 1. Construire et démarrer le service

```bash
# À la racine du projet
docker-compose up -d budget_profiling_service

# Vérifier que le service est démarré
docker ps | grep budget_profiling
```

### 2. Vérifier le health check

```bash
curl http://localhost:3006/health
```

**Réponse attendue :**
```json
{
  "status": "healthy",
  "service": "budget_profiling",
  "version": "1.0.0",
  "uptime_seconds": 12.5,
  "features": [
    "transaction_analysis",
    "fixed_charges_detection",
    "budget_profiling",
    "recommendations",
    "savings_goals",
    "seasonal_patterns"
  ],
  "timestamp": "2025-10-18T18:00:00Z"
}
```

### 3. Consulter les logs

```bash
# Logs en temps réel
docker logs -f harena_budget_profiling_service

# Dernières 50 lignes
docker logs --tail 50 harena_budget_profiling_service
```

## 🔐 Obtenir un token JWT

Le service nécessite un token JWT valide pour toutes les routes API (sauf `/health`).

### Méthode 1 : Via user_service

```bash
# 1. Se connecter au user_service
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "votre@email.com",
    "password": "votre_mot_de_passe"
  }'

# 2. Récupérer le token dans la réponse
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Méthode 2 : Token existant

Si vous avez déjà un token depuis le frontend ou une autre session :

```bash
export TOKEN="votre_token_jwt_ici"
```

## 🧪 Tester les endpoints

### 1. Analyser le profil budgétaire

```bash
# Analyser les 3 derniers mois
curl -X POST http://localhost:3006/api/v1/budget/profile/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"months_analysis": 3}'
```

**Réponse attendue :**
```json
{
  "user_segment": "équilibré",
  "behavioral_pattern": "planificateur",
  "avg_monthly_income": 3000.00,
  "avg_monthly_expenses": 2400.00,
  "avg_monthly_savings": 600.00,
  "savings_rate": 20.00,
  "fixed_charges_total": 1200.00,
  "semi_fixed_charges_total": 800.00,
  "variable_charges_total": 400.00,
  "remaining_to_live": 1800.00,
  "profile_completeness": 0.85,
  "last_analyzed_at": "2025-10-18T18:00:00Z"
}
```

### 2. Récupérer le profil existant

```bash
curl http://localhost:3006/api/v1/budget/profile \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Récupérer les charges fixes détectées

```bash
curl http://localhost:3006/api/v1/budget/fixed-charges \
  -H "Authorization: Bearer $TOKEN"
```

**Réponse attendue :**
```json
[
  {
    "id": 1,
    "merchant_name": "EDF",
    "category": "eau_electricite",
    "avg_amount": 85.50,
    "recurrence_day": 15,
    "recurrence_confidence": 0.92,
    "validated_by_user": false,
    "transaction_count": 6
  },
  {
    "id": 2,
    "merchant_name": "Orange",
    "category": "telephone_internet",
    "avg_amount": 39.99,
    "recurrence_day": 1,
    "recurrence_confidence": 0.98,
    "validated_by_user": false,
    "transaction_count": 6
  }
]
```

### 4. Récupérer les agrégats mensuels

```bash
curl "http://localhost:3006/api/v1/budget/monthly-aggregates?months=3" \
  -H "Authorization: Bearer $TOKEN"
```

**Réponse attendue :**
```json
[
  {
    "month": "2025-10",
    "total_income": 3100.00,
    "total_expenses": 2450.00,
    "net_cashflow": 650.00,
    "transaction_count": 145
  },
  {
    "month": "2025-09",
    "total_income": 3000.00,
    "total_expenses": 2400.00,
    "net_cashflow": 600.00,
    "transaction_count": 138
  },
  {
    "month": "2025-08",
    "total_income": 2900.00,
    "total_expenses": 2350.00,
    "net_cashflow": 550.00,
    "transaction_count": 142
  }
]
```

### 5. Récupérer la répartition par catégorie

```bash
curl "http://localhost:3006/api/v1/budget/category-breakdown?months=3" \
  -H "Authorization: Bearer $TOKEN"
```

**Réponse attendue :**
```json
{
  "alimentation": 450.00,
  "transport": 120.00,
  "loisirs": 230.00,
  "eau_electricite": 85.50,
  "telephone_internet": 39.99,
  "assurances": 150.00,
  "shopping": 180.00
}
```

## 🔄 Workflow complet d'utilisation

### Scénario : Premier utilisateur

```bash
# 1. L'utilisateur se connecte et obtient un token
TOKEN=$(curl -s -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}' \
  | jq -r '.access_token')

# 2. Lancer l'analyse du profil budgétaire
curl -X POST http://localhost:3006/api/v1/budget/profile/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"months_analysis": 6}' | jq

# 3. Consulter les charges fixes détectées
curl http://localhost:3006/api/v1/budget/fixed-charges \
  -H "Authorization: Bearer $TOKEN" | jq

# 4. Voir l'évolution mensuelle
curl "http://localhost:3006/api/v1/budget/monthly-aggregates?months=6" \
  -H "Authorization: Bearer $TOKEN" | jq

# 5. Analyser la répartition des dépenses
curl "http://localhost:3006/api/v1/budget/category-breakdown?months=6" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## 📊 Interprétation des résultats

### Segment utilisateur

- **budget_serré** : Dépenses > 90% des revenus
  - ⚠️ Attention : Peu de marge de manœuvre
  - Recommandation : Identifier les dépenses compressibles

- **équilibré** : Dépenses entre 70-90% des revenus
  - ✅ Bon équilibre
  - Recommandation : Possibilité d'optimisation modérée

- **confortable** : Dépenses < 70% des revenus
  - 🎉 Très bonne situation
  - Recommandation : Augmenter l'épargne ou investir

### Pattern comportemental

- **acheteur_impulsif** : Nombreuses petites transactions
  - Caractéristiques : >10 tx/semaine, montant moyen <20€
  - Conseil : Suivre de près les petites dépenses

- **dépensier_hebdomadaire** : Dépenses régulières moyennes
  - Caractéristiques : 5-10 tx/semaine, montant moyen 20-50€
  - Conseil : Planification hebdomadaire

- **planificateur** : Grosses transactions espacées
  - Caractéristiques : <5 tx/semaine, montant moyen >50€
  - Conseil : Maintenir la planification

### Score de confiance des charges fixes

- **0.9 - 1.0** : Très fiable
  - ✅ Charge fixe confirmée
  - Recommandation : Peut être budgétée avec confiance

- **0.7 - 0.9** : Fiable
  - ✅ Probablement une charge fixe
  - Recommandation : Vérifier la régularité

- **< 0.7** : Incertain
  - ⚠️ Besoin de plus de données
  - Recommandation : Attendre plus de mois

## 🐛 Dépannage

### Le service ne démarre pas

```bash
# Vérifier les logs
docker logs harena_budget_profiling_service

# Vérifier la connexion DB
docker exec harena_budget_profiling_service env | grep DATABASE_URL

# Redémarrer le service
docker-compose restart budget_profiling_service
```

### Erreur 401 Unauthorized

- Vérifier que le token JWT est valide
- Vérifier que le token n'est pas expiré
- Vérifier que le header Authorization est bien formaté

```bash
# Format correct
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Format incorrect
Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # Manque "Bearer "
```

### Erreur 404 Profil non trouvé

- L'utilisateur n'a pas encore lancé d'analyse
- Solution : Appeler d'abord `POST /api/v1/budget/profile/analyze`

### Pas de charges fixes détectées

- Besoin de minimum 3 occurrences d'une transaction
- Augmenter le paramètre `months_analysis` (ex: 6 ou 12 mois)
- Vérifier que l'utilisateur a des transactions régulières

## 📈 Métriques et monitoring

### Vérifier la santé du service

```bash
# Utiliser jq pour formater
curl -s http://localhost:3006/health | jq
```

### Surveiller les performances

```bash
# Statistiques Docker
docker stats harena_budget_profiling_service

# Utilisation mémoire
docker exec harena_budget_profiling_service ps aux
```

## 🔗 Intégration avec le frontend

### Exemple React/TypeScript

```typescript
// services/budgetProfilingService.ts
import axios from 'axios';

const API_URL = 'http://localhost:3006/api/budget';

export const budgetProfilingService = {
  // Analyser le profil
  async analyzeProfile(token: string, months: number = 3) {
    const response = await axios.post(
      `${API_URL}/profile/analyze`,
      { months_analysis: months },
      {
        headers: {
          Authorization: `Bearer ${token}`
        }
      }
    );
    return response.data;
  },

  // Récupérer le profil
  async getProfile(token: string) {
    const response = await axios.get(`${API_URL}/profile`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    return response.data;
  },

  // Récupérer les charges fixes
  async getFixedCharges(token: string) {
    const response = await axios.get(`${API_URL}/fixed-charges`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    return response.data;
  }
};
```

## 📝 Checklist de vérification

Avant de considérer le service opérationnel :

- [ ] Le service démarre sans erreur
- [ ] `/health` retourne 200 OK
- [ ] L'authentification JWT fonctionne
- [ ] L'analyse de profil retourne des données
- [ ] Les charges fixes sont détectées
- [ ] Les agrégats mensuels sont corrects
- [ ] Les logs sont compréhensibles
- [ ] Le service redémarre automatiquement en cas d'erreur

## 🎯 Prochaines étapes

Une fois le service validé :

1. **Tester avec des données réelles**
   - Plusieurs profils utilisateurs différents
   - Différents patterns de dépenses

2. **Ajuster les paramètres de détection**
   - Variance montant si trop de faux négatifs
   - Variance jour si trop de faux positifs

3. **Intégrer au frontend**
   - Page de profil budgétaire
   - Tableau de bord des charges fixes
   - Graphiques d'évolution

4. **Passer à la Phase 2**
   - Recommandations budgétaires
   - Objectifs d'épargne
   - Alertes

---

**Support :** Si vous rencontrez des problèmes, vérifiez d'abord les logs avec `docker logs harena_budget_profiling_service`
