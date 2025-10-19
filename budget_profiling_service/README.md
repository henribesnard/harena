# Budget Profiling Service

Service de profilage budgétaire et recommandations intelligentes pour Harena.

## 📋 Vue d'ensemble

Le Budget Profiling Service analyse automatiquement les transactions des utilisateurs pour :
- Détecter les charges fixes récurrentes
- Établir un profil financier personnalisé
- Calculer les métriques budgétaires (revenus, dépenses, épargne)
- Fournir des insights sur les habitudes de dépenses

## 🚀 Fonctionnalités

### Phase 1 : Fondations (Implémenté)

#### 1. Détection automatique des charges fixes
- Analyse de la récurrence des transactions
- Calcul de la régularité (montant, date)
- Score de confiance pour chaque charge détectée
- Catégorisation automatique

#### 2. Profil budgétaire utilisateur
- Segmentation : `budget_serré`, `équilibré`, `confortable`
- Pattern comportemental : `acheteur_impulsif`, `dépensier_hebdomadaire`, `planificateur`
- Métriques mensuelles moyennes (revenus, dépenses, épargne)
- Répartition des charges (fixes, semi-fixes, variables)
- Reste à vivre après charges fixes

#### 3. Agrégations et analytics
- Agrégats mensuels (revenus, dépenses, cashflow)
- Répartition par catégorie
- Évolution temporelle

## 📊 Modèles de données

### 1. UserBudgetProfile
Stocke le profil budgétaire complet de l'utilisateur.

### 2. FixedCharge
Charges fixes détectées automatiquement.

### 3. SavingsGoal
Objectifs d'épargne définis par l'utilisateur (à venir).

### 4. BudgetRecommendation
Recommandations personnalisées (à venir).

### 5. SeasonalPattern
Patterns saisonniers de dépenses (à venir).

## 🔧 API Endpoints

### Profil budgétaire

#### `GET /api/v1/budget/profile`
Récupère le profil budgétaire de l'utilisateur.

**Réponse :**
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
  "last_analyzed_at": "2025-01-14T12:00:00Z"
}
```

#### `POST /api/v1/budget/profile/analyze`
Analyse et calcule le profil budgétaire.

**Requête :**
```json
{
  "months_analysis": 3
}
```

**Réponse :**
Même format que GET /profile

### Charges fixes

#### `GET /api/v1/budget/fixed-charges`
Récupère les charges fixes détectées.

**Réponse :**
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
  }
]
```

### Agrégats

#### `GET /api/v1/budget/monthly-aggregates?months=3`
Récupère les agrégats mensuels.

**Réponse :**
```json
[
  {
    "month": "2025-01",
    "total_income": 3100.00,
    "total_expenses": 2450.00,
    "net_cashflow": 650.00,
    "transaction_count": 145
  }
]
```

#### `GET /api/v1/budget/category-breakdown?months=3`
Répartition des dépenses par catégorie.

**Réponse :**
```json
{
  "alimentation": 450.00,
  "transport": 120.00,
  "loisirs": 230.00,
  "eau_electricite": 85.50
}
```

## 🔐 Authentification

Toutes les routes (sauf `/health`) nécessitent un token JWT valide.

**Header requis :**
```
Authorization: Bearer <jwt_token>
```

Le token doit contenir le champ `sub` avec l'ID utilisateur.

## 🏗️ Architecture

### Services principaux

1. **TransactionService**
   - Récupération et formatage des transactions
   - Agrégations mensuelles
   - Breakdown par catégorie

2. **FixedChargeDetector**
   - Détection automatique des charges fixes
   - Calcul du score de confiance
   - Catégorisation

3. **BudgetProfiler**
   - Calcul du profil budgétaire complet
   - Détermination du segment utilisateur
   - Analyse du pattern comportemental

## 🚀 Démarrage

### Avec Docker (recommandé)

```bash
# Construire et démarrer le service
docker-compose up -d budget_profiling_service

# Vérifier les logs
docker logs -f harena_budget_profiling_service
```

### En local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Démarrer le service
python -m budget_profiling_service.main
```

Le service sera accessible sur `http://localhost:3006`

## 🧪 Tests

### Health check
```bash
curl http://localhost:3006/health
```

### Tester l'analyse (avec JWT)
```bash
curl -X POST http://localhost:3006/api/v1/budget/profile/analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"months_analysis": 3}'
```

## 📈 Roadmap

### Phase 2 : Recommandations (À venir)
- Génération de recommandations personnalisées
- Scénarios d'optimisation budgétaire
- Alertes de dépassement

### Phase 3 : Objectifs & Saisonnalité (À venir)
- Gestion des objectifs d'épargne
- Détection des patterns saisonniers
- Prédictions de dépenses futures

### Phase 4 : ML & Optimisations (À venir)
- Modèles prédictifs avancés
- Comparaison avec profils similaires
- Suggestions de changement de fournisseur

## 🔧 Configuration

Variables d'environnement :

```env
# Base de données
DATABASE_URL=postgresql://...

# Sécurité
SECRET_KEY=your-secret-key

# Service
BUDGET_PROFILING_ENABLED=true
BUDGET_PROFILING_LOG_LEVEL=INFO
BUDGET_PROFILING_PORT=3006
BUDGET_PROFILING_HOST=0.0.0.0

# Environnement
ENVIRONMENT=dev
```

## 📝 Logs

Les logs incluent :
- Détection de charges fixes
- Calcul de profils budgétaires
- Erreurs d'API
- Métriques de performance

Niveau de log configurable via `BUDGET_PROFILING_LOG_LEVEL`.

## 🤝 Intégration avec les autres services

- **db_service** : Accès aux modèles de données
- **user_service** : Authentification JWT
- **config_service** : Configuration partagée

## 📄 Licence

Propriétaire - Harena
