# User Profile Service Documentation

## Vue d'ensemble

Le User Profile Service gère la mémoire long-terme et la personnalisation pour chaque utilisateur. Il stocke :

1. **Préférences Explicites** : Catégories/marchands favoris, préférences de notification
2. **Habitudes Implicites** : Patterns de requêtes, comportements de dépenses
3. **Historique Interactions** : Recommandations acceptées/rejetées, alertes créées

## Architecture

```
conversation_service/
├── models/user_profile/
│   ├── __init__.py
│   └── entities.py              # Models Pydantic + SQLAlchemy
├── services/user_profile/
│   ├── __init__.py
│   ├── profile_service.py       # CRUD operations
│   └── README.md               # Cette documentation
└── alembic/versions/
    └── add_user_profiles_table.py  # Migration DB
```

## Modèles de Données

### UserProfile (Pydantic)

```python
class UserProfile(BaseModel):
    user_id: int
    preferences: UserPreferences
    habits: UserHabits
    interaction_history: InteractionHistory
    created_at: datetime
    last_updated: datetime
    profile_completeness: float  # 0.0 to 1.0
    total_queries: int
    total_sessions: int
    last_active: Optional[datetime]
```

### UserPreferences

```python
class UserPreferences(BaseModel):
    preferred_categories: List[str]
    preferred_merchants: List[str]
    notification_preference: NotificationPreference  # all, important_only, none
    email_notifications: bool
    push_notifications: bool
    currency: str  # Default: "EUR"
    date_format: str
    language: str  # Default: "fr"
    default_period: str  # Default: "month"
    show_trends: bool
    show_insights: bool
```

### UserHabits

```python
class UserHabits(BaseModel):
    frequent_query_patterns: List[QueryPattern]
    query_frequency: Dict[str, int]  # Pattern -> count
    average_spending_by_category: Dict[str, float]
    peak_spending_days: List[str]  # Days of week
    peak_spending_months: List[int]  # Months
    preferred_visualization_types: List[str]
    average_session_duration_minutes: float
    queries_per_session: float
```

### InteractionHistory

```python
class InteractionHistory(BaseModel):
    accepted_recommendations: List[Dict[str, Any]]
    dismissed_recommendations: List[Dict[str, Any]]
    created_alerts: List[Dict[str, Any]]
    triggered_alerts_count: int
    positive_feedback_count: int
    negative_feedback_count: int
```

## Query Patterns Détectés

```python
class QueryPattern(Enum):
    YOY_COMPARISONS = "yoy_comparisons"
    MOM_COMPARISONS = "mom_comparisons"
    MONTHLY_REPORTS = "monthly_reports"
    CATEGORY_ANALYSIS = "category_analysis"
    MERCHANT_TRACKING = "merchant_tracking"
    ANOMALY_DETECTION = "anomaly_detection"
    BUDGET_TRACKING = "budget_tracking"
```

## Utilisation du Service

### Initialisation

```python
from sqlalchemy.orm import Session
from conversation_service.services.user_profile import UserProfileService

# Initialiser le service avec session DB
service = UserProfileService(db_session=session)
```

### Opérations CRUD de Base

#### 1. Créer un profil

```python
# Créer profil vide par défaut
profile = await service.create_profile(user_id=100)

# Ou créer avec préférences initiales
initial_profile = UserProfile(
    user_id=100,
    preferences=UserPreferences(
        preferred_categories=["Alimentation", "Transport"],
        language="fr"
    )
)
profile = await service.create_profile(user_id=100, profile=initial_profile)
```

#### 2. Récupérer un profil

```python
profile = await service.get_profile(user_id=100)

if profile:
    print(f"Complétude: {profile.profile_completeness}")
    print(f"Dernière activité: {profile.last_active}")
```

#### 3. Mettre à jour un profil

```python
# Récupérer, modifier, sauvegarder
profile = await service.get_profile(user_id=100)
profile.preferences.currency = "USD"
profile.habits.average_spending_by_category["Alimentation"] = 450.0

updated_profile = await service.update_profile(user_id=100, profile=profile)
```

#### 4. Supprimer un profil

```python
success = await service.delete_profile(user_id=100)
```

### Opérations Avancées

#### Ajouter un pattern de requête détecté

```python
from conversation_service.models.user_profile.entities import QueryPattern

# Détecté une requête YoY
profile = await service.add_query_pattern(user_id=100, pattern=QueryPattern.YOY_COMPARISONS)

# Automatiquement incrémente la fréquence
print(profile.habits.query_frequency)  # {"yoy_comparisons": 5}
```

#### Ajouter catégories/marchands préférés

```python
# Ajouter catégorie favorite
profile = await service.add_preferred_category(user_id=100, category="Alimentation")

# Ajouter marchand favori
profile = await service.add_preferred_merchant(user_id=100, merchant="Tesla")
```

#### Enregistrer feedback recommandation

```python
recommendation_data = {
    "type": "budget_alert",
    "category": "Alimentation",
    "threshold": 500.0
}

# Recommandation acceptée
profile = await service.record_recommendation_feedback(
    user_id=100,
    recommendation_id="rec_12345",
    accepted=True,
    recommendation_data=recommendation_data
)

# Recommandation rejetée
profile = await service.record_recommendation_feedback(
    user_id=100,
    recommendation_id="rec_67890",
    accepted=False,
    recommendation_data=recommendation_data
)
```

#### Enregistrer création d'alerte

```python
alert_data = {
    "type": "transaction_threshold",
    "threshold": 200.0,
    "category": "all"
}

profile = await service.record_alert_created(user_id=100, alert_data=alert_data)
```

#### Mettre à jour activité

```python
# Incrémenter nombre de sessions
profile = await service.update_session_activity(user_id=100)

# Incrémenter nombre de requêtes
profile = await service.increment_query_count(user_id=100)
```

### Requêtes Analytics

#### Profils actifs récents

```python
# Profils actifs dans les 30 derniers jours
active_profiles = await service.get_active_profiles(days=30, limit=100)

for profile in active_profiles:
    print(f"User {profile.user_id}: {profile.total_queries} queries")
```

#### Profils par complétude

```python
# Profils incomplets (< 50% complétude)
incomplete_profiles = await service.get_profiles_by_completeness(
    min_completeness=0.0,
    max_completeness=0.5,
    limit=100
)

# Profils complets (> 80% complétude)
complete_profiles = await service.get_profiles_by_completeness(
    min_completeness=0.8,
    max_completeness=1.0,
    limit=100
)
```

## Calcul de Complétude

Le score de complétude (0.0 à 1.0) est calculé automatiquement :

```python
profile.calculate_completeness()
profile.update_completeness()
```

**Facteurs (7 points max)** :
- Préférences (3 points) :
  - Catégories favorites définies (1 pt)
  - Marchands favoris définis (1 pt)
  - Préférences notification configurées (1 pt)
- Habitudes (2 points) :
  - Patterns de requêtes détectés (1 pt)
  - Dépenses moyennes par catégorie calculées (1 pt)
- Historique (2 points) :
  - Feedbacks sur recommandations (1 pt)
  - Alertes créées (1 pt)

## Migration DB

### Appliquer la migration

```bash
# Générer migration (déjà créée)
alembic revision --autogenerate -m "add user_profiles table"

# Appliquer migration
alembic upgrade head
```

### Vérifier table créée

```sql
-- PostgreSQL
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'user_profiles';

-- Vérifier indexes
SELECT indexname FROM pg_indexes WHERE tablename = 'user_profiles';
```

## Exemples d'Intégration

### Dans le Response Generator

```python
from conversation_service.services.user_profile import UserProfileService

class ResponseGenerator:
    def __init__(self, db_session):
        self.profile_service = UserProfileService(db_session)

    async def generate_response(self, user_id: int, query: str):
        # Récupérer profil utilisateur
        profile = await self.profile_service.get_profile(user_id)

        if not profile:
            profile = await self.profile_service.create_profile(user_id)

        # Détecter pattern
        if "année dernière" in query.lower():
            await self.profile_service.add_query_pattern(
                user_id, QueryPattern.YOY_COMPARISONS
            )

        # Personnaliser réponse selon préférences
        currency = profile.preferences.currency
        show_trends = profile.preferences.show_trends

        # Incrémenter compteur requêtes
        await self.profile_service.increment_query_count(user_id)

        # ... génération réponse ...
```

### Background Job Analytics

```python
from conversation_service.services.user_profile import UserProfileService

async def update_spending_patterns():
    """
    Background job: Calcule patterns de dépenses pour profils actifs
    À exécuter hebdomadairement
    """
    service = UserProfileService(db_session)

    # Récupérer profils actifs
    active_profiles = await service.get_active_profiles(days=7, limit=1000)

    for profile in active_profiles:
        # Récupérer transactions utilisateur
        transactions = get_user_transactions(profile.user_id, days=30)

        # Calculer moyennes par catégorie
        avg_by_category = calculate_average_spending(transactions)

        # Mettre à jour profil
        profile.habits.average_spending_by_category = avg_by_category
        profile.update_completeness()

        await service.update_profile(profile.user_id, profile)
```

## Performance

| Opération | Temps Moyen | Notes |
|-----------|-------------|-------|
| `get_profile()` | ~10ms | Index sur user_id |
| `create_profile()` | ~20ms | Avec commit DB |
| `update_profile()` | ~30ms | JSON updates |
| `get_active_profiles(1000)` | ~100ms | Index sur last_active |

## Tests

### Test unitaire

```python
import pytest
from conversation_service.services.user_profile import UserProfileService
from conversation_service.models.user_profile.entities import QueryPattern

@pytest.fixture
def profile_service(db_session):
    return UserProfileService(db_session)

async def test_create_and_get_profile(profile_service):
    # Créer
    profile = await profile_service.create_profile(user_id=999)
    assert profile.user_id == 999
    assert profile.profile_completeness == 0.0

    # Récupérer
    fetched = await profile_service.get_profile(user_id=999)
    assert fetched.user_id == 999

async def test_add_query_pattern(profile_service):
    profile = await profile_service.create_profile(user_id=999)

    # Ajouter pattern
    updated = await profile_service.add_query_pattern(
        user_id=999, pattern=QueryPattern.YOY_COMPARISONS
    )

    assert QueryPattern.YOY_COMPARISONS in updated.habits.frequent_query_patterns
    assert updated.habits.query_frequency["yoy_comparisons"] == 1
```

## Roadmap

### Phase 1 ✅ (Complété)
- Modèles Pydantic + SQLAlchemy
- CRUD operations
- Migration Alembic
- Calcul complétude

### Phase 2 (À venir)
- Background jobs analytics
- API endpoints REST
- Webhooks pour événements
- Dashboard admin

### Phase 3 (Futur)
- ML pour prédiction préférences
- Segmentation utilisateurs
- A/B testing recommandations
- Export données RGPD

## Support

Pour questions ou bugs, contacter l'équipe data ou créer une issue.

## Changelog

### v1.0.0 (2025-01-12)
- ✅ Modèles User Profile complets
- ✅ Service CRUD fonctionnel
- ✅ Migration Alembic
- ✅ Documentation complète
