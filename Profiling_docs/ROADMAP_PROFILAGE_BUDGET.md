# ROADMAP IMPLÉMENTATION - MODULE PROFILAGE BUDGÉTAIRE

**Version** : 1.0
**Date** : 2025-01-14
**Durée totale estimée** : 6 mois (24 sprints de 1 semaine)
**Équipe requise** : 2-3 devs backend + 1 dev frontend + 1 data analyst

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Phase 0 : Préparation (2 semaines)](#phase-0--préparation-2-semaines)
3. [Phase 1 : Fondations (8 semaines)](#phase-1--fondations-8-semaines)
4. [Phase 2 : Recommandations (6 semaines)](#phase-2--recommandations-6-semaines)
5. [Phase 3 : Objectifs & Saisonnalité (6 semaines)](#phase-3--objectifs--saisonnalité-6-semaines)
6. [Phase 4 : Optimisations & ML (4 semaines)](#phase-4--optimisations--ml-4-semaines)
7. [Tests de Validation Globaux](#tests-de-validation-globaux)
8. [Critères de Go/No-Go par Phase](#critères-de-gono-go-par-phase)

---

## Vue d'ensemble

### Objectifs du Projet

✅ Analyser automatiquement les transactions pour établir le profil financier utilisateur
✅ Détecter et catégoriser les charges (fixes, semi-fixes, variables)
✅ Générer des recommandations personnalisées pour optimiser les dépenses
✅ Aider l'utilisateur à atteindre des objectifs d'épargne spécifiques
✅ Identifier les tendances saisonnières et patterns de dépenses

### Principes de Développement

- **Incrémental** : Chaque sprint délivre une fonctionnalité testable
- **Testé** : 80%+ couverture code + tests end-to-end par étape
- **Validé** : Critères de validation mesurables avant passage phase suivante
- **User-centric** : Validation beta users entre chaque phase
- **Réversible** : Feature flags pour rollback rapide si problème

### Architecture Cible

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (Existant)                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ User Service   │  │ Conversation│  │ Budget Profiling│ (Nouveau)
│ (Existant)     │  │ Service     │  │ Service         │
└────────────────┘  │ (Extensions)│  └─────────────────┘
                    └─────────────┘           │
                            │                 │
        ┌───────────────────┼─────────────────┼──────────┐
        │                   │                 │          │
┌───────▼────────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌▼──────────┐
│ Search Service │  │ Metric      │  │ Recommendation│ │ Goal      │
│ (Existant)     │  │ Service     │  │ Engine        │ │ Tracking  │
└────────────────┘  │ (Extensions)│  │ (Nouveau)     │ │ (Nouveau) │
                    └─────────────┘  └───────────────┘ └───────────┘
                            │
                    ┌───────▼───────┐
                    │  PostgreSQL   │
                    │  + 5 nouvelles│
                    │  tables       │
                    └───────────────┘
```

---

## Phase 0 : Préparation (2 semaines)

**Objectif** : Mettre en place l'infrastructure de base et les outils de développement

### Sprint 0.1 (Semaine 1) : Setup Environnement

#### Tâches

1. **Création structure projet**
   ```bash
   mkdir -p budget_profiling_service/{api,core,models,services,tests}
   mkdir -p budget_profiling_service/scripts/{migrations,seeds}
   ```

2. **Configuration service FastAPI**
   - Créer `budget_profiling_service/main.py`
   - Configuration environnement (.env)
   - Setup logging structuré
   - Health check endpoint

3. **Setup base de données**
   - Créer schéma `budget_profiling`
   - Configurer Alembic pour migrations
   - Script init DB : `scripts/init_budget_db.sh`

4. **Configuration CI/CD**
   - Tests automatisés (pytest)
   - Linting (black, flake8, mypy)
   - Pre-commit hooks
   - GitHub Actions workflow

#### Livrables

- [ ] Service FastAPI démarre sur port 8006
- [ ] Endpoint `GET /health` retourne 200
- [ ] Pipeline CI/CD opérationnel
- [ ] Documentation setup dans README.md

#### Tests de Validation

```bash
# Test 1: Service démarre
curl http://localhost:8006/health
# Expected: {"status": "healthy", "service": "budget_profiling"}

# Test 2: CI/CD fonctionne
git push origin feature/budget-profiling-setup
# Expected: Tests passent, build succeed

# Test 3: Connexion DB
python -c "from budget_profiling_service.db import engine; print(engine.connect())"
# Expected: Connection successful
```

---

### Sprint 0.2 (Semaine 2) : Modèles de Données

#### Tâches

1. **Création modèles SQLAlchemy**

```python
# budget_profiling_service/models/budget_profile.py

class UserBudgetProfile(Base, TimestampMixin):
    """Profil budgétaire utilisateur"""
    __tablename__ = "user_budget_profile"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Segmentation
    user_segment = Column(String(50))  # 'budget_serré', 'équilibré', 'confortable'
    behavioral_pattern = Column(String(50))

    # Métriques moyennes (3 derniers mois)
    avg_monthly_income = Column(Numeric(10, 2))
    avg_monthly_expenses = Column(Numeric(10, 2))
    avg_monthly_savings = Column(Numeric(10, 2))
    savings_rate = Column(Numeric(5, 2))  # %

    # Répartition charges
    fixed_charges_total = Column(Numeric(10, 2))
    semi_fixed_charges_total = Column(Numeric(10, 2))
    variable_charges_total = Column(Numeric(10, 2))

    # Reste à vivre
    remaining_to_live = Column(Numeric(10, 2))

    # Métadonnées
    profile_completeness = Column(Numeric(3, 2))  # 0.0 - 1.0
    last_analyzed_at = Column(DateTime(timezone=True))
```

```python
# budget_profiling_service/models/fixed_charge.py

class FixedCharge(Base, TimestampMixin):
    """Charge fixe détectée"""
    __tablename__ = "fixed_charges"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Identification
    merchant_name = Column(String(255))
    category = Column(String(100))

    # Caractéristiques récurrence
    avg_amount = Column(Numeric(10, 2))
    amount_variance = Column(Numeric(5, 2))  # %
    recurrence_day = Column(Integer)  # Jour du mois (1-31)
    recurrence_confidence = Column(Numeric(3, 2))  # 0.0 - 1.0

    # Statut
    validated_by_user = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Historique
    first_detected_date = Column(Date)
    last_transaction_date = Column(Date)
    transaction_count = Column(Integer)

    __table_args__ = (
        UniqueConstraint('user_id', 'merchant_name', name='uq_user_merchant'),
    )
```

```python
# budget_profiling_service/models/savings_goal.py

class SavingsGoal(Base, TimestampMixin):
    """Objectif d'épargne"""
    __tablename__ = "savings_goals"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Objectif
    goal_name = Column(String(255))  # "Vacances été", "Réserve urgence"
    target_amount = Column(Numeric(10, 2))
    target_date = Column(Date)

    # Progression
    current_amount = Column(Numeric(10, 2), default=0)
    monthly_contribution = Column(Numeric(10, 2))

    # Plan épargne
    suggested_categories = Column(JSON)  # [{"category": "loisirs", "reduction_pct": 20}]

    # Statut
    status = Column(String(50), default='active')  # 'active', 'completed', 'abandoned'
```

```python
# budget_profiling_service/models/recommendation.py

class BudgetRecommendation(Base, TimestampMixin):
    """Recommandation budgétaire"""
    __tablename__ = "budget_recommendations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Type
    recommendation_type = Column(String(100))
    priority = Column(String(20))  # 'high', 'medium', 'low'

    # Contenu
    title = Column(String(255))
    description = Column(Text)
    estimated_savings = Column(Numeric(10, 2))

    # Actions
    actions = Column(JSON)  # [{"type": "create_alert", "parameters": {...}}]

    # Feedback
    status = Column(String(50), default='pending')
    user_feedback = Column(Text)
    feedback_timestamp = Column(DateTime(timezone=True))

    # Efficacité
    actual_impact = Column(Numeric(10, 2))

    # Expiration
    expires_at = Column(DateTime(timezone=True))
```

```python
# budget_profiling_service/models/seasonal_pattern.py

class SeasonalPattern(Base, TimestampMixin):
    """Pattern saisonnier détecté"""
    __tablename__ = "seasonal_patterns"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Période
    month = Column(Integer)  # 1-12
    pattern_type = Column(String(50))  # 'high_spending', 'multiple_bills', 'vacation'

    # Métriques
    avg_amount = Column(Numeric(10, 2))
    variance_vs_avg = Column(Numeric(5, 2))  # %

    # Description
    description = Column(Text)
    key_expenses = Column(JSON)  # [{"merchant": "...", "amount": ...}]

    # Confiance
    confidence = Column(Numeric(3, 2))  # 0.0 - 1.0
    years_data = Column(Integer)

    __table_args__ = (
        UniqueConstraint('user_id', 'month', 'pattern_type', name='uq_user_month_pattern'),
    )
```

2. **Migrations Alembic**
   ```bash
   alembic revision --autogenerate -m "Create budget profiling tables"
   alembic upgrade head
   ```

3. **Scripts de seed données test**
   ```python
   # scripts/seed_budget_test_data.py
   # Générer 3 utilisateurs test avec 6 mois d'historique
   ```

#### Livrables

- [ ] 5 modèles SQLAlchemy créés
- [ ] Migrations exécutées avec succès
- [ ] Script seed génère données test
- [ ] Relations FK validées

#### Tests de Validation

```python
# tests/test_models.py

def test_user_budget_profile_creation(db_session):
    """Test création profil budgétaire"""
    profile = UserBudgetProfile(
        user_id=1,
        user_segment='équilibré',
        avg_monthly_income=3000,
        avg_monthly_expenses=2400,
        savings_rate=20.0
    )
    db_session.add(profile)
    db_session.commit()

    assert profile.id is not None
    assert profile.savings_rate == 20.0

def test_fixed_charge_unique_constraint(db_session):
    """Test contrainte unicité user_id + merchant_name"""
    charge1 = FixedCharge(user_id=1, merchant_name="EDF", avg_amount=80)
    db_session.add(charge1)
    db_session.commit()

    charge2 = FixedCharge(user_id=1, merchant_name="EDF", avg_amount=85)
    db_session.add(charge2)

    with pytest.raises(IntegrityError):
        db_session.commit()

def test_savings_goal_relationships(db_session):
    """Test relations FK"""
    goal = SavingsGoal(
        user_id=1,
        goal_name="Vacances",
        target_amount=2000,
        target_date=date(2025, 7, 1)
    )
    db_session.add(goal)
    db_session.commit()

    assert goal.user is not None  # FK vers users fonctionne
```

```bash
# Test migration rollback
alembic downgrade -1
alembic upgrade head
# Expected: Pas d'erreur

# Test seed data
python scripts/seed_budget_test_data.py
psql -d harena -c "SELECT COUNT(*) FROM user_budget_profile;"
# Expected: 3 rows
```

---

## Phase 1 : Fondations (8 semaines)

**Objectif** : Infrastructure de base + détection charges fixes + profil budgétaire

---

### Sprint 1.1 (Semaine 3) : Service de Récupération Transactions

#### Tâches

1. **Créer module récupération transactions**

```python
# budget_profiling_service/services/transaction_service.py

from typing import List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import select, and_
from db_service.models.sync import RawTransaction

class TransactionService:
    """Service récupération et préparation transactions pour analyse"""

    def __init__(self, db_session):
        self.db = db_session

    async def get_user_transactions(
        self,
        user_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        months_back: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Récupère les transactions d'un utilisateur sur une période

        Args:
            user_id: ID utilisateur
            start_date: Date début (optionnel)
            end_date: Date fin (optionnel)
            months_back: Nombre de mois en arrière si dates non fournies

        Returns:
            Liste de transactions formatées pour analyse
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30 * months_back)
        if not end_date:
            end_date = datetime.now()

        query = select(RawTransaction).where(
            and_(
                RawTransaction.user_id == user_id,
                RawTransaction.date >= start_date,
                RawTransaction.date <= end_date,
                RawTransaction.deleted == False
            )
        ).order_by(RawTransaction.date.desc())

        result = await self.db.execute(query)
        transactions = result.scalars().all()

        # Formater pour analyse
        formatted = []
        for tx in transactions:
            formatted.append({
                'id': tx.bridge_transaction_id,
                'user_id': tx.user_id,
                'merchant_name': tx.merchant_name or tx.clean_description,
                'amount': float(tx.amount),
                'date': tx.date,
                'category': self._get_category_name(tx.category_id),
                'operation_type': tx.operation_type,
                'is_debit': tx.amount < 0,
                'is_credit': tx.amount > 0
            })

        return formatted

    def _get_category_name(self, category_id: int) -> str:
        """Récupère nom catégorie depuis category_id"""
        # TODO: Cache categories in memory
        if not category_id:
            return 'uncategorized'

        from db_service.models.sync import Category
        category = self.db.query(Category).filter_by(category_id=category_id).first()
        return category.category_name if category else 'uncategorized'

    async def get_monthly_aggregates(
        self,
        user_id: int,
        months: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Agrégations mensuelles (revenus, dépenses)

        Returns:
            [
                {
                    'month': '2025-01',
                    'total_income': 3000.0,
                    'total_expenses': 2400.0,
                    'net_cashflow': 600.0,
                    'transaction_count': 145
                },
                ...
            ]
        """
        transactions = await self.get_user_transactions(
            user_id,
            months_back=months
        )

        # Grouper par mois
        monthly_data = {}
        for tx in transactions:
            month_key = tx['date'].strftime('%Y-%m')

            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': month_key,
                    'total_income': 0.0,
                    'total_expenses': 0.0,
                    'transaction_count': 0
                }

            if tx['is_credit']:
                monthly_data[month_key]['total_income'] += tx['amount']
            else:
                monthly_data[month_key]['total_expenses'] += abs(tx['amount'])

            monthly_data[month_key]['transaction_count'] += 1

        # Calculer net cashflow
        result = []
        for month, data in monthly_data.items():
            data['net_cashflow'] = data['total_income'] - data['total_expenses']
            result.append(data)

        return sorted(result, key=lambda x: x['month'], reverse=True)
```

2. **Tests unitaires service**

```python
# tests/services/test_transaction_service.py

import pytest
from datetime import datetime, timedelta
from budget_profiling_service.services.transaction_service import TransactionService

@pytest.fixture
def transaction_service(db_session):
    return TransactionService(db_session)

@pytest.fixture
def sample_transactions(db_session):
    """Créer 20 transactions test sur 3 mois"""
    from db_service.models.sync import RawTransaction

    base_date = datetime.now() - timedelta(days=60)
    transactions = []

    for i in range(20):
        tx = RawTransaction(
            bridge_transaction_id=1000 + i,
            user_id=100,
            account_id=1,
            merchant_name=f"Merchant_{i % 5}",
            amount=-50.0 if i % 3 == 0 else 1500.0,
            date=base_date + timedelta(days=i * 3),
            operation_type='card' if i % 3 == 0 else 'transfer'
        )
        transactions.append(tx)
        db_session.add(tx)

    db_session.commit()
    return transactions

async def test_get_user_transactions(transaction_service, sample_transactions):
    """Test récupération transactions"""
    transactions = await transaction_service.get_user_transactions(
        user_id=100,
        months_back=3
    )

    assert len(transactions) == 20
    assert all('merchant_name' in tx for tx in transactions)
    assert all('amount' in tx for tx in transactions)

async def test_get_monthly_aggregates(transaction_service, sample_transactions):
    """Test agrégations mensuelles"""
    aggregates = await transaction_service.get_monthly_aggregates(
        user_id=100,
        months=3
    )

    assert len(aggregates) >= 2  # Au moins 2 mois
    assert all('total_income' in agg for agg in aggregates)
    assert all('total_expenses' in agg for agg in aggregates)
    assert all('net_cashflow' in agg for agg in aggregates)

    # Vérifier calculs
    first_month = aggregates[0]
    expected_net = first_month['total_income'] - first_month['total_expenses']
    assert abs(first_month['net_cashflow'] - expected_net) < 0.01

async def test_filter_by_date_range(transaction_service, sample_transactions):
    """Test filtrage par plage de dates"""
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    transactions = await transaction_service.get_user_transactions(
        user_id=100,
        start_date=start_date,
        end_date=end_date
    )

    assert all(start_date <= tx['date'] <= end_date for tx in transactions)
```

#### Livrables

- [ ] `TransactionService` opérationnel
- [ ] Tests unitaires 100% couverture
- [ ] Documentation API (docstrings)

#### Tests de Validation

```bash
# Test intégration
pytest tests/services/test_transaction_service.py -v
# Expected: All tests pass

# Test performance (10K transactions)
python scripts/perf_test_transaction_service.py
# Expected: get_user_transactions < 300ms

# Test edge cases
pytest tests/services/test_transaction_service.py::test_empty_user -v
pytest tests/services/test_transaction_service.py::test_future_dates -v
# Expected: Gère correctement cas limites
```

---

### Sprint 1.2 (Semaine 4) : Algorithme Détection Charges Fixes

#### Tâches

1. **Créer service détection charges fixes**

```python
# budget_profiling_service/services/fixed_charge_detector.py

import logging
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DetectedCharge:
    """Charge fixe détectée"""
    merchant_name: str
    category: str
    avg_amount: float
    amount_variance: float  # %
    recurrence_day: int
    recurrence_confidence: float  # 0.0 - 1.0
    transaction_count: int
    first_date: datetime
    last_date: datetime
    monthly_occurrences: int

class FixedChargeDetector:
    """Détection automatique des charges fixes"""

    # Catégories éligibles aux charges fixes
    ELIGIBLE_CATEGORIES = [
        'rent', 'utilities', 'insurance', 'loan', 'childcare',
        'phone', 'internet', 'subscription', 'membership'
    ]

    # Seuils de détection
    MIN_MONTHS = 3
    MAX_AMOUNT_VARIANCE = 10.0  # %
    MAX_DAY_VARIANCE = 5  # jours
    MIN_CONFIDENCE = 0.7

    def __init__(self, transaction_service):
        self.transaction_service = transaction_service

    async def detect(
        self,
        user_id: int,
        analysis_months: int = 6
    ) -> List[DetectedCharge]:
        """
        Détecte les charges fixes d'un utilisateur

        Critères:
        - Récurrence mensuelle stable (±5 jours autour de la même date)
        - Montant stable (variance ≤10%)
        - Minimum 3 occurrences sur la période
        - Catégories éligibles

        Args:
            user_id: ID utilisateur
            analysis_months: Nombre de mois à analyser (défaut 6)

        Returns:
            Liste de charges fixes détectées
        """
        logger.info(f"Detecting fixed charges for user {user_id} over {analysis_months} months")

        # 1. Récupérer transactions (débits uniquement)
        transactions = await self.transaction_service.get_user_transactions(
            user_id=user_id,
            months_back=analysis_months
        )

        debits = [tx for tx in transactions if tx['is_debit']]

        if not debits:
            logger.warning(f"No debit transactions found for user {user_id}")
            return []

        # 2. Convertir en DataFrame pour analyse
        df = pd.DataFrame(debits)
        df['amount_abs'] = df['amount'].abs()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.to_period('M')

        # 3. Grouper par marchand
        detected_charges = []

        for merchant_name, merchant_txs in df.groupby('merchant_name'):
            charge = self._analyze_merchant_recurrence(
                merchant_name,
                merchant_txs,
                analysis_months
            )

            if charge and charge.recurrence_confidence >= self.MIN_CONFIDENCE:
                detected_charges.append(charge)

        logger.info(f"Detected {len(detected_charges)} fixed charges for user {user_id}")

        return sorted(detected_charges, key=lambda c: c.recurrence_confidence, reverse=True)

    def _analyze_merchant_recurrence(
        self,
        merchant_name: str,
        transactions: pd.DataFrame,
        total_months: int
    ) -> DetectedCharge | None:
        """
        Analyse la récurrence d'un marchand spécifique

        Returns:
            DetectedCharge si critères remplis, None sinon
        """
        # Vérifier nombre de mois avec transaction
        monthly_count = transactions['month'].nunique()

        if monthly_count < self.MIN_MONTHS:
            return None

        # Vérifier stabilité montant
        amounts = transactions['amount_abs']
        avg_amount = amounts.mean()
        amount_std = amounts.std()
        amount_variance_pct = (amount_std / avg_amount * 100) if avg_amount > 0 else 100

        if amount_variance_pct > self.MAX_AMOUNT_VARIANCE:
            return None

        # Vérifier régularité jour du mois
        days = transactions['day_of_month']
        avg_day = days.mean()
        day_std = days.std()

        if day_std > self.MAX_DAY_VARIANCE:
            return None

        # Vérifier catégorie éligible
        category = transactions['category'].iloc[0] if len(transactions) > 0 else 'uncategorized'

        # Bonus confiance si catégorie éligible
        category_bonus = 0.2 if category in self.ELIGIBLE_CATEGORIES else 0.0

        # Calculer score de confiance
        confidence = self._calculate_confidence(
            monthly_count=monthly_count,
            total_months=total_months,
            amount_variance_pct=amount_variance_pct,
            day_variance=day_std,
            category_bonus=category_bonus
        )

        return DetectedCharge(
            merchant_name=merchant_name,
            category=category,
            avg_amount=float(avg_amount),
            amount_variance=float(amount_variance_pct),
            recurrence_day=int(round(avg_day)),
            recurrence_confidence=confidence,
            transaction_count=len(transactions),
            first_date=transactions['date'].min(),
            last_date=transactions['date'].max(),
            monthly_occurrences=monthly_count
        )

    def _calculate_confidence(
        self,
        monthly_count: int,
        total_months: int,
        amount_variance_pct: float,
        day_variance: float,
        category_bonus: float
    ) -> float:
        """
        Calcule le score de confiance de la détection

        Formule:
        - 50% poids: récurrence (nb_mois / total_mois)
        - 30% poids: stabilité montant (1 - variance/100)
        - 20% poids: régularité date (1 - day_variance/30)
        + Bonus catégorie éligible (+0.2)

        Returns:
            Score entre 0.0 et 1.0
        """
        recurrence_score = monthly_count / total_months
        amount_stability_score = max(0, 1 - (amount_variance_pct / 100))
        day_stability_score = max(0, 1 - (day_variance / 30))

        confidence = (
            recurrence_score * 0.5 +
            amount_stability_score * 0.3 +
            day_stability_score * 0.2 +
            category_bonus
        )

        return min(confidence, 1.0)
```

2. **Tests unitaires détecteur**

```python
# tests/services/test_fixed_charge_detector.py

import pytest
from datetime import datetime, timedelta
from budget_profiling_service.services.fixed_charge_detector import FixedChargeDetector

@pytest.fixture
def fixed_charge_detector(transaction_service):
    return FixedChargeDetector(transaction_service)

@pytest.fixture
def perfect_fixed_charge_data(db_session):
    """
    Créer transactions parfaitement récurrentes:
    - EDF: 80€ le 5 de chaque mois (6 mois)
    - Loyer: 750€ le 1er de chaque mois (6 mois)
    """
    from db_service.models.sync import RawTransaction

    base_date = datetime.now() - timedelta(days=180)

    for month in range(6):
        # EDF - 5 de chaque mois
        edf_date = base_date + timedelta(days=30 * month + 5)
        db_session.add(RawTransaction(
            bridge_transaction_id=2000 + month,
            user_id=200,
            account_id=1,
            merchant_name="EDF",
            amount=-80.0,
            date=edf_date,
            category_id=10,  # utilities
            operation_type='direct_debit'
        ))

        # Loyer - 1er de chaque mois
        rent_date = base_date + timedelta(days=30 * month + 1)
        db_session.add(RawTransaction(
            bridge_transaction_id=2100 + month,
            user_id=200,
            account_id=1,
            merchant_name="Immobilier Dupont",
            amount=-750.0,
            date=rent_date,
            category_id=20,  # rent
            operation_type='transfer'
        ))

    db_session.commit()

async def test_detect_perfect_fixed_charges(
    fixed_charge_detector,
    perfect_fixed_charge_data
):
    """Test détection charges fixes parfaites"""
    charges = await fixed_charge_detector.detect(user_id=200, analysis_months=6)

    assert len(charges) == 2

    # Vérifier EDF
    edf = next(c for c in charges if c.merchant_name == "EDF")
    assert edf.avg_amount == pytest.approx(80.0, abs=0.01)
    assert edf.recurrence_day == 5
    assert edf.recurrence_confidence >= 0.9
    assert edf.transaction_count == 6
    assert edf.amount_variance < 1.0  # Très stable

    # Vérifier Loyer
    rent = next(c for c in charges if "Immobilier" in c.merchant_name)
    assert rent.avg_amount == pytest.approx(750.0, abs=0.01)
    assert rent.recurrence_day == 1
    assert rent.recurrence_confidence >= 0.9

@pytest.fixture
def variable_charge_data(db_session):
    """Transactions variables (ne doivent PAS être détectées)"""
    from db_service.models.sync import RawTransaction

    base_date = datetime.now() - timedelta(days=180)

    # Carrefour - montants très variables, jours aléatoires
    for i in range(10):
        random_day = (i * 7 + i % 5) % 30
        random_amount = -50.0 - (i * 15.5)  # 50, 65.5, 81, ...

        db_session.add(RawTransaction(
            bridge_transaction_id=3000 + i,
            user_id=201,
            account_id=1,
            merchant_name="Carrefour",
            amount=random_amount,
            date=base_date + timedelta(days=random_day + i * 18),
            category_id=30,  # groceries
            operation_type='card'
        ))

    db_session.commit()

async def test_no_detection_variable_charges(
    fixed_charge_detector,
    variable_charge_data
):
    """Test que les charges variables ne sont PAS détectées"""
    charges = await fixed_charge_detector.detect(user_id=201, analysis_months=6)

    # Carrefour ne doit pas être détecté (variance trop élevée)
    carrefour_charges = [c for c in charges if c.merchant_name == "Carrefour"]
    assert len(carrefour_charges) == 0

@pytest.fixture
def insufficient_data(db_session):
    """Seulement 2 mois de données (insuffisant)"""
    from db_service.models.sync import RawTransaction

    base_date = datetime.now() - timedelta(days=60)

    for month in range(2):
        db_session.add(RawTransaction(
            bridge_transaction_id=4000 + month,
            user_id=202,
            account_id=1,
            merchant_name="Netflix",
            amount=-15.99,
            date=base_date + timedelta(days=30 * month + 10),
            category_id=40,  # subscription
            operation_type='card'
        ))

    db_session.commit()

async def test_insufficient_months(
    fixed_charge_detector,
    insufficient_data
):
    """Test rejet si moins de 3 mois de données"""
    charges = await fixed_charge_detector.detect(user_id=202, analysis_months=6)

    # Netflix ne doit pas être détecté (seulement 2 mois)
    assert len(charges) == 0

async def test_confidence_calculation(fixed_charge_detector):
    """Test calcul score de confiance"""
    confidence = fixed_charge_detector._calculate_confidence(
        monthly_count=6,
        total_months=6,
        amount_variance_pct=2.0,  # Très stable
        day_variance=1.0,  # Très régulier
        category_bonus=0.2  # Catégorie éligible
    )

    # Score attendu: ~0.98 (presque parfait)
    assert confidence >= 0.95

    # Test avec données médiocres
    low_confidence = fixed_charge_detector._calculate_confidence(
        monthly_count=3,
        total_months=6,
        amount_variance_pct=8.0,
        day_variance=4.0,
        category_bonus=0.0
    )

    assert low_confidence < 0.75
```

#### Livrables

- [ ] `FixedChargeDetector` fonctionnel
- [ ] Tests unitaires 90%+ couverture
- [ ] Validation sur jeu de données réel (user test)

#### Tests de Validation

```bash
# Test unitaires
pytest tests/services/test_fixed_charge_detector.py -v
# Expected: All 5+ tests pass

# Test performance
python scripts/perf_test_detector.py --transactions=10000
# Expected: Detection complète < 2s

# Test sur données réelles (user 100)
python scripts/manual_test_detector.py --user_id=100
# Expected: Affiche charges détectées avec confiance
```

**Exemple sortie attendue** :
```
Charges fixes détectées pour user 100:

1. EDF (confidence: 0.94)
   - Montant moyen: 78.50€
   - Jour récurrence: 5
   - Variance: 3.2%
   - Occurrences: 6/6 mois

2. Loyer Immobilier (confidence: 0.92)
   - Montant moyen: 750.00€
   - Jour récurrence: 1
   - Variance: 0.0%
   - Occurrences: 6/6 mois
```

---

### Sprint 1.3 (Semaine 5) : Catégorisation des Charges

#### Tâches

1. **Service de catégorisation**

```python
# budget_profiling_service/services/charge_categorizer.py

from typing import List, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChargeType(Enum):
    """Types de charges"""
    FIXED = "fixed"  # Charges fixes
    SEMI_FIXED = "semi_fixed"  # Charges semi-fixes
    VARIABLE = "variable"  # Charges variables

class ChargeCategorizer:
    """Catégorisation automatique des charges"""

    # Mapping catégories → type de charge
    CATEGORY_MAPPING = {
        # Charges fixes
        'rent': ChargeType.FIXED,
        'loan': ChargeType.FIXED,
        'insurance': ChargeType.FIXED,
        'childcare': ChargeType.FIXED,
        'phone': ChargeType.FIXED,
        'internet': ChargeType.FIXED,

        # Charges semi-fixes
        'groceries': ChargeType.SEMI_FIXED,
        'utilities': ChargeType.SEMI_FIXED,
        'transport': ChargeType.SEMI_FIXED,
        'health': ChargeType.SEMI_FIXED,
        'maintenance': ChargeType.SEMI_FIXED,

        # Charges variables
        'entertainment': ChargeType.VARIABLE,
        'dining': ChargeType.VARIABLE,
        'shopping': ChargeType.VARIABLE,
        'travel': ChargeType.VARIABLE,
        'hobbies': ChargeType.VARIABLE
    }

    # Potentiel d'optimisation par type
    OPTIMIZATION_POTENTIAL = {
        ChargeType.FIXED: (0, 5),  # 0-5%
        ChargeType.SEMI_FIXED: (10, 30),  # 10-30%
        ChargeType.VARIABLE: (20, 50)  # 20-50%
    }

    def __init__(self, fixed_charge_detector, transaction_service):
        self.detector = fixed_charge_detector
        self.transaction_service = transaction_service

    async def categorize_user_charges(
        self,
        user_id: int,
        analysis_months: int = 6
    ) -> Dict[str, Dict]:
        """
        Catégorise toutes les charges d'un utilisateur

        Returns:
            {
                'fixed': {
                    'charges': [...],
                    'total_monthly_avg': 1200.0,
                    'optimization_potential': '0-5%'
                },
                'semi_fixed': {...},
                'variable': {...}
            }
        """
        logger.info(f"Categorizing charges for user {user_id}")

        # 1. Détecter charges fixes
        fixed_charges = await self.detector.detect(user_id, analysis_months)

        # 2. Récupérer toutes transactions
        all_transactions = await self.transaction_service.get_user_transactions(
            user_id, months_back=analysis_months
        )

        debits = [tx for tx in all_transactions if tx['is_debit']]

        # 3. Identifier marchands fixes
        fixed_merchants = {charge.merchant_name for charge in fixed_charges}

        # 4. Catégoriser transactions restantes
        semi_fixed_txs = []
        variable_txs = []

        for tx in debits:
            if tx['merchant_name'] in fixed_merchants:
                continue  # Déjà catégorisé comme fixe

            charge_type = self._classify_transaction(tx)

            if charge_type == ChargeType.SEMI_FIXED:
                semi_fixed_txs.append(tx)
            else:
                variable_txs.append(tx)

        # 5. Calculer totaux mensuels moyens
        fixed_total = sum(c.avg_amount for c in fixed_charges)
        semi_fixed_total = self._calculate_monthly_avg(semi_fixed_txs, analysis_months)
        variable_total = self._calculate_monthly_avg(variable_txs, analysis_months)

        return {
            'fixed': {
                'charges': fixed_charges,
                'total_monthly_avg': fixed_total,
                'optimization_potential': f"{self.OPTIMIZATION_POTENTIAL[ChargeType.FIXED][0]}-{self.OPTIMIZATION_POTENTIAL[ChargeType.FIXED][1]}%",
                'charge_count': len(fixed_charges)
            },
            'semi_fixed': {
                'transactions': semi_fixed_txs,
                'total_monthly_avg': semi_fixed_total,
                'optimization_potential': f"{self.OPTIMIZATION_POTENTIAL[ChargeType.SEMI_FIXED][0]}-{self.OPTIMIZATION_POTENTIAL[ChargeType.SEMI_FIXED][1]}%",
                'transaction_count': len(semi_fixed_txs)
            },
            'variable': {
                'transactions': variable_txs,
                'total_monthly_avg': variable_total,
                'optimization_potential': f"{self.OPTIMIZATION_POTENTIAL[ChargeType.VARIABLE][0]}-{self.OPTIMIZATION_POTENTIAL[ChargeType.VARIABLE][1]}%",
                'transaction_count': len(variable_txs)
            },
            'summary': {
                'total_monthly_expenses': fixed_total + semi_fixed_total + variable_total,
                'fixed_percentage': (fixed_total / (fixed_total + semi_fixed_total + variable_total)) * 100 if (fixed_total + semi_fixed_total + variable_total) > 0 else 0,
                'semi_fixed_percentage': (semi_fixed_total / (fixed_total + semi_fixed_total + variable_total)) * 100 if (fixed_total + semi_fixed_total + variable_total) > 0 else 0,
                'variable_percentage': (variable_total / (fixed_total + semi_fixed_total + variable_total)) * 100 if (fixed_total + semi_fixed_total + variable_total) > 0 else 0
            }
        }

    def _classify_transaction(self, transaction: Dict) -> ChargeType:
        """
        Classifie une transaction selon sa catégorie

        Args:
            transaction: Dict avec 'category' key

        Returns:
            ChargeType (SEMI_FIXED ou VARIABLE)
        """
        category = transaction.get('category', 'uncategorized')

        # Chercher dans mapping
        return self.CATEGORY_MAPPING.get(category, ChargeType.VARIABLE)

    def _calculate_monthly_avg(
        self,
        transactions: List[Dict],
        months: int
    ) -> float:
        """Calcule moyenne mensuelle des dépenses"""
        if not transactions:
            return 0.0

        total = sum(abs(tx['amount']) for tx in transactions)
        return total / months
```

2. **Tests de catégorisation**

```python
# tests/services/test_charge_categorizer.py

import pytest
from budget_profiling_service.services.charge_categorizer import (
    ChargeCategorizer,
    ChargeType
)

@pytest.fixture
def charge_categorizer(fixed_charge_detector, transaction_service):
    return ChargeCategorizer(fixed_charge_detector, transaction_service)

@pytest.fixture
def mixed_charges_data(db_session):
    """
    Données mixtes:
    - 2 charges fixes (EDF, Loyer)
    - Transactions semi-fixes (Carrefour courses)
    - Transactions variables (restaurants, loisirs)
    """
    from db_service.models.sync import RawTransaction
    from datetime import datetime, timedelta

    base_date = datetime.now() - timedelta(days=180)

    # Charges fixes (6 mois)
    for month in range(6):
        # EDF - utilities (semi-fixe selon config, mais détecté comme fixe)
        db_session.add(RawTransaction(
            bridge_transaction_id=5000 + month,
            user_id=300,
            account_id=1,
            merchant_name="EDF",
            amount=-80.0,
            date=base_date + timedelta(days=30 * month + 5),
            category_id=10,  # utilities
            operation_type='direct_debit'
        ))

        # Loyer - rent (fixe)
        db_session.add(RawTransaction(
            bridge_transaction_id=5100 + month,
            user_id=300,
            account_id=1,
            merchant_name="Loyer",
            amount=-750.0,
            date=base_date + timedelta(days=30 * month + 1),
            category_id=20,  # rent
            operation_type='transfer'
        ))

    # Transactions semi-fixes (courses)
    for i in range(24):  # ~4 par mois
        db_session.add(RawTransaction(
            bridge_transaction_id=5200 + i,
            user_id=300,
            account_id=1,
            merchant_name="Carrefour",
            amount=-(50 + i * 5),  # Variable 50-165€
            date=base_date + timedelta(days=i * 7 + 3),
            category_id=30,  # groceries (semi-fixe)
            operation_type='card'
        ))

    # Transactions variables (restaurants)
    for i in range(12):  # 2 par mois
        db_session.add(RawTransaction(
            bridge_transaction_id=5300 + i,
            user_id=300,
            account_id=1,
            merchant_name=f"Restaurant_{i % 4}",
            amount=-(30 + i * 10),  # 30-140€
            date=base_date + timedelta(days=i * 15 + 7),
            category_id=40,  # dining (variable)
            operation_type='card'
        ))

    db_session.commit()

async def test_categorize_mixed_charges(charge_categorizer, mixed_charges_data):
    """Test catégorisation charges mixtes"""
    result = await charge_categorizer.categorize_user_charges(
        user_id=300,
        analysis_months=6
    )

    # Vérifier structure
    assert 'fixed' in result
    assert 'semi_fixed' in result
    assert 'variable' in result
    assert 'summary' in result

    # Vérifier charges fixes
    fixed = result['fixed']
    assert fixed['charge_count'] == 2  # EDF + Loyer
    assert fixed['total_monthly_avg'] == pytest.approx(830.0, abs=1.0)  # 80 + 750

    # Vérifier semi-fixes
    semi_fixed = result['semi_fixed']
    assert semi_fixed['transaction_count'] == 24  # Carrefour
    assert semi_fixed['total_monthly_avg'] > 0

    # Vérifier variables
    variable = result['variable']
    assert variable['transaction_count'] == 12  # Restaurants

    # Vérifier résumé
    summary = result['summary']
    assert summary['total_monthly_expenses'] > 0
    assert summary['fixed_percentage'] > 50  # Loyer pèse lourd
    assert summary['semi_fixed_percentage'] > 0
    assert summary['variable_percentage'] > 0

    # Somme des % = 100
    total_pct = (
        summary['fixed_percentage'] +
        summary['semi_fixed_percentage'] +
        summary['variable_percentage']
    )
    assert total_pct == pytest.approx(100.0, abs=0.1)

async def test_optimization_potential_mapping(charge_categorizer):
    """Test que les potentiels d'optimisation sont corrects"""
    assert charge_categorizer.OPTIMIZATION_POTENTIAL[ChargeType.FIXED] == (0, 5)
    assert charge_categorizer.OPTIMIZATION_POTENTIAL[ChargeType.SEMI_FIXED] == (10, 30)
    assert charge_categorizer.OPTIMIZATION_POTENTIAL[ChargeType.VARIABLE] == (20, 50)

def test_transaction_classification(charge_categorizer):
    """Test classification transactions individuelles"""
    # Test fixe
    tx_rent = {'category': 'rent', 'amount': -750}
    assert charge_categorizer._classify_transaction(tx_rent) == ChargeType.FIXED

    # Test semi-fixe
    tx_groceries = {'category': 'groceries', 'amount': -100}
    assert charge_categorizer._classify_transaction(tx_groceries) == ChargeType.SEMI_FIXED

    # Test variable
    tx_dining = {'category': 'dining', 'amount': -50}
    assert charge_categorizer._classify_transaction(tx_dining) == ChargeType.VARIABLE

    # Test catégorie inconnue (défaut = variable)
    tx_unknown = {'category': 'unknown_category', 'amount': -30}
    assert charge_categorizer._classify_transaction(tx_unknown) == ChargeType.VARIABLE
```

#### Livrables

- [ ] `ChargeCategorizer` opérationnel
- [ ] Mapping catégories complet (30+ catégories)
- [ ] Tests unitaires 85%+ couverture
- [ ] Documentation potentiels d'optimisation

#### Tests de Validation

```bash
# Tests unitaires
pytest tests/services/test_charge_categorizer.py -v
# Expected: All tests pass

# Test sur données réelles
python scripts/test_categorization.py --user_id=100
# Expected: Affiche répartition charges

# Exemple sortie attendue:
# Répartition des charges (user 100):
# - Fixes: 830€/mois (55%) - Optimisation: 0-5%
# - Semi-fixes: 450€/mois (30%) - Optimisation: 10-30%
# - Variables: 220€/mois (15%) - Optimisation: 20-50%
# TOTAL: 1500€/mois
```

---

### Sprint 1.4 (Semaine 6) : Calcul Profil Budgétaire

#### Tâches

1. **Service calcul profil**

```python
# budget_profiling_service/services/budget_profile_calculator.py

from typing import Dict, Any
from datetime import datetime
import logging
from budget_profiling_service.models.budget_profile import UserBudgetProfile

logger = logging.getLogger(__name__)

class BudgetProfileCalculator:
    """Calcul du profil budgétaire complet"""

    # Seuils segmentation utilisateur
    SEGMENTS = {
        'budget_serré': lambda ratio: ratio > 0.9,  # >90% dépenses/revenus
        'équilibré': lambda ratio: 0.7 <= ratio <= 0.9,  # 70-90%
        'confortable': lambda ratio: ratio < 0.7  # <70%
    }

    def __init__(
        self,
        transaction_service,
        charge_categorizer,
        db_session
    ):
        self.transaction_service = transaction_service
        self.categorizer = charge_categorizer
        self.db = db_session

    async def calculate_profile(
        self,
        user_id: int,
        analysis_months: int = 3
    ) -> UserBudgetProfile:
        """
        Calcule le profil budgétaire complet d'un utilisateur

        Calcule:
        - Revenus/dépenses moyens mensuels
        - Taux d'épargne
        - Répartition charges (fixes/semi-fixes/variables)
        - Reste à vivre
        - Segmentation utilisateur

        Args:
            user_id: ID utilisateur
            analysis_months: Nombre de mois pour calcul moyennes (défaut 3)

        Returns:
            UserBudgetProfile instance (non sauvegardé)
        """
        logger.info(f"Calculating budget profile for user {user_id}")

        # 1. Récupérer agrégations mensuelles
        monthly_agg = await self.transaction_service.get_monthly_aggregates(
            user_id,
            months=analysis_months
        )

        if not monthly_agg:
            raise ValueError(f"No transaction data for user {user_id}")

        # 2. Calculer moyennes
        avg_income = sum(m['total_income'] for m in monthly_agg) / len(monthly_agg)
        avg_expenses = sum(m['total_expenses'] for m in monthly_agg) / len(monthly_agg)
        avg_savings = avg_income - avg_expenses

        # 3. Catégoriser charges
        charge_breakdown = await self.categorizer.categorize_user_charges(
            user_id,
            analysis_months=analysis_months
        )

        fixed_total = charge_breakdown['fixed']['total_monthly_avg']
        semi_fixed_total = charge_breakdown['semi_fixed']['total_monthly_avg']
        variable_total = charge_breakdown['variable']['total_monthly_avg']

        # 4. Calculer métriques
        savings_rate = (avg_savings / avg_income * 100) if avg_income > 0 else 0
        expense_ratio = avg_expenses / avg_income if avg_income > 0 else 0
        remaining_to_live = avg_income - fixed_total

        # 5. Déterminer segment
        user_segment = self._determine_segment(expense_ratio)

        # 6. Calculer complétude profil
        profile_completeness = self._calculate_completeness(
            has_income=avg_income > 0,
            has_expenses=avg_expenses > 0,
            has_fixed_charges=len(charge_breakdown['fixed']['charges']) > 0,
            months_data=len(monthly_agg)
        )

        # 7. Créer ou mettre à jour profil
        profile = await self._get_or_create_profile(user_id)

        # Mettre à jour données
        profile.user_segment = user_segment
        profile.avg_monthly_income = float(avg_income)
        profile.avg_monthly_expenses = float(avg_expenses)
        profile.avg_monthly_savings = float(avg_savings)
        profile.savings_rate = float(savings_rate)
        profile.fixed_charges_total = float(fixed_total)
        profile.semi_fixed_charges_total = float(semi_fixed_total)
        profile.variable_charges_total = float(variable_total)
        profile.remaining_to_live = float(remaining_to_live)
        profile.profile_completeness = profile_completeness
        profile.last_analyzed_at = datetime.now()

        logger.info(f"Profile calculated: segment={user_segment}, savings_rate={savings_rate:.1f}%")

        return profile

    def _determine_segment(self, expense_ratio: float) -> str:
        """Détermine le segment utilisateur selon ratio dépenses/revenus"""
        for segment, condition in self.SEGMENTS.items():
            if condition(expense_ratio):
                return segment
        return 'équilibré'  # Défaut

    def _calculate_completeness(
        self,
        has_income: bool,
        has_expenses: bool,
        has_fixed_charges: bool,
        months_data: int
    ) -> float:
        """
        Calcule score de complétude du profil (0.0 - 1.0)

        Critères:
        - Données revenus: 30%
        - Données dépenses: 30%
        - Charges fixes détectées: 20%
        - Historique suffisant (3+ mois): 20%
        """
        score = 0.0

        if has_income:
            score += 0.3
        if has_expenses:
            score += 0.3
        if has_fixed_charges:
            score += 0.2
        if months_data >= 3:
            score += 0.2

        return min(score, 1.0)

    async def _get_or_create_profile(self, user_id: int) -> UserBudgetProfile:
        """Récupère profil existant ou crée nouveau"""
        from sqlalchemy import select

        query = select(UserBudgetProfile).where(
            UserBudgetProfile.user_id == user_id
        )
        result = await self.db.execute(query)
        profile = result.scalar_one_or_none()

        if not profile:
            profile = UserBudgetProfile(user_id=user_id)
            self.db.add(profile)

        return profile

    async def save_profile(self, profile: UserBudgetProfile):
        """Sauvegarde le profil en base"""
        await self.db.commit()
        await self.db.refresh(profile)
        logger.info(f"Profile saved for user {profile.user_id}")
```

2. **Tests calcul profil**

```python
# tests/services/test_budget_profile_calculator.py

import pytest
from budget_profiling_service.services.budget_profile_calculator import BudgetProfileCalculator

@pytest.fixture
def profile_calculator(transaction_service, charge_categorizer, db_session):
    return BudgetProfileCalculator(
        transaction_service,
        charge_categorizer,
        db_session
    )

@pytest.fixture
async def complete_user_data(db_session):
    """
    Utilisateur avec profil complet:
    - Revenus: 3000€/mois (salaire)
    - Dépenses: 2400€/mois
    - Charges fixes: 830€ (loyer + EDF)
    - Épargne: 600€/mois (20%)
    """
    from db_service.models.sync import RawTransaction
    from datetime import datetime, timedelta

    user_id = 400
    base_date = datetime.now() - timedelta(days=90)

    # Revenus mensuels (3 mois)
    for month in range(3):
        # Salaire
        db_session.add(RawTransaction(
            bridge_transaction_id=6000 + month,
            user_id=user_id,
            account_id=1,
            merchant_name="Employeur SA",
            amount=3000.0,  # Crédit
            date=base_date + timedelta(days=30 * month + 28),
            category_id=100,  # salary
            operation_type='transfer'
        ))

        # Loyer (fixe)
        db_session.add(RawTransaction(
            bridge_transaction_id=6100 + month,
            user_id=user_id,
            account_id=1,
            merchant_name="Loyer",
            amount=-750.0,
            date=base_date + timedelta(days=30 * month + 1),
            category_id=20,  # rent
            operation_type='transfer'
        ))

        # EDF (fixe)
        db_session.add(RawTransaction(
            bridge_transaction_id=6200 + month,
            user_id=user_id,
            account_id=1,
            merchant_name="EDF",
            amount=-80.0,
            date=base_date + timedelta(days=30 * month + 5),
            category_id=10,  # utilities
            operation_type='direct_debit'
        ))

    # Autres dépenses (semi-fixes et variables) pour atteindre 2400€/mois
    # Courses: ~400€/mois
    for i in range(12):  # 4 par mois
        db_session.add(RawTransaction(
            bridge_transaction_id=6300 + i,
            user_id=user_id,
            account_id=1,
            merchant_name="Carrefour",
            amount=-100.0,
            date=base_date + timedelta(days=i * 7 + 3),
            category_id=30,  # groceries
            operation_type='card'
        ))

    # Restaurants + loisirs: ~370€/mois
    for i in range(9):  # 3 par mois
        db_session.add(RawTransaction(
            bridge_transaction_id=6400 + i,
            user_id=user_id,
            account_id=1,
            merchant_name=f"Restaurant_{i}",
            amount=-(40 + i * 10),
            date=base_date + timedelta(days=i * 10 + 7),
            category_id=40,  # dining
            operation_type='card'
        ))

    db_session.commit()
    return user_id

async def test_calculate_complete_profile(profile_calculator, complete_user_data):
    """Test calcul profil complet"""
    user_id = complete_user_data

    profile = await profile_calculator.calculate_profile(user_id, analysis_months=3)

    # Vérifier calculs
    assert profile.user_id == user_id
    assert profile.avg_monthly_income == pytest.approx(3000.0, abs=10.0)
    assert profile.avg_monthly_expenses == pytest.approx(2400.0, abs=100.0)
    assert profile.avg_monthly_savings == pytest.approx(600.0, abs=100.0)
    assert profile.savings_rate == pytest.approx(20.0, abs=5.0)

    # Vérifier répartition charges
    assert profile.fixed_charges_total == pytest.approx(830.0, abs=10.0)  # Loyer + EDF
    assert profile.semi_fixed_charges_total > 0
    assert profile.variable_charges_total > 0

    # Vérifier reste à vivre
    assert profile.remaining_to_live == pytest.approx(2170.0, abs=50.0)  # 3000 - 830

    # Vérifier segment
    assert profile.user_segment == 'équilibré'  # 80% dépenses/revenus

    # Vérifier complétude
    assert profile.profile_completeness >= 0.8  # Profil quasi-complet

    # Vérifier timestamp
    assert profile.last_analyzed_at is not None

async def test_segmentation_logic(profile_calculator):
    """Test logique de segmentation"""
    # Budget serré (>90%)
    assert profile_calculator._determine_segment(0.95) == 'budget_serré'

    # Équilibré (70-90%)
    assert profile_calculator._determine_segment(0.80) == 'équilibré'
    assert profile_calculator._determine_segment(0.70) == 'équilibré'

    # Confortable (<70%)
    assert profile_calculator._determine_segment(0.60) == 'confortable'
    assert profile_calculator._determine_segment(0.30) == 'confortable'

def test_completeness_calculation(profile_calculator):
    """Test calcul complétude profil"""
    # Profil complet
    score_full = profile_calculator._calculate_completeness(
        has_income=True,
        has_expenses=True,
        has_fixed_charges=True,
        months_data=3
    )
    assert score_full == 1.0

    # Profil partiel (pas de charges fixes détectées)
    score_partial = profile_calculator._calculate_completeness(
        has_income=True,
        has_expenses=True,
        has_fixed_charges=False,
        months_data=3
    )
    assert score_partial == 0.8

    # Profil incomplet (seulement 1 mois)
    score_incomplete = profile_calculator._calculate_completeness(
        has_income=True,
        has_expenses=False,
        has_fixed_charges=False,
        months_data=1
    )
    assert score_incomplete == 0.3

async def test_profile_persistence(profile_calculator, complete_user_data, db_session):
    """Test sauvegarde profil en DB"""
    user_id = complete_user_data

    # Calculer et sauvegarder
    profile = await profile_calculator.calculate_profile(user_id)
    await profile_calculator.save_profile(profile)

    # Vérifier sauvegarde
    assert profile.id is not None

    # Récupérer depuis DB
    from sqlalchemy import select
    from budget_profiling_service.models.budget_profile import UserBudgetProfile

    query = select(UserBudgetProfile).where(UserBudgetProfile.user_id == user_id)
    result = await db_session.execute(query)
    saved_profile = result.scalar_one()

    assert saved_profile.user_id == user_id
    assert saved_profile.avg_monthly_income == profile.avg_monthly_income
    assert saved_profile.user_segment == profile.user_segment

async def test_profile_update(profile_calculator, complete_user_data, db_session):
    """Test mise à jour profil existant"""
    user_id = complete_user_data

    # Premier calcul
    profile1 = await profile_calculator.calculate_profile(user_id)
    await profile_calculator.save_profile(profile1)
    profile1_id = profile1.id

    # Deuxième calcul (mise à jour)
    profile2 = await profile_calculator.calculate_profile(user_id)
    await profile_calculator.save_profile(profile2)

    # Vérifier que c'est le même profil (update, pas insert)
    assert profile2.id == profile1_id
```

#### Livrables

- [ ] `BudgetProfileCalculator` fonctionnel
- [ ] Logique segmentation validée
- [ ] Tests unitaires 90%+ couverture
- [ ] Profils sauvegardés en DB

#### Tests de Validation

```bash
# Tests unitaires
pytest tests/services/test_budget_profile_calculator.py -v
# Expected: All tests pass (7+ tests)

# Test calcul sur user réel
python scripts/calculate_profile.py --user_id=100
# Expected: Affiche profil complet

# Exemple sortie:
# Profil budgétaire (user 100):
# - Segment: équilibré
# - Revenus moyens: 3000€/mois
# - Dépenses moyennes: 2400€/mois
# - Épargne: 600€/mois (20%)
# - Charges fixes: 830€ (35%)
# - Charges semi-fixes: 450€ (19%)
# - Charges variables: 1120€ (47%)
# - Reste à vivre: 2170€
# - Complétude profil: 100%

# Vérifier DB
psql -d harena -c "SELECT * FROM user_budget_profile WHERE user_id=100;"
# Expected: 1 row avec données correctes
```

---

### Sprint 1.5-1.6 (Semaines 7-8) : API Endpoints & Interface Validation

#### Tâches Sprint 1.5 (Semaine 7)

1. **Créer API REST endpoints**

```python
# budget_profiling_service/api/budget_profile.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from pydantic import BaseModel
from datetime import datetime

from budget_profiling_service.services.budget_profile_calculator import BudgetProfileCalculator
from budget_profiling_service.services.fixed_charge_detector import FixedChargeDetector, DetectedCharge
from budget_profiling_service.models.budget_profile import UserBudgetProfile
from db_service.database import get_db

router = APIRouter(prefix="/api/v1/budget", tags=["Budget Profiling"])

# Response models
class FixedChargeResponse(BaseModel):
    merchant_name: str
    category: str
    avg_amount: float
    amount_variance: float
    recurrence_day: int
    recurrence_confidence: float
    transaction_count: int
    validated_by_user: bool = False

    class Config:
        from_attributes = True

class BudgetProfileResponse(BaseModel):
    user_id: int
    user_segment: str
    avg_monthly_income: float
    avg_monthly_expenses: float
    avg_monthly_savings: float
    savings_rate: float
    fixed_charges_total: float
    semi_fixed_charges_total: float
    variable_charges_total: float
    remaining_to_live: float
    profile_completeness: float
    last_analyzed_at: datetime

    class Config:
        from_attributes = True

class ChargeBreakdownResponse(BaseModel):
    fixed: dict
    semi_fixed: dict
    variable: dict
    summary: dict

# GET /api/v1/budget/profile/{user_id}
@router.get("/profile/{user_id}", response_model=BudgetProfileResponse)
async def get_budget_profile(
    user_id: int,
    recalculate: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Récupère le profil budgétaire d'un utilisateur

    Args:
        user_id: ID utilisateur
        recalculate: Force recalcul (défaut: utilise cache)

    Returns:
        Profil budgétaire complet
    """
    from sqlalchemy import select

    # Si pas de recalcul, chercher profil existant
    if not recalculate:
        query = select(UserBudgetProfile).where(UserBudgetProfile.user_id == user_id)
        result = await db.execute(query)
        profile = result.scalar_one_or_none()

        if profile:
            return profile

    # Calculer profil
    from budget_profiling_service.services.transaction_service import TransactionService
    from budget_profiling_service.services.charge_categorizer import ChargeCategorizer

    tx_service = TransactionService(db)
    detector = FixedChargeDetector(tx_service)
    categorizer = ChargeCategorizer(detector, tx_service)
    calculator = BudgetProfileCalculator(tx_service, categorizer, db)

    profile = await calculator.calculate_profile(user_id)
    await calculator.save_profile(profile)

    return profile

# GET /api/v1/budget/fixed-charges/{user_id}
@router.get("/fixed-charges/{user_id}", response_model=List[FixedChargeResponse])
async def get_fixed_charges(
    user_id: int,
    validated_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Récupère les charges fixes d'un utilisateur

    Args:
        user_id: ID utilisateur
        validated_only: Seulement charges validées par user

    Returns:
        Liste des charges fixes
    """
    from sqlalchemy import select, and_
    from budget_profiling_service.models.fixed_charge import FixedCharge

    conditions = [FixedCharge.user_id == user_id, FixedCharge.is_active == True]

    if validated_only:
        conditions.append(FixedCharge.validated_by_user == True)

    query = select(FixedCharge).where(and_(*conditions))
    result = await db.execute(query)
    charges = result.scalars().all()

    return charges

# POST /api/v1/budget/fixed-charges/{user_id}/detect
@router.post("/fixed-charges/{user_id}/detect", response_model=List[FixedChargeResponse])
async def detect_fixed_charges(
    user_id: int,
    months: int = 6,
    db: AsyncSession = Depends(get_db)
):
    """
    Détecte les charges fixes pour un utilisateur

    Args:
        user_id: ID utilisateur
        months: Nombre de mois à analyser (défaut 6)

    Returns:
        Liste des charges fixes détectées
    """
    from budget_profiling_service.services.transaction_service import TransactionService

    tx_service = TransactionService(db)
    detector = FixedChargeDetector(tx_service)

    detected = await detector.detect(user_id, analysis_months=months)

    # Sauvegarder en DB
    from budget_profiling_service.models.fixed_charge import FixedCharge

    saved_charges = []
    for charge in detected:
        # Vérifier si existe déjà
        from sqlalchemy import select, and_
        query = select(FixedCharge).where(
            and_(
                FixedCharge.user_id == user_id,
                FixedCharge.merchant_name == charge.merchant_name
            )
        )
        result = await db.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            # Mettre à jour
            existing.avg_amount = charge.avg_amount
            existing.amount_variance = charge.amount_variance
            existing.recurrence_day = charge.recurrence_day
            existing.recurrence_confidence = charge.recurrence_confidence
            existing.last_transaction_date = charge.last_date
            existing.transaction_count = charge.transaction_count
            saved_charges.append(existing)
        else:
            # Créer nouveau
            new_charge = FixedCharge(
                user_id=user_id,
                merchant_name=charge.merchant_name,
                category=charge.category,
                avg_amount=charge.avg_amount,
                amount_variance=charge.amount_variance,
                recurrence_day=charge.recurrence_day,
                recurrence_confidence=charge.recurrence_confidence,
                first_detected_date=charge.first_date.date(),
                last_transaction_date=charge.last_date.date(),
                transaction_count=charge.transaction_count
            )
            db.add(new_charge)
            saved_charges.append(new_charge)

    await db.commit()

    return saved_charges

# PUT /api/v1/budget/fixed-charges/{charge_id}/validate
@router.put("/fixed-charges/{charge_id}/validate")
async def validate_fixed_charge(
    charge_id: int,
    validated: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Valide ou rejette une charge fixe détectée

    Args:
        charge_id: ID charge fixe
        validated: True pour valider, False pour rejeter
    """
    from sqlalchemy import select
    from budget_profiling_service.models.fixed_charge import FixedCharge

    query = select(FixedCharge).where(FixedCharge.id == charge_id)
    result = await db.execute(query)
    charge = result.scalar_one_or_none()

    if not charge:
        raise HTTPException(status_code=404, detail="Fixed charge not found")

    charge.validated_by_user = validated
    charge.is_active = validated

    await db.commit()

    return {"status": "success", "validated": validated}

# GET /api/v1/budget/breakdown/{user_id}
@router.get("/breakdown/{user_id}", response_model=ChargeBreakdownResponse)
async def get_charge_breakdown(
    user_id: int,
    months: int = 6,
    db: AsyncSession = Depends(get_db)
):
    """
    Récupère la répartition détaillée des charges

    Args:
        user_id: ID utilisateur
        months: Nombre de mois à analyser

    Returns:
        Répartition charges (fixes, semi-fixes, variables)
    """
    from budget_profiling_service.services.transaction_service import TransactionService
    from budget_profiling_service.services.charge_categorizer import ChargeCategorizer

    tx_service = TransactionService(db)
    detector = FixedChargeDetector(tx_service)
    categorizer = ChargeCategorizer(detector, tx_service)

    breakdown = await categorizer.categorize_user_charges(user_id, analysis_months=months)

    # Formater pour réponse
    # (Conversion DetectedCharge → dict déjà fait par categorizer)

    return breakdown
```

2. **Tests API endpoints**

```python
# tests/api/test_budget_profile_endpoints.py

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

@pytest.fixture
def test_client(app):
    """Client de test FastAPI"""
    return TestClient(app)

async def test_get_budget_profile_not_found(test_client):
    """Test profil non existant"""
    response = test_client.get("/api/v1/budget/profile/999999")

    # Devrait calculer automatiquement
    assert response.status_code in [200, 404]

async def test_get_budget_profile_existing(test_client, complete_user_data):
    """Test récupération profil existant"""
    user_id = complete_user_data

    # Calculer d'abord
    response = test_client.get(f"/api/v1/budget/profile/{user_id}?recalculate=true")
    assert response.status_code == 200

    data = response.json()
    assert data['user_id'] == user_id
    assert 'avg_monthly_income' in data
    assert 'savings_rate' in data
    assert 'user_segment' in data

async def test_detect_fixed_charges_endpoint(test_client, perfect_fixed_charge_data):
    """Test endpoint détection charges fixes"""
    response = test_client.post(
        "/api/v1/budget/fixed-charges/200/detect",
        params={"months": 6}
    )

    assert response.status_code == 200
    charges = response.json()

    assert len(charges) == 2  # EDF + Loyer
    assert all('merchant_name' in c for c in charges)
    assert all('recurrence_confidence' in c for c in charges)

async def test_validate_fixed_charge(test_client, db_session):
    """Test validation charge fixe"""
    from budget_profiling_service.models.fixed_charge import FixedCharge

    # Créer charge test
    charge = FixedCharge(
        user_id=500,
        merchant_name="Test Merchant",
        category="test",
        avg_amount=100.0,
        recurrence_confidence=0.9
    )
    db_session.add(charge)
    await db_session.commit()
    await db_session.refresh(charge)

    # Valider via API
    response = test_client.put(
        f"/api/v1/budget/fixed-charges/{charge.id}/validate",
        params={"validated": True}
    )

    assert response.status_code == 200
    assert response.json()['validated'] == True

    # Vérifier DB
    await db_session.refresh(charge)
    assert charge.validated_by_user == True

async def test_get_charge_breakdown(test_client, mixed_charges_data):
    """Test endpoint répartition charges"""
    response = test_client.get(
        "/api/v1/budget/breakdown/300",
        params={"months": 6}
    )

    assert response.status_code == 200
    breakdown = response.json()

    assert 'fixed' in breakdown
    assert 'semi_fixed' in breakdown
    assert 'variable' in breakdown
    assert 'summary' in breakdown

    # Vérifier structure
    assert 'total_monthly_avg' in breakdown['fixed']
    assert 'optimization_potential' in breakdown['variable']
```

#### Livrables Sprint 1.5

- [ ] 5+ endpoints API créés
- [ ] Documentation OpenAPI (Swagger)
- [ ] Tests API 80%+ couverture
- [ ] Validation erreurs et edge cases

#### Tâches Sprint 1.6 (Semaine 8)

1. **Interface frontend validation charges fixes**

```typescript
// frontend/src/components/BudgetProfiling/FixedChargesValidation.tsx

import React, { useState, useEffect } from 'react';
import { Card, Button, Badge, Progress } from '@/components/ui';
import { Check, X, AlertCircle } from 'lucide-react';

interface FixedCharge {
  id: number;
  merchant_name: string;
  category: string;
  avg_amount: number;
  amount_variance: number;
  recurrence_day: number;
  recurrence_confidence: number;
  transaction_count: number;
  validated_by_user: boolean;
}

export const FixedChargesValidation: React.FC<{ userId: number }> = ({ userId }) => {
  const [charges, setCharges] = useState<FixedCharge[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFixedCharges();
  }, [userId]);

  const fetchFixedCharges = async () => {
    const response = await fetch(`/api/v1/budget/fixed-charges/${userId}/detect?months=6`);
    const data = await response.json();
    setCharges(data);
    setLoading(false);
  };

  const handleValidate = async (chargeId: number, validated: boolean) => {
    await fetch(`/api/v1/budget/fixed-charges/${chargeId}/validate`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ validated })
    });

    // Mettre à jour l'état local
    setCharges(charges.map(c =>
      c.id === chargeId ? { ...c, validated_by_user: validated } : c
    ));
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.9) return <Badge variant="success">Haute confiance</Badge>;
    if (confidence >= 0.7) return <Badge variant="warning">Confiance moyenne</Badge>;
    return <Badge variant="danger">Faible confiance</Badge>;
  };

  if (loading) return <div>Chargement...</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Validation de vos charges fixes</h2>
      <p className="text-gray-600">
        Nous avons détecté {charges.length} charges fixes récurrentes.
        Validez-les pour affiner votre profil budgétaire.
      </p>

      {charges.map(charge => (
        <Card key={charge.id} className="p-4">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-lg font-semibold">{charge.merchant_name}</h3>
                {getConfidenceBadge(charge.recurrence_confidence)}
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Montant moyen:</span>
                  <span className="font-semibold ml-2">{charge.avg_amount.toFixed(2)}€</span>
                </div>
                <div>
                  <span className="text-gray-600">Jour de prélèvement:</span>
                  <span className="font-semibold ml-2">Le {charge.recurrence_day}</span>
                </div>
                <div>
                  <span className="text-gray-600">Variance:</span>
                  <span className="font-semibold ml-2">±{charge.amount_variance.toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-600">Occurrences:</span>
                  <span className="font-semibold ml-2">{charge.transaction_count} mois</span>
                </div>
              </div>

              <div className="mt-3">
                <div className="text-xs text-gray-600 mb-1">
                  Score de confiance: {(charge.recurrence_confidence * 100).toFixed(0)}%
                </div>
                <Progress value={charge.recurrence_confidence * 100} className="h-2" />
              </div>
            </div>

            <div className="flex gap-2 ml-4">
              {!charge.validated_by_user ? (
                <>
                  <Button
                    size="sm"
                    variant="success"
                    onClick={() => handleValidate(charge.id, true)}
                  >
                    <Check className="w-4 h-4 mr-1" />
                    Valider
                  </Button>
                  <Button
                    size="sm"
                    variant="danger"
                    onClick={() => handleValidate(charge.id, false)}
                  >
                    <X className="w-4 h-4 mr-1" />
                    Rejeter
                  </Button>
                </>
              ) : (
                <Badge variant="success" className="px-3 py-2">
                  <Check className="w-4 h-4 mr-1 inline" />
                  Validé
                </Badge>
              )}
            </div>
          </div>
        </Card>
      ))}

      {charges.length === 0 && (
        <Card className="p-8 text-center">
          <AlertCircle className="w-12 h-12 mx-auto text-gray-400 mb-3" />
          <p className="text-gray-600">
            Aucune charge fixe détectée. Ajoutez plus de transactions pour une meilleure analyse.
          </p>
        </Card>
      )}
    </div>
  );
};
```

2. **Dashboard profil budgétaire**

```typescript
// frontend/src/components/BudgetProfiling/ProfileDashboard.tsx

import React, { useState, useEffect } from 'react';
import { Card, Progress } from '@/components/ui';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Target } from 'lucide-react';

interface BudgetProfile {
  user_segment: string;
  avg_monthly_income: number;
  avg_monthly_expenses: number;
  avg_monthly_savings: number;
  savings_rate: number;
  fixed_charges_total: number;
  semi_fixed_charges_total: number;
  variable_charges_total: number;
  remaining_to_live: number;
  profile_completeness: number;
}

export const ProfileDashboard: React.FC<{ userId: number }> = ({ userId }) => {
  const [profile, setProfile] = useState<BudgetProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile();
  }, [userId]);

  const fetchProfile = async () => {
    const response = await fetch(`/api/v1/budget/profile/${userId}`);
    const data = await response.json();
    setProfile(data);
    setLoading(false);
  };

  if (loading) return <div>Chargement...</div>;
  if (!profile) return <div>Profil non disponible</div>;

  const chargeData = [
    { name: 'Charges fixes', value: profile.fixed_charges_total, color: '#ef4444' },
    { name: 'Charges semi-fixes', value: profile.semi_fixed_charges_total, color: '#f59e0b' },
    { name: 'Charges variables', value: profile.variable_charges_total, color: '#10b981' }
  ];

  const getSegmentColor = (segment: string) => {
    switch (segment) {
      case 'confortable': return 'bg-green-100 text-green-800';
      case 'équilibré': return 'bg-blue-100 text-blue-800';
      case 'budget_serré': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Votre profil budgétaire</h1>
        <Badge className={getSegmentColor(profile.user_segment)}>
          {profile.user_segment.toUpperCase()}
        </Badge>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Revenus moyens</p>
              <p className="text-2xl font-bold">{profile.avg_monthly_income.toFixed(0)}€</p>
            </div>
            <DollarSign className="w-8 h-8 text-green-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Dépenses moyennes</p>
              <p className="text-2xl font-bold">{profile.avg_monthly_expenses.toFixed(0)}€</p>
            </div>
            <TrendingDown className="w-8 h-8 text-red-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Épargne</p>
              <p className="text-2xl font-bold">{profile.avg_monthly_savings.toFixed(0)}€</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Taux d'épargne</p>
              <p className="text-2xl font-bold">{profile.savings_rate.toFixed(1)}%</p>
            </div>
            <Target className="w-8 h-8 text-purple-500" />
          </div>
        </Card>
      </div>

      {/* Répartition charges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Répartition de vos charges</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chargeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {chargeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Détail mensuel</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Charges fixes</span>
                <span className="text-sm font-semibold">{profile.fixed_charges_total.toFixed(0)}€</span>
              </div>
              <Progress value={(profile.fixed_charges_total / profile.avg_monthly_expenses) * 100} className="h-2 bg-red-200" />
              <p className="text-xs text-gray-500 mt-1">Potentiel optimisation: 0-5%</p>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Charges semi-fixes</span>
                <span className="text-sm font-semibold">{profile.semi_fixed_charges_total.toFixed(0)}€</span>
              </div>
              <Progress value={(profile.semi_fixed_charges_total / profile.avg_monthly_expenses) * 100} className="h-2 bg-orange-200" />
              <p className="text-xs text-gray-500 mt-1">Potentiel optimisation: 10-30%</p>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Charges variables</span>
                <span className="text-sm font-semibold">{profile.variable_charges_total.toFixed(0)}€</span>
              </div>
              <Progress value={(profile.variable_charges_total / profile.avg_monthly_expenses) * 100} className="h-2 bg-green-200" />
              <p className="text-xs text-gray-500 mt-1">Potentiel optimisation: 20-50%</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Reste à vivre */}
      <Card className="p-6 bg-gradient-to-r from-blue-50 to-blue-100">
        <h3 className="text-lg font-semibold mb-2">Reste à vivre</h3>
        <p className="text-3xl font-bold text-blue-900">{profile.remaining_to_live.toFixed(0)}€</p>
        <p className="text-sm text-gray-600 mt-2">
          Après vos charges fixes, il vous reste {profile.remaining_to_live.toFixed(0)}€ par mois pour
          vos dépenses courantes et votre épargne.
        </p>
      </Card>

      {/* Complétude profil */}
      {profile.profile_completeness < 1.0 && (
        <Card className="p-4 bg-yellow-50 border-yellow-200">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-6 h-6 text-yellow-600" />
            <div className="flex-1">
              <p className="font-semibold">Profil incomplet ({(profile.profile_completeness * 100).toFixed(0)}%)</p>
              <p className="text-sm text-gray-600">
                Ajoutez plus de transactions pour améliorer la précision de votre profil.
              </p>
            </div>
            <Progress value={profile.profile_completeness * 100} className="w-32" />
          </div>
        </Card>
      )}
    </div>
  );
};
```

#### Livrables Sprint 1.6

- [ ] Interface validation charges fixes
- [ ] Dashboard profil budgétaire
- [ ] Tests frontend (Cypress/Playwright)
- [ ] Documentation utilisateur

#### Tests de Validation Phase 1

```bash
# Tests end-to-end
npm run test:e2e -- tests/budget-profiling/
# Expected: All scenarios pass

# Test parcours utilisateur complet
# 1. Connexion user
# 2. Navigation dashboard profil
# 3. Validation 3 charges fixes
# 4. Vérification mise à jour profil
# Expected: Parcours fluide sans erreur

# Performance API
ab -n 100 -c 10 http://localhost:8006/api/v1/budget/profile/100
# Expected:
# - Mean response time < 500ms
# - No failed requests

# Validation données
python scripts/validate_phase1.py
# Expected:
# ✓ 5 tables créées
# ✓ API endpoints fonctionnels (5/5)
# ✓ Tests unitaires >85% couverture
# ✓ Interface utilisateur opérationnelle
```

---

**CRITÈRE GO/NO-GO PHASE 1** ✅

Pour passer à la Phase 2, TOUS les critères suivants doivent être validés:

- [x] ✅ Base de données: 5 tables créées et migrées
- [x] ✅ Détection charges fixes: Taux succès >85% sur jeu test
- [x] ✅ Catégorisation: Logique validée sur 30+ catégories
- [x] ✅ Profil budgétaire: Calculs corrects (vérifiés manuellement)
- [x] ✅ API endpoints: 5 endpoints fonctionnels
- [x] ✅ Interface: Dashboard + validation charges
- [x] ✅ Tests: Couverture code >85%
- [x] ✅ Performance: API <500ms P95
- [x] ✅ Documentation: README à jour

**Si critères non atteints** : Prolonger Phase 1 de 1-2 semaines.

---

## Phase 2 : Recommandations (6 semaines)

*[À développer dans même format détaillé]*

**Objectif** : Moteur de recommandations actionnables

**Sprints** :
- Sprint 2.1 (Semaine 9): Moteur de règles business
- Sprint 2.2 (Semaine 10): Règles d'optimisation (20 règles)
- Sprint 2.3 (Semaine 11): Calcul scénarios économies
- Sprint 2.4 (Semaine 12): Extensions règles (60+ total)
- Sprint 2.5 (Semaine 13): API + Interface recommandations
- Sprint 2.6 (Semaine 14): Tracking efficacité + feedback

---

## Phase 3 : Objectifs & Saisonnalité (6 semaines)

*[À développer]*

---

## Phase 4 : Optimisations & ML (4 semaines)

*[À développer]*

---

## Tests de Validation Globaux

### Tests de Bout en Bout (End-to-End)

```python
# tests/e2e/test_complete_flow.py

async def test_complete_user_journey():
    """
    Test parcours utilisateur complet:
    1. Nouveau user avec 6 mois de transactions
    2. Calcul profil budgétaire
    3. Détection charges fixes
    4. Validation charges par user
    5. Génération recommandations
    6. Création objectif épargne
    7. Suivi progression
    """
    # À implémenter
```

### Tests de Charge (Load Testing)

```bash
# scripts/load_test.sh

# Test 100 utilisateurs simultanés
locust -f tests/load/budget_profiling_load.py --users=100 --spawn-rate=10
# Expected:
# - 95% requests < 1s
# - 0% failed requests
# - Throughput > 50 req/s
```

---

## Critères de Go/No-Go par Phase

### Phase 1 → Phase 2

- [ ] Toutes fonctionnalités Phase 1 validées
- [ ] Tests end-to-end passent (5+ scénarios)
- [ ] Beta test 10 users: Feedback >4/5
- [ ] Performance API <500ms P95
- [ ] Pas de bugs critiques

### Phase 2 → Phase 3

- [ ] 60+ règles de recommandations actives
- [ ] Taux d'acceptation recommandations >15%
- [ ] Tests end-to-end enrichis (10+ scénarios)
- [ ] Beta test 50 users: Feedback >4/5

### Phase 3 → Phase 4

- [ ] Objectifs d'épargne fonctionnels
- [ ] Détection saisonnalité validée (75% précision)
- [ ] Beta test 100 users: NPS >+10

### Phase 4 → Production

- [ ] Tests de charge passés (100+ users simultanés)
- [ ] Sécurité RGPD validée (audit externe)
- [ ] Documentation complète (user + dev)
- [ ] Monitoring en place (alertes configurées)
- [ ] Plan de rollback testé

---

**FIN DE LA ROADMAP** - Document vivant, mise à jour continue pendant développement
