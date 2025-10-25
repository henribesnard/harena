# 📘 Explication des Endpoints - Budget Profiling Service

## Vue d'ensemble

Le Budget Profiling Service expose 5 endpoints principaux qui permettent d'analyser les habitudes financières d'un utilisateur et de calculer son profil budgétaire.

**Base URL:** `http://localhost:3006/api/v1/budget`

**Authentification:** Tous les endpoints nécessitent un token JWT valide (sauf `/health`)

---

## 1. POST /api/v1/budget/profile/analyze

### 🎯 Objectif
Lance l'analyse complète du profil budgétaire d'un utilisateur sur une période donnée.

### 📥 Requête

```bash
POST /api/v1/budget/profile/analyze
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "months_analysis": 3  // Nombre de mois à analyser (1-12)
}
```

### 🔄 Processus d'analyse

#### Étape 1 : Détection des charges fixes
```
1. Récupère toutes les transactions de l'utilisateur (6-12 derniers mois)
2. Groupe les transactions par marchand (ex: EDF, Orange, Netflix)
3. Pour chaque marchand, analyse la récurrence :
   - Montant stable ? (variance < 10%)
   - Date stable ? (±5 jours du même jour du mois)
   - Fréquence mensuelle ? (~30 jours entre transactions)
   - Minimum 3 occurrences ?
4. Calcule un score de confiance (0.0-1.0) pour chaque charge détectée
5. Catégorise automatiquement (loyer, électricité, téléphone, etc.)
6. Sauvegarde les charges fixes détectées en base
```

**Exemple de charge fixe détectée :**
```json
{
  "merchant_name": "EDF",
  "category": "eau_electricite",
  "avg_amount": 85.50,
  "amount_variance": 3.2,      // 3.2% de variance
  "recurrence_day": 15,         // Prélèvement le 15 du mois
  "recurrence_confidence": 0.92, // 92% de confiance
  "transaction_count": 6        // 6 occurrences trouvées
}
```

#### Étape 2 : Calcul des agrégats mensuels
```
1. Récupère les transactions des N derniers mois
2. Groupe par mois
3. Calcule pour chaque mois :
   - Total revenus (transactions positives)
   - Total dépenses (transactions négatives)
   - Cashflow net (revenus - dépenses)
   - Nombre de transactions
```

#### Étape 3 : Calcul des moyennes
```
1. Revenus mensuels moyens
2. Dépenses mensuelles moyennes
3. Épargne mensuelle moyenne (revenus - dépenses)
4. Taux d'épargne (épargne / revenus * 100)
```

#### Étape 4 : Répartition des charges
```
1. Charges fixes : Total des charges fixes détectées
2. Charges semi-fixes : Catégories comme alimentation, carburant, santé
3. Charges variables : Catégories comme loisirs, restaurants, shopping
4. Reste à vivre : Revenus - charges fixes
```

#### Étape 5 : Détermination du segment utilisateur
```
Calcul du ratio : dépenses / revenus

- Ratio > 90% → "budget_serré"
  → L'utilisateur dépense presque tous ses revenus

- Ratio 70-90% → "équilibré"
  → L'utilisateur a un équilibre sain

- Ratio < 70% → "confortable"
  → L'utilisateur a une bonne capacité d'épargne
```

#### Étape 6 : Analyse du pattern comportemental
```
Analyse des transactions du dernier mois :

1. Compte le nombre de transactions par semaine
2. Calcule le montant moyen par transaction

Classification :
- >10 tx/semaine ET montant moyen <20€ → "acheteur_impulsif"
  → Nombreuses petites dépenses fréquentes

- <5 tx/semaine ET montant moyen >50€ → "planificateur"
  → Peu de transactions, montants importants

- Entre les deux → "dépensier_hebdomadaire"
  → Pattern standard de dépenses régulières
```

#### Étape 7 : Score de complétude
```
Calcule la fiabilité du profil (0.0-1.0) :

1. Données temporelles (40%) : Plus de mois = plus fiable
2. Charges fixes détectées (30%) : Plus de charges = plus précis
3. Présence de revenus (30%) : Revenus réguliers = profil complet
```

### 📤 Réponse

```json
{
  "user_segment": "équilibré",
  "behavioral_pattern": "planificateur",
  "avg_monthly_income": 3000.00,
  "avg_monthly_expenses": 2400.00,
  "avg_monthly_savings": 600.00,
  "savings_rate": 20.00,           // 20% d'épargne
  "fixed_charges_total": 1200.00,  // Loyer, factures, etc.
  "semi_fixed_charges_total": 800.00, // Courses, essence, etc.
  "variable_charges_total": 400.00,   // Loisirs, sorties, etc.
  "remaining_to_live": 1800.00,    // Revenus - charges fixes
  "profile_completeness": 0.85,    // 85% de fiabilité
  "last_analyzed_at": "2025-10-18T19:00:00Z"
}
```

### 💡 Cas d'usage

**Premier utilisateur :**
```bash
# L'utilisateur vient de créer son compte et a synchronisé ses comptes bancaires
# On lance l'analyse sur 6 mois pour avoir le maximum de données
curl -X POST http://localhost:3006/api/v1/budget/profile/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"months_analysis": 6}'
```

**Mise à jour mensuelle :**
```bash
# Chaque mois, on peut relancer l'analyse pour mettre à jour le profil
curl -X POST http://localhost:3006/api/v1/budget/profile/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"months_analysis": 3}'
```

---

## 2. GET /api/v1/budget/profile

### 🎯 Objectif
Récupère le profil budgétaire déjà calculé de l'utilisateur (sans relancer l'analyse).

### 📥 Requête

```bash
GET /api/v1/budget/profile
Authorization: Bearer <jwt_token>
```

### 🔄 Processus

```
1. Extrait le user_id depuis le token JWT
2. Cherche le profil en base de données (table user_budget_profile)
3. Si trouvé : retourne le profil
4. Si non trouvé : erreur 404 "Profil non trouvé. Lancez une analyse d'abord."
```

### 📤 Réponse

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
  "last_analyzed_at": "2025-10-18T19:00:00Z"
}
```

### 💡 Cas d'usage

**Affichage du dashboard :**
```bash
# Le frontend charge le profil à chaque visite du dashboard
# Pas besoin de relancer l'analyse à chaque fois
curl http://localhost:3006/api/v1/budget/profile \
  -H "Authorization: Bearer $TOKEN"
```

**Vérification de l'état :**
```bash
# Vérifier si l'utilisateur a déjà un profil calculé
# Si 404, proposer de lancer l'analyse
curl http://localhost:3006/api/v1/budget/profile \
  -H "Authorization: Bearer $TOKEN"
```

---

## 3. GET /api/v1/budget/fixed-charges

### 🎯 Objectif
Récupère la liste des charges fixes détectées automatiquement pour l'utilisateur.

### 📥 Requête

```bash
GET /api/v1/budget/fixed-charges
Authorization: Bearer <jwt_token>
```

### 🔄 Processus

```
1. Extrait le user_id depuis le token JWT
2. Récupère toutes les charges fixes actives (table fixed_charges)
3. Filtre les charges où is_active = true
4. Trie par ordre de confiance décroissante
5. Retourne la liste
```

### 📤 Réponse

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
  },
  {
    "id": 3,
    "merchant_name": "Netflix",
    "category": "abonnements",
    "avg_amount": 13.49,
    "recurrence_day": 10,
    "recurrence_confidence": 0.95,
    "validated_by_user": true,
    "transaction_count": 6
  }
]
```

### 🎨 Interprétation des champs

**recurrence_confidence :**
- `0.9 - 1.0` : Très fiable → Charge fixe confirmée
- `0.7 - 0.9` : Fiable → Probablement une charge fixe
- `< 0.7` : Incertain → Besoin de plus de données

**recurrence_day :**
- Jour du mois où la charge est généralement prélevée
- Ex: 15 = prélèvement le 15 de chaque mois
- Utile pour prévoir les flux de trésorerie

**validated_by_user :**
- `true` : L'utilisateur a confirmé cette charge
- `false` : Détection automatique non validée
- (Fonctionnalité Phase 2 : permettre à l'utilisateur de valider/rejeter)

### 💡 Cas d'usage

**Page "Mes charges fixes" :**
```bash
# Afficher toutes les charges fixes détectées
# L'utilisateur peut voir ses abonnements et factures récurrentes
curl http://localhost:3006/api/v1/budget/fixed-charges \
  -H "Authorization: Bearer $TOKEN"
```

**Widget dashboard :**
```javascript
// Afficher les 3 plus grosses charges fixes
const charges = await getFixedCharges(token);
const topCharges = charges
  .sort((a, b) => b.avg_amount - a.avg_amount)
  .slice(0, 3);
```

**Alerte anomalie (Phase 2) :**
```bash
# Si une charge fixe n'apparaît pas ce mois-ci, alerter l'utilisateur
# Ex: "Votre facture EDF (85.50€) n'a pas été prélevée ce mois-ci"
```

---

## 4. GET /api/v1/budget/monthly-aggregates

### 🎯 Objectif
Récupère les agrégats mensuels (revenus, dépenses, cashflow) sur les N derniers mois.

### 📥 Requête

```bash
GET /api/v1/budget/monthly-aggregates?months=3
Authorization: Bearer <jwt_token>
```

**Paramètres :**
- `months` (optionnel, défaut: 3) : Nombre de mois à récupérer

### 🔄 Processus

```
1. Récupère toutes les transactions des N derniers mois
2. Groupe les transactions par mois (année-mois)
3. Pour chaque mois, calcule :
   - total_income : Somme des transactions positives (revenus)
   - total_expenses : Somme des transactions négatives (dépenses)
   - net_cashflow : total_income - total_expenses
   - transaction_count : Nombre de transactions
4. Trie par ordre chronologique (du plus ancien au plus récent)
```

### 📤 Réponse

```json
[
  {
    "month": "2025-08",
    "total_income": 2900.00,
    "total_expenses": 2350.00,
    "net_cashflow": 550.00,
    "transaction_count": 142
  },
  {
    "month": "2025-09",
    "total_income": 3000.00,
    "total_expenses": 2400.00,
    "net_cashflow": 600.00,
    "transaction_count": 138
  },
  {
    "month": "2025-10",
    "total_income": 3100.00,
    "total_expenses": 2450.00,
    "net_cashflow": 650.00,
    "transaction_count": 145
  }
]
```

### 📊 Analyses possibles

**1. Évolution des revenus :**
```javascript
// Vérifier si les revenus augmentent ou diminuent
const revenuesTrend = months.map(m => m.total_income);
// [2900, 3000, 3100] → Tendance à la hausse
```

**2. Évolution des dépenses :**
```javascript
// Vérifier si l'utilisateur dépense plus ou moins
const expensesTrend = months.map(m => m.total_expenses);
// [2350, 2400, 2450] → Légère hausse
```

**3. Capacité d'épargne :**
```javascript
// Voir combien l'utilisateur peut épargner chaque mois
const savingsTrend = months.map(m => m.net_cashflow);
// [550, 600, 650] → Amélioration de l'épargne
```

**4. Détection d'anomalies :**
```javascript
// Identifier les mois inhabituels
const avgExpenses = moyenne(months.map(m => m.total_expenses));
const anomalies = months.filter(m =>
  m.total_expenses > avgExpenses * 1.2  // +20% vs moyenne
);
```

### 💡 Cas d'usage

**Graphique d'évolution :**
```javascript
// Afficher un graphique revenus/dépenses/épargne sur 6 mois
const data = await getMonthlyAggregates(token, 6);

chart.data = {
  labels: data.map(m => m.month),
  datasets: [
    { label: 'Revenus', data: data.map(m => m.total_income) },
    { label: 'Dépenses', data: data.map(m => m.total_expenses) },
    { label: 'Épargne', data: data.map(m => m.net_cashflow) }
  ]
};
```

**Comparaison mois en cours vs moyenne :**
```bash
# Récupérer les 3 derniers mois
curl "http://localhost:3006/api/v1/budget/monthly-aggregates?months=3" \
  -H "Authorization: Bearer $TOKEN"

# Calculer la moyenne des 2 premiers mois
# Comparer avec le mois en cours
# "Ce mois-ci, vous dépensez 5% de moins que votre moyenne"
```

---

## 5. GET /api/v1/budget/category-breakdown

### 🎯 Objectif
Récupère la répartition détaillée des dépenses par catégorie sur les N derniers mois.

### 📥 Requête

```bash
GET /api/v1/budget/category-breakdown?months=3
Authorization: Bearer <jwt_token>
```

**Paramètres :**
- `months` (optionnel, défaut: 3) : Nombre de mois à analyser

### 🔄 Processus

```
1. Récupère toutes les transactions des N derniers mois
2. Filtre uniquement les débits (dépenses)
3. Groupe par catégorie
4. Calcule le total des dépenses pour chaque catégorie
5. Retourne un dictionnaire {catégorie: montant}
```

### 📤 Réponse

```json
{
  "alimentation": 450.00,
  "transport": 120.00,
  "loisirs": 230.00,
  "eau_electricite": 85.50,
  "telephone_internet": 39.99,
  "assurances": 150.00,
  "shopping": 180.00,
  "restaurants": 140.00,
  "sante": 75.00,
  "carburant": 95.00
}
```

### 📊 Analyses possibles

**1. Top 5 des catégories :**
```javascript
// Identifier où l'utilisateur dépense le plus
const sorted = Object.entries(breakdown)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 5);

// Résultat: [
//   ["alimentation", 450.00],
//   ["loisirs", 230.00],
//   ["shopping", 180.00],
//   ["assurances", 150.00],
//   ["restaurants", 140.00]
// ]
```

**2. Répartition en pourcentages :**
```javascript
const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
const percentages = {};

for (const [category, amount] of Object.entries(breakdown)) {
  percentages[category] = (amount / total * 100).toFixed(1);
}

// {
//   "alimentation": "27.4%",
//   "loisirs": "14.0%",
//   "shopping": "11.0%",
//   ...
// }
```

**3. Catégories optimisables :**
```javascript
// Identifier les catégories où on peut réduire
const variable = ["loisirs", "restaurants", "shopping"];
const optimizable = Object.entries(breakdown)
  .filter(([cat]) => variable.includes(cat))
  .reduce((acc, [cat, amount]) => {
    acc[cat] = amount;
    return acc;
  }, {});

// Potentiel d'économie sur loisirs + restaurants + shopping
// = 230 + 140 + 180 = 550€
```

**4. Comparaison avec budgets cibles :**
```javascript
const budgets = {
  "alimentation": 400,  // Budget cible
  "loisirs": 200,
  "restaurants": 100
};

for (const [category, spent] of Object.entries(breakdown)) {
  if (budgets[category] && spent > budgets[category]) {
    console.log(`⚠️ ${category}: ${spent}€ (budget: ${budgets[category]}€)`);
  }
}
```

### 💡 Cas d'usage

**Diagramme circulaire (pie chart) :**
```javascript
// Afficher la répartition des dépenses par catégorie
const breakdown = await getCategoryBreakdown(token, 3);

pieChart.data = {
  labels: Object.keys(breakdown),
  datasets: [{
    data: Object.values(breakdown),
    backgroundColor: colors
  }]
};
```

**Recommandations d'économies (Phase 2) :**
```javascript
// "Si vous réduisez vos dépenses en loisirs de 15% (34.50€)
//  et en restaurants de 20% (28€), vous économiserez 62.50€/mois"

const breakdown = await getCategoryBreakdown(token, 3);
const loisirs = breakdown.loisirs * 0.15;  // 15% de réduction
const restaurants = breakdown.restaurants * 0.20;  // 20% de réduction
const savings = loisirs + restaurants;  // 62.50€
```

**Alertes de dépassement :**
```bash
# Récupérer le breakdown du mois en cours
curl "http://localhost:3006/api/v1/budget/category-breakdown?months=1" \
  -H "Authorization: Bearer $TOKEN"

# Comparer avec la moyenne des 3 derniers mois
curl "http://localhost:3006/api/v1/budget/category-breakdown?months=3" \
  -H "Authorization: Bearer $TOKEN"

# "Attention : vos dépenses en shopping sont 40% plus élevées ce mois-ci"
```

---

## 🔄 Workflow typique

### Première utilisation

```bash
# 1. L'utilisateur crée son compte et connecte ses banques
# (via user_service et sync_service)

# 2. Lancer l'analyse du profil sur 6 mois
POST /api/v1/budget/profile/analyze
{"months_analysis": 6}

# 3. Afficher le profil dans le dashboard
GET /api/v1/budget/profile

# 4. Afficher les charges fixes détectées
GET /api/v1/budget/fixed-charges

# 5. Afficher l'évolution sur 6 mois
GET /api/v1/budget/monthly-aggregates?months=6

# 6. Afficher la répartition des dépenses
GET /api/v1/budget/category-breakdown?months=6
```

### Utilisation mensuelle

```bash
# 1. Chaque mois, relancer l'analyse pour mettre à jour
POST /api/v1/budget/profile/analyze
{"months_analysis": 3}

# 2. Récupérer le profil mis à jour
GET /api/v1/budget/profile

# 3. Comparer avec le mois précédent
GET /api/v1/budget/monthly-aggregates?months=2
```

### Dashboard interactif

```bash
# Au chargement de la page
GET /api/v1/budget/profile           # Profil global
GET /api/v1/budget/fixed-charges     # Widget charges fixes
GET /api/v1/budget/monthly-aggregates?months=3  # Graphique 3 mois

# Quand l'utilisateur clique sur "Voir détails"
GET /api/v1/budget/category-breakdown?months=3  # Diagramme catégories
```

---

## 🎨 Exemples d'intégration Frontend

### React/TypeScript

```typescript
// services/budgetService.ts
export const budgetService = {
  async analyzeProfile(months: number = 3) {
    const response = await api.post('/api/v1/budget/profile/analyze', {
      months_analysis: months
    });
    return response.data;
  },

  async getProfile() {
    const response = await api.get('/api/v1/budget/profile');
    return response.data;
  },

  async getFixedCharges() {
    const response = await api.get('/api/v1/budget/fixed-charges');
    return response.data;
  },

  async getMonthlyAggregates(months: number = 3) {
    const response = await api.get(
      `/api/v1/budget/monthly-aggregates?months=${months}`
    );
    return response.data;
  },

  async getCategoryBreakdown(months: number = 3) {
    const response = await api.get(
      `/api/v1/budget/category-breakdown?months=${months}`
    );
    return response.data;
  }
};
```

### Composant Dashboard

```typescript
const BudgetDashboard = () => {
  const [profile, setProfile] = useState(null);
  const [charges, setCharges] = useState([]);
  const [aggregates, setAggregates] = useState([]);

  useEffect(() => {
    const loadData = async () => {
      setProfile(await budgetService.getProfile());
      setCharges(await budgetService.getFixedCharges());
      setAggregates(await budgetService.getMonthlyAggregates(6));
    };
    loadData();
  }, []);

  return (
    <div>
      <ProfileCard profile={profile} />
      <FixedChargesWidget charges={charges} />
      <MonthlyChart data={aggregates} />
    </div>
  );
};
```

---

## 🚀 Prochaines fonctionnalités (Phase 2)

### Endpoints à venir

1. **POST /api/v1/budget/recommendations/generate**
   - Génère des recommandations personnalisées
   - Scénarios d'économies mensuelles
   - Objectifs d'épargne suggérés

2. **GET /api/v1/budget/recommendations**
   - Liste des recommandations actives
   - Tri par priorité (high, medium, low)

3. **POST /api/v1/budget/goals**
   - Créer un objectif d'épargne
   - Ex: "Économiser 2000€ pour les vacances en juillet"

4. **GET /api/v1/budget/goals**
   - Liste des objectifs d'épargne
   - Progression vers chaque objectif

5. **PATCH /api/v1/budget/fixed-charges/{id}**
   - Valider/rejeter une charge fixe détectée
   - Marquer comme validée par l'utilisateur

---

## ❓ FAQ

**Q: Pourquoi mon profil retourne 404 ?**
A: Vous devez d'abord lancer une analyse avec `POST /api/v1/budget/profile/analyze`.

**Q: Puis-je analyser plus de 12 mois ?**
A: Non, le maximum est 12 mois pour des raisons de performance. Pour plus, utiliser plusieurs requêtes.

**Q: Les charges fixes sont-elles mises à jour automatiquement ?**
A: Oui, à chaque appel de `POST /api/v1/budget/profile/analyze`.

**Q: Comment savoir si une charge fixe est fiable ?**
A: Regarder le `recurrence_confidence`. Score > 0.9 = très fiable.

**Q: Les données sont-elles en temps réel ?**
A: Non, elles reflètent le dernier appel à `analyze`. Relancer l'analyse pour mettre à jour.

---

**Documentation complète:** http://localhost:3006/docs
