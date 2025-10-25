# Spécification Fonctionnelle : Module de Profilage Utilisateur et Gestion Budgétaire Intelligente

## 1. Vue d'ensemble

Ce document décrit la fonctionnalité de profilage utilisateur et de gestion budgétaire intelligente. L'objectif est d'analyser automatiquement les transactions de l'utilisateur pour établir son profil financier, identifier ses différents types de charges, et lui proposer des recommandations personnalisées pour optimiser ses dépenses et atteindre ses objectifs d'épargne.

## 2. Objectifs fonctionnels

- Analyser automatiquement l'historique des transactions de l'utilisateur
- Catégoriser les dépenses en trois types : fixes, semi-fixes et variables
- Établir un profil comportemental et financier de l'utilisateur
- Détecter les tendances saisonnières et mensuelles de dépenses
- Proposer des scénarios d'optimisation budgétaire personnalisés
- Aider l'utilisateur à atteindre des objectifs d'épargne spécifiques

## 3. Catégorisation des charges

### 3.1 Charges fixes
**Définition** : Dépenses récurrentes mensuelles incompressibles ou difficilement réductibles.

**Caractéristiques** :
- Récurrence mensuelle stable (±5 jours autour de la même date)
- Montant identique ou avec faible variance (±10%)
- Catégories concernées : Loyer, Eau/Électricité, Gaz, Téléphone/Internet, Assurances, Garde d'enfants, Pension alimentaire, Crédits/Prêts

**Détection automatique** :
- Analyser les 3-6 derniers mois minimum
- Identifier les transactions récurrentes par bénéficiaire et montant
- Score de confiance basé sur la régularité

**Interaction utilisateur** :
- Proposer une validation des charges fixes détectées
- Permettre l'ajout manuel de charges fixes non détectées
- Option de paramétrage dans les préférences utilisateur

### 3.2 Charges variables
**Définition** : Dépenses discrétionnaires sur lesquelles l'utilisateur a un contrôle direct.

**Caractéristiques** :
- Montants et fréquences irréguliers
- Catégories concernées : Loisirs, Sorties/Restaurants, Shopping/Vêtements, Abonnements optionnels (streaming, salle de sport), Cadeaux, Voyages

**Potentiel d'optimisation** : Élevé (20-50% de réduction possible)

### 3.3 Charges semi-fixes
**Définition** : Dépenses nécessaires mais modulables en fonction des choix de consommation.

**Caractéristiques** :
- Récurrence régulière mais montants variables
- Catégories concernées : Alimentation/Courses, Carburant/Transport, Santé non remboursée, Entretien maison/véhicule

**Potentiel d'optimisation** : Moyen (10-30% de réduction possible)

## 4. Profil utilisateur

### 4.1 Comportement et habitudes transactionnelles

**Indicateurs à calculer** :
- Fréquence moyenne des transactions (par jour, par semaine)
- Montant moyen par transaction
- Canaux de paiement préférés (CB, virement, prélèvement)
- Heures et jours de transaction privilégiés
- Bénéficiaires fréquents

**Patterns comportementaux** :
- "Dépensier hebdomadaire" : concentration des dépenses en début/fin de semaine
- "Acheteur impulsif" : nombreuses petites transactions
- "Planificateur" : grosses transactions espacées et régulières

### 4.2 Niveau de dépenses et revenus

**Calculs mensuels** :
- Revenus totaux (salaires, aides, autres)
- Dépenses totales par catégorie
- Ratio dépenses/revenus
- Taux d'épargne actuel
- Reste à vivre après charges fixes

**Segments utilisateur** :
- Budget serré (ratio > 90%)
- Budget équilibré (ratio 70-90%)
- Capacité d'épargne confortable (ratio < 70%)

### 4.3 Analyse des tendances temporelles

**Sur plusieurs années (minimum 12 mois)** :

**Analyse mensuelle** :
- Identification des mois à forte dépense (ex : janvier avec les soldes, septembre avec la rentrée)
- Identification des mois à échéances multiples (ex : juin avec taxes, assurances)
- Variation des revenus (primes, 13e mois)

**Exemples de détection** :
- "Janvier : +25% de dépenses vs moyenne annuelle (soldes, abonnements annuels)"
- "Juin : 3 échéances importantes détectées (assurance auto, taxe d'habitation, cotisation professionnelle)"
- "Août : -40% de dépenses alimentation (vacances hors domicile)"

**Saisonnalité** :
- Périodes de forte consommation (vacances, fêtes)
- Cycles d'achat récurrents
- Anticiper les mois difficiles

## 5. Recommandations et scénarios d'optimisation

### 5.1 Principe général

Le système ne propose **pas** de réduire les charges fixes (considérées comme incompressibles), mais se concentre sur les charges variables et semi-fixes.

### 5.2 Types de recommandations

#### A. Économies mensuelles simples

**Format** : "Si tu réduis de X% tes dépenses en [catégorie 1] et de Y% en [catégorie 2], tu pourrais économiser Z€ par mois."

**Exemple concret** :
- "Si tu réduis de 10% tes dépenses en alimentation (30€) et de 15% en loisirs (22,50€), tu pourrais économiser 52,50€ par mois."

**Critères de génération** :
- Pourcentages réalistes et progressifs (10%, 15%, 20%)
- Montants concrets calculés sur la moyenne des 3 derniers mois
- Priorisation des catégories à fort potentiel d'économie
- Adaptation au profil utilisateur (ne pas suggérer de réduire une catégorie déjà minimale)

#### B. Objectifs d'épargne ciblés

**Format** : "Pour avoir un budget de X€ pour [objectif] en [mois], il faut que tu mettes de côté Y€ par mois en réduisant de Z% tes [catégories]."

**Exemple concret** :
- "Pour avoir un budget de 1000€ pour les vacances en juillet, il faut que tu mettes de côté 150€ à partir du mois prochain en réduisant de 20% tes dépenses courantes (40€) et de 30% tes loisirs (110€)."

**Éléments du calcul** :
- Nombre de mois avant l'objectif
- Montant mensuel à épargner = Objectif / Nombre de mois
- Répartition suggérée de l'effort sur plusieurs catégories
- Vérification de la faisabilité (montant à épargner < reste à vivre)

**Questions interactives** :
- "Quelles sont les dépenses sur lesquelles tu peux agir ?"
- "Préfères-tu réduire plutôt les loisirs ou l'alimentation ?"
- "Cet objectif te semble-t-il réaliste ?"

#### C. Alertes et détections proactives

**Dépassements détectés** :
- "Ce mois-ci, tes dépenses en [catégorie] sont 35% plus élevées que ta moyenne."
- "Attention : tu as déjà dépensé 80% de ton budget loisirs et nous sommes le 15 du mois."

**Opportunités d'optimisation** :
- "Tu as 3 abonnements dans la catégorie streaming pour un total de 35€/mois. Veux-tu les analyser ?"
- "Tes courses sont 20% plus chères le week-end. Essayer de faire tes courses en semaine pourrait t'économiser 30€/mois."

#### D. Comparaison et benchmark

**Par rapport aux mois précédents** :
- "Ce mois-ci, tu as économisé 45€ par rapport au mois dernier !"
- "Ta moyenne de dépenses en loisirs a baissé de 18% sur les 3 derniers mois."

**Évolution vers l'objectif** :
- "À ce rythme, tu atteindras ton objectif de 1000€ dans 6 mois au lieu de 7 !"
- "Il te reste 350€ à économiser pour ton objectif vacances. Tu es à 65% !"

### 5.3 Personnalisation des recommandations

**Selon le profil comportemental** :
- Dépensier impulsif → Suggestions de plafonds hebdomadaires, alertes fréquentes
- Planificateur → Recommandations mensuelles, analyse des tendances
- Budget serré → Focus sur micro-économies, identification des fuites

**Selon le niveau de revenus** :
- Revenus élevés → Objectifs d'épargne ambitieux, optimisation fiscale
- Revenus modestes → Économies réalistes, prévention des découverts
- Revenus irréguliers → Lissage des dépenses, constitution d'un coussin de sécurité

**Selon l'historique** :
- Nouvel utilisateur (< 3 mois) → Profil en construction, recommandations génériques
- Utilisateur établi (3-12 mois) → Recommandations personnalisées basiques
- Utilisateur fidèle (> 12 mois) → Analyse avancée avec saisonnalité et prédictions

## 6. Parcours utilisateur

### 6.1 Première configuration (onboarding)

**Étape 1 : Connexion bancaire**
- L'utilisateur connecte son/ses compte(s) bancaire(s)
- Récupération automatique de l'historique (6-24 mois selon disponibilité)

**Étape 2 : Analyse initiale**
- Le système analyse les transactions (traitement de 2-3 minutes)
- Message : "Nous analysons vos transactions pour créer votre profil financier..."

**Étape 3 : Validation des charges fixes**
- Affichage des charges fixes détectées avec score de confiance
- "Nous avons détecté ces charges fixes récurrentes. Sont-elles correctes ?"
- Possibilité d'ajouter/modifier/supprimer

**Étape 4 : Premier bilan**
- Présentation du profil : répartition des charges, moyenne mensuelle, catégories principales
- "Voici votre portrait financier des 6 derniers mois"

### 6.2 Utilisation quotidienne

**Tableau de bord principal** :
- Vision synthétique du mois en cours
- Comparaison avec la moyenne
- Alertes et recommandations prioritaires
- Progression vers les objectifs

**Section Analyse** :
- Répartition détaillée des dépenses
- Évolution temporelle (graphiques)
- Tendances et patterns détectés

**Section Objectifs** :
- Liste des objectifs d'épargne
- Scénarios de réduction de dépenses
- Suivi de progression

**Section Profil** :
- Gestion des charges fixes
- Paramétrage des catégories
- Préférences de notification

### 6.3 Interaction avec les recommandations

**Affichage** :
- Carte cliquable pour chaque recommandation
- Badge de priorité (🔴 Urgent, 🟠 Important, 🟢 Conseil)
- Montant d'économie potentielle mis en avant

**Actions possibles** :
- "Créer un objectif basé sur cette recommandation"
- "Voir le détail des transactions concernées"
- "Ignorer cette recommandation"
- "Me rappeler dans 1 mois"

**Feedback utilisateur** :
- "Cette recommandation t'a-t-elle été utile ?"
- Ajustement des algorithmes selon les retours

## 7. Règles de gestion importantes

### 7.1 Protection et confidentialité

- Les données financières ne sortent jamais du système
- Chiffrement des données sensibles
- Anonymisation pour les analyses agrégées
- Respect RGPD : droit à l'effacement, portabilité, information

### 7.2 Fiabilité des calculs

- Exclusion des transactions exceptionnelles (remboursements, virements internes)
- Gestion des transactions en plusieurs fois
- Détection des doublons
- Gestion de la TVA pour les professionnels (si applicable)

### 7.3 Limites et disclaimers

- "Ces recommandations sont basées sur votre historique et ne constituent pas un conseil financier."
- "Les économies mentionnées sont des estimations basées sur vos habitudes actuelles."
- "Pour les décisions financières importantes, nous vous recommandons de consulter un conseiller."

### 7.4 Mise à jour continue

- Recalcul automatique du profil tous les mois
- Actualisation des charges fixes tous les 3 mois
- Révision des tendances annuelles chaque année
- Notification des changements significatifs de comportement

## 8. Indicateurs de succès

### 8.1 Métriques d'engagement

- Taux de validation des charges fixes détectées (objectif > 85%)
- Nombre de recommandations consultées par utilisateur/mois
- Taux de création d'objectifs d'épargne (objectif > 40%)
- Temps passé dans la section Analyse

### 8.2 Métriques d'impact

- Économies réalisées vs économies suggérées
- Nombre d'objectifs d'épargne atteints
- Amélioration du taux d'épargne moyen
- Réduction des découverts bancaires

### 8.3 Métriques de satisfaction

- Note de satisfaction des recommandations (feedback utilisateur)
- Taux de désactivation des notifications
- Commentaires et retours qualitatifs
- Net Promoter Score (NPS)

## 9. Évolutions futures possibles

### Phase 2 (court terme)
- Comparaison avec des profils similaires (benchmark anonymisé)
- Suggestions de changement de fournisseur (énergie, téléphonie)
- Défis d'économie gamifiés

### Phase 3 (moyen terme)
- Assistant conversationnel pour affiner les objectifs
- Prédiction des dépenses futures (IA)
- Alertes prédictives avant dépassement

### Phase 4 (long terme)
- Conseil patrimonial personnalisé
- Optimisation fiscale
- Simulation de projets de vie (achat immobilier, retraite)

---

## 10. Annexes : Exemples de formulations

### Recommandations d'économies mensuelles

✅ **Bon exemple** :
"Si tu réduis de 10% tes dépenses en alimentation (25€) et de 15% en loisirs (30€), tu pourrais économiser 55€ par mois. Cela représente 660€ sur l'année !"

❌ **Mauvais exemple** :
"Tu devrais dépenser moins" (trop vague, pas actionnable)

### Objectifs d'épargne

✅ **Bon exemple** :
"Pour avoir un budget de 1200€ pour les vacances en août, il faut mettre de côté 200€ par mois à partir de février (6 mois). Tu peux y arriver en réduisant de 25% tes sorties restaurants (60€), de 15% tes loisirs (50€) et de 12% tes courses (90€). Quelles dépenses te semblent les plus faciles à ajuster ?"

❌ **Mauvais exemple** :
"Il faut économiser pour les vacances" (pas de plan concret)

### Alertes tendances

✅ **Bon exemple** :
"📊 Tendance détectée : En janvier, tu dépenses en moyenne 18% de plus qu'en décembre (soldes, abonnements annuels). Cette année, anticipe 280€ de dépenses supplémentaires. Veux-tu créer une réserve ?"

❌ **Mauvais exemple** :
"Tes dépenses varient" (pas assez spécifique)

### Analyse saisonnière

✅ **Bon exemple** :
"🔍 Analyse annuelle : Tes mois les plus coûteux sont janvier (+23%), juin (+18%) et septembre (+15%). En juin, 3 échéances majeures arrivent en même temps : assurance auto (450€), taxe foncière (850€) et cotisation mutuelle (120€). Veux-tu lisser ces dépenses ?"

❌ **Mauvais exemple** :
"Certains mois coûtent plus cher" (pas exploitable)