# 🧪 Scénarios de Test MVP Harena
## Guide de Validation pour Présentation Investisseurs

**Version:** 4.1.5
**Date:** 21 Octobre 2025
**Architecture:** conversation_service_v3 avec Intent Routing Intelligent

---

## 📊 Vue d'ensemble des tests

| Catégorie | Nombre de scénarios | Objectif |
|-----------|---------------------|----------|
| **Conversationnel** | 15 | Validation persona et UX |
| **Recherche Simple** | 20 | Requêtes basiques transactions |
| **Agrégations** | 25 | Calculs et statistiques |
| **Analyses Temporelles** | 20 | Tendances et périodes |
| **Multi-critères** | 15 | Filtres combinés |
| **Edge Cases** | 10 | Robustesse système |
| **Performance** | 5 | Benchmarks vitesse |
| **TOTAL** | **110** | **Couverture complète** |

---

## 🎭 Catégorie 1: Tests Conversationnels (15)
**Objectif:** Valider le routage intelligent et l'absence de recherches inutiles

### ✅ Attendu: Réponse en < 3s, requires_search=false, aucune requête ES

1. **Salutations basiques**
   - "Bonjour"
   - "Salut"
   - "Hello"
   - "Bonsoir"
   - "Coucou"

2. **Questions sur les capacités**
   - "Que peux-tu faire ?"
   - "Quelles sont tes fonctionnalités ?"
   - "Comment tu peux m'aider ?"
   - "C'est quoi tes capacités ?"
   - "Tu sais faire quoi ?"

3. **Demandes d'aide**
   - "Aide"
   - "Comment ça marche ?"
   - "J'ai besoin d'aide"
   - "Explique-moi comment utiliser Harena"

4. **Politesses**
   - "Merci"
   - "Merci beaucoup"
   - "Au revoir"

---

## 🔍 Catégorie 2: Recherches Simples (20)
**Objectif:** Valider la recherche basique de transactions

### ✅ Attendu: Résultats pertinents, temps < 5s, requires_search=true

5. **Recherche par marchand**
   - "Montre-moi mes transactions Carrefour"
   - "Mes achats chez Amazon"
   - "Transactions McDonald's"
   - "Mes paiements Netflix"
   - "Achats Fnac"

6. **Recherche par catégorie**
   - "Mes dépenses alimentaires"
   - "Transactions dans les restaurants"
   - "Mes achats de loisirs"
   - "Dépenses en transport"
   - "Mes frais de santé"

7. **Recherche par type d'opération**
   - "Mes virements"
   - "Tous mes prélèvements"
   - "Mes retraits d'argent"
   - "Paiements par carte"
   - "Mes chèques"

8. **Recherche par montant**
   - "Transactions de plus de 100€"
   - "Mes petites dépenses de moins de 10€"
   - "Achats entre 50€ et 150€"
   - "Ma plus grosse dépense"
   - "Mes transactions supérieures à 500€"

---

## 💰 Catégorie 3: Agrégations et Calculs (25)
**Objectif:** Valider les calculs statistiques et agrégations

### ✅ Attendu: Résultats numériques corrects, agrégations visibles

9. **Sommes par catégorie**
   - "Combien j'ai dépensé en courses ?"
   - "Total de mes dépenses restaurants"
   - "Somme de mes achats loisirs"
   - "Mes dépenses totales en transport"
   - "Combien j'ai dépensé en santé ?"

10. **Sommes par marchand**
    - "Combien j'ai dépensé chez Carrefour ?"
    - "Total de mes achats Amazon"
    - "Mes dépenses McDonald's"
    - "Combien j'ai payé à EDF ?"
    - "Total de mes courses Leclerc"

11. **Sommes par période**
    - "Combien j'ai dépensé ce mois-ci ?"
    - "Mes dépenses cette semaine"
    - "Total de mes achats aujourd'hui"
    - "Dépenses du mois dernier"
    - "Mes achats de l'année"

12. **Moyennes**
    - "Moyenne de mes dépenses alimentaires"
    - "Montant moyen de mes transactions restaurant"
    - "Dépense moyenne par jour"
    - "Ticket moyen Carrefour"
    - "Montant moyen de mes achats"

13. **Comptages**
    - "Combien de transactions chez Carrefour ?"
    - "Nombre de restaurants ce mois-ci"
    - "Combien de fois j'ai acheté sur Amazon ?"
    - "Nombre de transactions aujourd'hui"
    - "Combien de retraits cette semaine ?"

---

## 📈 Catégorie 4: Analyses Temporelles (20)
**Objectif:** Valider la gestion des périodes et tendances

### ✅ Attendu: Filtrage temporel correct, comparaisons pertinentes

14. **Périodes relatives simples**
    - "Mes dépenses aujourd'hui"
    - "Transactions de cette semaine"
    - "Achats de ce mois-ci"
    - "Dépenses de l'année"
    - "Mes transactions d'hier"

15. **Périodes relatives complexes**
    - "Mes dépenses des 7 derniers jours"
    - "Transactions des 30 derniers jours"
    - "Achats des 3 derniers mois"
    - "Dépenses du trimestre"
    - "Mes transactions de la semaine dernière"

16. **Comparaisons temporelles**
    - "Compare mes dépenses ce mois-ci et le mois dernier"
    - "Différence entre cette semaine et la semaine dernière"
    - "Évolution de mes dépenses alimentaires"
    - "Ai-je dépensé plus cette semaine ?"
    - "Tendance de mes achats restaurants"

17. **Dates spécifiques**
    - "Mes transactions le 5 septembre"
    - "Dépenses entre le 1er et le 15 octobre"
    - "Achats du mois d'août"
    - "Transactions en septembre 2025"
    - "Mes dépenses avant le 10 octobre"

---

## 🎯 Catégorie 5: Requêtes Multi-critères (15)
**Objectif:** Valider les filtres combinés complexes

### ✅ Attendu: Filtrage précis sur plusieurs dimensions

18. **Catégorie + Période**
    - "Mes dépenses alimentaires ce mois-ci"
    - "Restaurants cette semaine"
    - "Achats loisirs en septembre"
    - "Transport cette année"
    - "Santé le mois dernier"

19. **Marchand + Période**
    - "Mes achats Carrefour ce mois-ci"
    - "Amazon cette semaine"
    - "McDonald's aujourd'hui"
    - "Netflix cette année"
    - "EDF le mois dernier"

20. **Catégorie + Montant**
    - "Mes grosses dépenses alimentaires de plus de 50€"
    - "Petits achats loisirs de moins de 20€"
    - "Restaurants de plus de 30€"
    - "Achats transport entre 10€ et 50€"
    - "Santé supérieure à 100€"

21. **Marchand + Montant + Période**
    - "Mes achats Carrefour de plus de 100€ ce mois-ci"
    - "Amazon entre 20€ et 50€ cette semaine"
    - "McDonald's de moins de 15€ aujourd'hui"
    - "Courses Leclerc supérieures à 80€ en septembre"
    - "Achats Fnac de plus de 30€ cette année"

22. **Analyse Top/Bottom**
    - "Mes 10 plus grosses dépenses"
    - "Top 5 des marchands ce mois-ci"
    - "Mes 3 catégories où je dépense le plus"
    - "Les 5 dernières transactions"
    - "Top 10 des restaurants"

---

## ⚠️ Catégorie 6: Edge Cases et Robustesse (10)
**Objectif:** Valider la gestion des cas limites

### ✅ Attendu: Réponses élégantes, pas de crash

23. **Requêtes ambiguës**
    - "Mes trucs"
    - "Les machins de la semaine"
    - "Combien ?"
    - "Montre tout"
    - "C'est combien ?"

24. **Requêtes sans résultats**
    - "Mes achats chez Tesla"
    - "Dépenses spatiales"
    - "Transactions de demain"
    - "Achats de 10000€"
    - "Mes dépenses en 1990"

25. **Requêtes contradictoires**
    - "Mes dépenses aujourd'hui le mois dernier"
    - "Achats de plus de 100€ et moins de 50€"
    - "Transactions Carrefour chez Amazon"
    - "Dépenses alimentaires en loisirs"

26. **Fautes de frappe**
    - "Carfour" (au lieu de Carrefour)
    - "restoran" (au lieu de restaurant)
    - "aliemntation" (au lieu de alimentation)
    - "Macdo" (au lieu de McDonald's)
    - "trasport" (au lieu de transport)

27. **Questions complexes**
    - "Quelle est l'évolution de mes dépenses alimentaires par rapport à mes revenus en tenant compte de l'inflation ?"
    - "Analyse prédictive de mes dépenses futures"
    - "Optimise mon budget pour économiser 500€"
    - "Quelle est la corrélation entre mes achats loisirs et restaurants ?"

---

## ⚡ Catégorie 7: Tests de Performance (5)
**Objectif:** Benchmarks et validation SLA

### ✅ Attendu: Temps de réponse respectés

28. **Réponses conversationnelles**
   - Temps cible: < 3s
   - Test: "Bonjour" × 10 fois
   - Métrique: avg_response_time

29. **Recherches simples**
   - Temps cible: < 5s
   - Test: "Mes transactions Carrefour" × 10 fois
   - Métrique: avg_search_time

30. **Agrégations complexes**
   - Temps cible: < 8s
   - Test: "Somme de mes dépenses par catégorie ce mois-ci" × 5 fois
   - Métrique: avg_aggregation_time

31. **Streaming**
   - Temps au premier token: < 2s
   - Test: Requête financière en mode stream
   - Métrique: time_to_first_chunk

32. **Concurrence**
   - Requêtes simultanées: 10 users
   - Test: Questions mixtes en parallèle
   - Métrique: throughput, error_rate

---

## 🎯 Scénarios de Démonstration Investisseurs

### 📺 Démo 1: L'Expérience Utilisateur (2 min)
**Objectif:** Montrer la fluidité et l'intelligence du système

```
User: "Bonjour"
→ Réponse instantanée avec persona

User: "Que peux-tu faire ?"
→ Présentation des capacités

User: "Combien j'ai dépensé en courses ce mois-ci ?"
→ Réponse avec agrégation et insights

User: "Montre-moi mes 5 plus grosses dépenses"
→ Top 5 avec détails
```

### 📺 Démo 2: L'Intelligence Analytique (3 min)
**Objectif:** Démontrer la puissance d'analyse

```
User: "Mes dépenses restaurants cette semaine"
→ Liste + statistiques

User: "Compare avec la semaine dernière"
→ Analyse comparative avec tendance

User: "Quelle est ma catégorie la plus dépensière ?"
→ Top catégories avec insights

User: "Donne-moi le détail de mes achats loisirs"
→ Transactions détaillées + patterns
```

### 📺 Démo 3: La Robustesse (2 min)
**Objectif:** Prouver la fiabilité du système

```
User: "Mes achats chez Carfour" (faute de frappe)
→ Correction automatique vers Carrefour

User: "Mes dépenses spatiales"
→ Gestion élégante "aucun résultat"

User: "Montre tout"
→ Question ambiguë gérée intelligemment

User: "Mes 10 plus grosses dépenses entre 50€ et 200€ ce mois-ci"
→ Requête complexe multi-critères résolue
```

---

## 📊 Métriques de Succès MVP

### KPIs Techniques
- ✅ **Intent Routing Accuracy**: > 95%
- ✅ **Search Avoidance Rate**: > 70%
- ✅ **Response Time (Conversational)**: < 3s
- ✅ **Response Time (Search)**: < 5s
- ✅ **Response Time (Aggregation)**: < 8s
- ✅ **Query Success Rate**: > 90%
- ✅ **Error Rate**: < 5%

### KPIs Business
- ✅ **User Satisfaction**: "Réponse utile" > 85%
- ✅ **Query Diversity**: > 50 types de questions
- ✅ **Conversation Length**: > 3 tours en moyenne
- ✅ **Feature Coverage**: 100% use cases MVP

---

## 🧪 Protocole de Test

### Phase 1: Tests Automatisés (30 min)
1. Lancer les 110 scénarios via script
2. Collecter métriques de performance
3. Identifier les cas d'échec
4. Générer rapport automatique

### Phase 2: Tests Manuels (1h)
1. Parcourir 20 scénarios aléatoires
2. Valider qualité des réponses
3. Tester variations de formulation
4. Noter les edge cases

### Phase 3: Démo Investisseurs (10 min)
1. Exécuter Démo 1 (UX)
2. Exécuter Démo 2 (Analytics)
3. Exécuter Démo 3 (Robustesse)
4. Présenter dashboard de métriques

---

## 🎓 Questions Bonus: Validation Avancée (10)

33. **Compréhension contextuelle**
    - "Mes dépenses Carrefour" → puis "Et chez Leclerc ?"
    - "Mes restaurants" → puis "Lesquels ?"

34. **Négations**
    - "Mes transactions sauf les restaurants"
    - "Toutes mes dépenses hors alimentation"

35. **Approximations**
    - "Environ combien j'ai dépensé ?"
    - "À peu près mes achats loisirs"

36. **Langage naturel varié**
    - "J'ai claqué combien en bouffe ?"
    - "C'est quoi mes thunes en courses ?"

37. **Multi-langues (si applicable)**
    - "How much did I spend on groceries?"
    - "Quanto ho speso in ristoranti?"

38. **Expressions régionales**
    - "Mes emplettes"
    - "Mes courses alimentaires"

39. **Abréviations**
    - "Mes TX Carrefour"
    - "Dépenses resto"

40. **Émojis**
    - "Mes dépenses 🍔 ce mois"
    - "Achats 🎮 cette semaine"

41. **Demandes de clarification**
    - "Précise ta question" après requête floue
    - Gestion dialogue multi-tours

42. **Méta-questions**
    - "Combien de fois tu m'as aidé aujourd'hui ?"
    - "Quelles sont mes questions les plus fréquentes ?"

---

## ✅ Checklist de Validation MVP

### Avant Démo Investisseurs

- [ ] Les 110 scénarios passent à > 90%
- [ ] Temps de réponse respectés (3s/5s/8s)
- [ ] Intent routing fonctionne (> 95% accuracy)
- [ ] Aucun crash sur edge cases
- [ ] Dashboard métriques opérationnel
- [ ] Logs propres (pas d'erreurs critiques)
- [ ] Frontend responsive et fluide
- [ ] Streaming fonctionne parfaitement
- [ ] Persistence des conversations OK
- [ ] Statistiques temps réel disponibles

### Pendant Démo

- [ ] Backup plan si connexion ES échoue
- [ ] Exemples pré-chargés dans la base
- [ ] Compte démo avec données réalistes
- [ ] Screenshots des meilleurs résultats
- [ ] Vidéo de backup si problème live

### Après Démo

- [ ] Collecter feedback investisseurs
- [ ] Noter questions non couvertes
- [ ] Identifier améliorations prioritaires
- [ ] Préparer roadmap v5.0

---

## 🚀 Conclusion

Ce document fournit **110+ scénarios de test** couvrant:
- ✅ Conversationnel et UX
- ✅ Recherche et filtrage
- ✅ Agrégations et calculs
- ✅ Analyses temporelles
- ✅ Requêtes complexes
- ✅ Robustesse et edge cases
- ✅ Performance et scalabilité

**Utilisation:**
1. Automatiser les tests avec un script Python
2. Valider manuellement les cas critiques
3. Exécuter les 3 démos investisseurs
4. Présenter le dashboard de métriques

**Objectif:** Présenter le MVP en **toute confiance** avec une couverture de test > 90%

---

**Date de dernière mise à jour:** 21 Octobre 2025
**Version du système:** v4.1.5
**Auteur:** Équipe Harena
**Status:** ✅ Ready for Investor Demo
