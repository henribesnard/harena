# Changelog - conversation_service_v3

## v4.1.3 (2025-10-21)

### Fixed
- **Simple aggregations templates**: Résolution du problème d'agrégations simples (total_amount, total_count, etc.)
  - Ajout de templates SIMPLE_METRIC_TEMPLATES pour garantir une syntaxe Elasticsearch correcte
  - Les agrégations simples ne dépendent plus de la génération LLM (qui produisait des erreurs)
  - Taux de succès: 100% (8/8 tests) vs 87.5% (7/8) avant

### Changed
- **AggregationEnricher**: Logique améliorée pour séparer agrégations simples et complexes
  - Templates simples appliqués automatiquement pour: total_amount, total_count, average_amount, max_amount, min_amount
  - Templates complexes conservés pour: by_category, by_merchant, monthly_trend, spending_statistics
  - Pas de régression sur les agrégations complexes

### Technical Details
- Fichier modifié: `app/core/aggregation_enricher.py`
  - Lignes 37-54: Ajout SIMPLE_METRIC_TEMPLATES
  - Lignes 274-307: Logique de détection simple vs complexe
- Tests validés: 8/8 questions représentatives avec succès
- Latence moyenne: ~20-30s (acceptable pour volume de données)

### Removed
- Nettoyage des fichiers temporaires et documents obsolètes
- Conservation uniquement des fichiers essentiels au fonctionnement

---

## v4.1.0 (2025-10-21)

### Added
- **AggregationEnricher**: Système de templates d'agrégations pré-validés
- **Templates complexes**: by_category, by_merchant, monthly_trend, weekly_trend, by_weekday, spending_statistics
- **Detection patterns**: Mapping intent → templates avec fallback sur keywords

### Features
- Architecture: QueryAnalyzer → ElasticsearchBuilder → AggregationEnricher → Execute
- Garantie syntaxe Elasticsearch correcte pour agrégations complexes
- Support API v1 avec compatibilité complète

### Documentation
- test_results_analysis.md: Analyse complète des résultats de tests
- README.md: Documentation d'utilisation

---

## v3.0.0 (2025-10-20)

### Initial Release
- LangChain agents architecture
- OpenAI function calling for query building
- Elasticsearch integration
- API v3 endpoints
