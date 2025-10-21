"""
AggregationEnricher - Enrichit les queries avec des templates d'agrégations pré-définis

Résout le problème de génération LLM incorrecte pour agrégations complexes.
Au lieu de laisser le LLM générer la syntaxe Elasticsearch complexe,
on applique des templates validés qui garantissent une syntaxe correcte.

Author: Claude Code
Date: 2025-10-21
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AggregationEnricher:
    """
    Enrichit les queries avec des templates d'agrégations pré-définis

    Résout le problème de génération LLM incorrecte pour agrégations complexes
    en appliquant des templates Elasticsearch validés.

    Architecture:
        QueryAnalyzer → ElasticsearchBuilder → AggregationEnricher → Execute

    Templates disponibles:
        - by_category: Répartition par catégorie
        - by_merchant: Top marchands
        - monthly_trend: Évolution mensuelle
        - weekly_trend: Évolution hebdomadaire
        - by_weekday: Pattern par jour de la semaine
        - spending_statistics: Statistiques globales complètes
    """

    # Templates pour agrégations simples (métriques racine)
    SIMPLE_METRIC_TEMPLATES = {
        "total_amount": {
            "sum": {"field": "amount_abs"}
        },
        "total_count": {
            "value_count": {"field": "transaction_id"}
        },
        "average_amount": {
            "avg": {"field": "amount_abs"}
        },
        "max_amount": {
            "max": {"field": "amount_abs"}
        },
        "min_amount": {
            "min": {"field": "amount_abs"}
        }
    }

    # Templates d'agrégations validés (syntaxe ES correcte garantie)
    TEMPLATES = {
        "by_category": {
            "terms": {
                "field": "category_name.keyword",
                "size": 20,
                "order": {"total_amount": "desc"}
            },
            "aggs": {
                "total_amount": {
                    "sum": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                },
                "avg_transaction": {
                    "avg": {"field": "amount_abs"}
                }
            }
        },

        "by_merchant": {
            "terms": {
                "field": "merchant_name.keyword",
                "size": 10,
                "order": {"total_spent": "desc"}
            },
            "aggs": {
                "total_spent": {
                    "sum": {"field": "amount_abs"}
                },
                "frequency": {
                    "value_count": {"field": "transaction_id"}
                },
                "avg_basket": {
                    "avg": {"field": "amount_abs"}
                }
            }
        },

        "monthly_trend": {
            "date_histogram": {
                "field": "date",
                "calendar_interval": "month",
                "format": "yyyy-MM"
            },
            "aggs": {
                "total_spent": {
                    "sum": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                },
                "avg_transaction": {
                    "avg": {"field": "amount_abs"}
                }
            }
        },

        "weekly_trend": {
            "date_histogram": {
                "field": "date",
                "calendar_interval": "week",
                "format": "yyyy-'W'ww"
            },
            "aggs": {
                "total_spent": {
                    "sum": {"field": "amount_abs"}
                },
                "transaction_count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        },

        "by_weekday": {
            "terms": {
                "field": "weekday",
                "size": 7,
                "order": {"_key": "asc"}
            },
            "aggs": {
                "total_spent": {
                    "sum": {"field": "amount_abs"}
                },
                "avg_spent": {
                    "avg": {"field": "amount_abs"}
                },
                "count": {
                    "value_count": {"field": "transaction_id"}
                }
            }
        }
    }

    # Templates composites (plusieurs agrégations)
    COMPOSITE_TEMPLATES = {
        "spending_statistics": {
            "global_stats": {
                "stats": {"field": "amount_abs"}
            },
            "total_transactions": {
                "value_count": {"field": "transaction_id"}
            },
            "debit_stats": {
                "filter": {"term": {"transaction_type": "debit"}},
                "aggs": {
                    "sum_debit": {
                        "sum": {"field": "amount_abs"}
                    },
                    "count_debit": {
                        "value_count": {"field": "transaction_id"}
                    }
                }
            },
            "credit_stats": {
                "filter": {"term": {"transaction_type": "credit"}},
                "aggs": {
                    "sum_credit": {
                        "sum": {"field": "amount"}
                    },
                    "count_credit": {
                        "value_count": {"field": "transaction_id"}
                    }
                }
            }
        }
    }

    # Mapping intent/keywords → templates
    INTENT_MAPPING = {
        # Intent exact matching (from QueryAnalyzer)
        "by_category": ["by_category"],
        "by_merchant": ["by_merchant"],
        "by_category_and_merchant": ["by_category", "by_merchant"],
        "monthly_trend": ["monthly_trend"],
        "weekly_trend": ["weekly_trend"],
        "by_weekday": ["by_weekday"],
        "spending_statistics": ["spending_statistics"],
        "overview": ["spending_statistics", "by_category"],

        # Alias courants (pour compatibilité avec QueryAnalyzer)
        "by_date": ["monthly_trend"],  # QueryAnalyzer retourne souvent "by_date"
        "statistics": ["spending_statistics"],  # Alias pour statistiques globales
        "by_time": ["monthly_trend"],  # Autre alias temporel
        "temporal": ["monthly_trend"],  # Autre alias temporel
    }

    # Patterns de détection pour fallback
    DETECTION_PATTERNS = {
        "by_category": [
            "par catégorie", "par categorie", "répartition", "catégories",
            "quelle catégorie", "dans chaque catégorie", "breakdown",
            "distribution par catégorie"
        ],
        "by_merchant": [
            "par marchand", "où je dépense", "quels magasins",
            "top marchands", "marchands fréquents", "où dépensé",
            "principaux marchands", "magasins favoris"
        ],
        "monthly_trend": [
            "évolution mensuelle", "par mois", "tendance mensuelle",
            "chaque mois", "mois par mois", "mensuel", "mensuelle",
            "derniers mois", "ces derniers mois"
        ],
        "weekly_trend": [
            "par semaine", "hebdomadaire", "chaque semaine",
            "semaine par semaine", "dernières semaines"
        ],
        "by_weekday": [
            "jour de la semaine", "quel jour", "weekend",
            "lundi", "mardi", "mercredi", "jeudi", "vendredi",
            "samedi", "dimanche", "jour par jour"
        ],
        "spending_statistics": [
            "résumé", "statistiques", "vue d'ensemble",
            "bilan", "overview", "global", "aperçu",
            "résumé financier", "état financier"
        ]
    }

    # Agrégations simples qui NE nécessitent PAS de templates
    # Ces agrégations peuvent être générées correctement par le LLM
    SIMPLE_AGGREGATIONS = [
        "total_amount",
        "total_count",
        "average_amount",
        "max_amount",
        "min_amount",
        "sum",
        "count",
        "avg",
        "max",
        "min",
        "stats"
    ]

    def enrich(
        self,
        query: Dict[str, Any],
        aggregations_requested: List[str]
    ) -> Dict[str, Any]:
        """
        Enrichit une query avec les templates d'agrégations appropriés

        Args:
            query: Query générée par ElasticsearchBuilder
            aggregations_requested: Liste des agrégations demandées (from QueryAnalysis)

        Returns:
            Query enrichie avec templates
        """

        # Si pas d'agrégations demandées, retourner tel quel
        if not aggregations_requested:
            logger.debug("No aggregations requested, skipping enrichment")
            return query

        # Séparer agrégations simples et complexes
        simple_aggs_requested = [
            agg for agg in aggregations_requested
            if agg in self.SIMPLE_AGGREGATIONS
        ]
        complex_aggs = [
            agg for agg in aggregations_requested
            if agg not in self.SIMPLE_AGGREGATIONS
        ]

        # Si uniquement des agrégations simples, utiliser les templates simples
        if not complex_aggs:
            logger.info(f"Only simple aggregations requested {aggregations_requested}, applying simple templates")

            # Initialiser aggregations si absent
            if "aggregations" not in query:
                query["aggregations"] = {}

            # Appliquer les templates simples
            templates_applied = []
            for agg_requested in simple_aggs_requested:
                if agg_requested in self.SIMPLE_METRIC_TEMPLATES:
                    query["aggregations"][agg_requested] = self.SIMPLE_METRIC_TEMPLATES[agg_requested]
                    templates_applied.append(agg_requested)
                    logger.info(f"✅ Applied simple template: {agg_requested}")
                else:
                    logger.warning(f"No simple template found for: {agg_requested}")

            if templates_applied:
                logger.info(f"🎯 Simple aggregations applied: {templates_applied}")
            else:
                logger.warning("No simple templates were applied")

            return query

        # Initialiser aggregations si absent
        if "aggregations" not in query:
            query["aggregations"] = {}

        # Stocker les agrégations simples générées par le LLM (à préserver)
        simple_aggs = query.get("aggregations", {}).copy()

        # Réinitialiser pour appliquer les templates
        query["aggregations"] = {}

        # Appliquer les templates correspondants (uniquement pour agrégations complexes)
        templates_applied = []

        for agg_requested in complex_aggs:
            template_names = self.INTENT_MAPPING.get(agg_requested, [])

            if not template_names:
                logger.warning(f"No template mapping found for: {agg_requested}")
                continue

            for template_name in template_names:
                # Vérifier si c'est un template simple ou composite
                if template_name in self.TEMPLATES:
                    template = self.TEMPLATES[template_name]
                    query["aggregations"][template_name] = template
                    templates_applied.append(template_name)
                    logger.info(f"✅ Applied aggregation template: {template_name}")

                elif template_name in self.COMPOSITE_TEMPLATES:
                    # Template composite: merger toutes les aggs
                    composite = self.COMPOSITE_TEMPLATES[template_name]
                    query["aggregations"].update(composite)
                    templates_applied.append(template_name)
                    logger.info(f"✅ Applied composite template: {template_name}")

                else:
                    logger.warning(f"Template not found: {template_name}")

        # Ré-ajouter les agrégations simples du LLM (sum, count basiques)
        # Ne pas écraser les templates appliqués
        # IMPORTANT: Ne réajouter QUE si AUCUN template n'a été appliqué ET que l'agg est valide
        if not templates_applied:
            # Si PAS de templates appliqués, utiliser les agrégations générées par le LLM
            for agg_name, agg_def in simple_aggs.items():
                if agg_name not in query["aggregations"]:
                    # Vérifier que l'agg est valide (pas d'imbrication incorrecte)
                    if self._is_valid_simple_aggregation(agg_def):
                        query["aggregations"][agg_name] = agg_def
                        logger.debug(f"Preserved simple aggregation from LLM: {agg_name}")
                    else:
                        logger.warning(f"Skipped invalid LLM aggregation: {agg_name}")

        if templates_applied:
            logger.info(f"🎯 Enrichment complete. Templates applied: {templates_applied}")
        else:
            logger.warning("No templates were applied during enrichment")

        return query

    def _is_valid_simple_aggregation(self, agg_def: Dict[str, Any]) -> bool:
        """
        Vérifie si une agrégation est valide (pas d'imbrication incorrecte)

        Détecte les erreurs comme:
        {
          "sum": {...},
          "transaction_count": {...}  ← ERREUR: deux types dans une agg
        }

        Args:
            agg_def: Définition de l'agrégation

        Returns:
            True si valide, False sinon
        """
        if not isinstance(agg_def, dict):
            return False

        # Liste des types d'agrégations Elasticsearch
        agg_types = ["sum", "avg", "min", "max", "count", "value_count", "stats",
                     "terms", "date_histogram", "histogram", "range", "filter"]

        # Compter combien de types d'agrégation sont présents
        found_types = [key for key in agg_def.keys() if key in agg_types]

        # Une agrégation valide ne doit avoir QU'UN SEUL type
        if len(found_types) > 1:
            logger.warning(f"Invalid aggregation: multiple types found: {found_types}")
            return False

        return True

    def detect_from_query_text(self, user_message: str) -> List[str]:
        """
        Détecte les templates nécessaires à partir du message utilisateur

        Fallback si QueryAnalyzer ne détecte pas correctement les agrégations.
        Utilise pattern matching sur des mots-clés.

        Args:
            user_message: Message original de l'utilisateur

        Returns:
            Liste des templates détectés
        """

        message_lower = user_message.lower()
        detected = []

        for template_name, keywords in self.DETECTION_PATTERNS.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(template_name)
                logger.info(f"🔍 Detected template from message: {template_name}")

        return detected

    def get_available_templates(self) -> List[str]:
        """Retourne la liste des templates disponibles"""
        return list(self.TEMPLATES.keys()) + list(self.COMPOSITE_TEMPLATES.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Retourne les informations sur un template

        Args:
            template_name: Nom du template

        Returns:
            Dict avec structure du template et description
        """

        info = {
            "name": template_name,
            "type": None,
            "template": None,
            "description": None
        }

        if template_name in self.TEMPLATES:
            info["type"] = "simple"
            info["template"] = self.TEMPLATES[template_name]
        elif template_name in self.COMPOSITE_TEMPLATES:
            info["type"] = "composite"
            info["template"] = self.COMPOSITE_TEMPLATES[template_name]

        # Ajouter descriptions
        descriptions = {
            "by_category": "Répartition des dépenses par catégorie avec total, count et moyenne",
            "by_merchant": "Top marchands avec total dépensé, fréquence et panier moyen",
            "monthly_trend": "Évolution mensuelle des dépenses",
            "weekly_trend": "Évolution hebdomadaire des dépenses",
            "by_weekday": "Pattern de dépenses par jour de la semaine",
            "spending_statistics": "Statistiques globales complètes (débit, crédit, stats)"
        }

        info["description"] = descriptions.get(template_name, "No description available")

        return info
