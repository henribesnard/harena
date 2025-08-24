#!/usr/bin/env python3
"""
Test Complet Search Service - Challenge Système Actuel
=====================================================

Script de test exhaustif pour valider toutes les capacités de search_service
et identifier les fonctionnalités manquantes ou défaillantes.

Usage: python test_search_service.py
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Configuration
SEARCH_SERVICE_URL = "http://localhost:8000/api/v1/search"
TEST_USER_ID = 34  # Utilisateur avec données de test
TIMEOUT_SECONDS = 30
OUTPUT_FILE = "search_service_test_results.md"

@dataclass
class TestResult:
    """Résultat d'un test individuel."""
    name: str
    category: str
    intention: str
    payload: Dict[str, Any]
    success: bool
    response_time_ms: float
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    expected_fields: List[str] = None
    missing_fields: List[str] = None
    status_code: Optional[int] = None

class SearchServiceTester:
    """Testeur complet pour search_service."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_endpoint(self, test_name: str, category: str, intention: str, 
                          payload: Dict[str, Any], expected_fields: List[str]) -> TestResult:
        """Teste un endpoint spécifique et retourne le résultat."""
        
        print(f"🧪 Testing {test_name}...")
        start_time = time.perf_counter()
        
        try:
            async with self.session.post(f"{SEARCH_SERVICE_URL}/search", json=payload) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Vérifier les champs attendus
                    missing_fields = []
                    if expected_fields:
                        missing_fields = [field for field in expected_fields 
                                        if not self._field_exists(data, field)]
                    
                    success = len(missing_fields) == 0 and data.get('success', False)
                    
                    return TestResult(
                        name=test_name,
                        category=category,
                        intention=intention,
                        payload=payload,
                        success=success,
                        response_time_ms=elapsed_ms,
                        response_data=data,
                        expected_fields=expected_fields,
                        missing_fields=missing_fields,
                        status_code=response.status
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        name=test_name,
                        category=category,
                        intention=intention,
                        payload=payload,
                        success=False,
                        response_time_ms=(time.perf_counter() - start_time) * 1000,
                        error_message=f"HTTP {response.status}: {error_text}",
                        expected_fields=expected_fields,
                        status_code=response.status
                    )
                    
        except Exception as e:
            return TestResult(
                name=test_name,
                category=category,
                intention=intention,
                payload=payload,
                success=False,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                error_message=str(e),
                expected_fields=expected_fields
            )

    def _field_exists(self, data: Dict[str, Any], field_path: str) -> bool:
        """Vérifie si un champ existe dans la réponse (support notation pointée)."""
        keys = field_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    # Convertir les clés numériques potentielles pour les dictionnaires
                    key = int(key) if key.isdigit() else key
                    current = current[key]
                elif isinstance(current, list):
                    index = int(key)
                    current = current[index]
                    # Gérer les indices de liste avec conversion en entier et contrôle des erreurs
                    try:
                        current = current[int(key)]
                    except (ValueError, IndexError, TypeError):
                        return False

                else:
                    return False
            return current is not None
        except (KeyError, TypeError, IndexError, ValueError):
            return False

    async def run_all_tests(self):
        """Lance tous les tests de recherche."""
        print("🚀 Démarrage des tests complets search_service")
        print(f"🎯 URL: {SEARCH_SERVICE_URL}")
        print(f"👤 User ID: {TEST_USER_ID}")
        print("=" * 60)
        
        # 1. Tests Recherches Transactionnelles (11 types)
        await self._test_transactional_searches()
        
        # 2. Tests Analyses Financières (7 types)  
        await self._test_financial_analysis()
        
        # 3. Tests Soldes et Comptes (3 types)
        await self._test_balance_analysis()
        
        # 4. Tests Recherches Spécialisées (6 types)
        await self._test_specialized_searches()
        
        # 5. Tests de Stress et Edge Cases
        await self._test_edge_cases()

    async def _test_transactional_searches(self):
        """Tests des 11 types de recherches transactionnelles."""
        print("\n📄 === RECHERCHES TRANSACTIONNELLES ===")
        
        # 1.1 Recherche générale de transactions
        result = await self.test_endpoint(
            "Recherche Générale Transactions",
            "Transactionnelles", 
            "TRANSACTION_SEARCH",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "aggregation_only": False,
                "page_size": 50
            },
            ["results", "response_metadata.total_results", "response_metadata.processing_time_ms"]
        )
        self.results.append(result)
        
        # 1.2 Recherche par date
        result = await self.test_endpoint(
            "Recherche par Date",
            "Transactionnelles",
            "SEARCH_BY_DATE", 
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "date": {
                        "gte": "2025-05-01",
                        "lte": "2025-05-31"
                    }
                },
                "aggregations": {
                    "stats_periode": {"stats": {"field": "amount_abs"}},
                    "count_by_type": {
                        "terms": {"field": "transaction_type.keyword"},
                        "aggs": {"total": {"sum": {"field": "amount_abs"}}}
                    }
                },
                "aggregation_only": False,
                "page_size": 30
            },
            ["results", "aggregations.stats_periode", "aggregations.count_by_type"]
        )
        self.results.append(result)
        
        # 1.3 Recherche par montant
        result = await self.test_endpoint(
            "Recherche par Montant",
            "Transactionnelles",
            "SEARCH_BY_AMOUNT",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "amount_abs": {
                        "gte": 50,
                        "lte": 200
                    }
                },
                "aggregations": {
                    "montant_ranges": {
                        "range": {
                            "field": "amount_abs",
                            "ranges": [
                                {"from": 50, "to": 100, "key": "50-100€"},
                                {"from": 100, "to": 200, "key": "100-200€"}
                            ]
                        }
                    }
                },
                "page_size": 25
            },
            ["results", "aggregations.montant_ranges"]
        )
        self.results.append(result)
        
        # 1.4 Recherche par marchand
        result = await self.test_endpoint(
            "Recherche par Marchand",
            "Transactionnelles", 
            "SEARCH_BY_MERCHANT",
            {
                "user_id": TEST_USER_ID,
                "query": "Amazon",
                "filters": {},
                "aggregations": {
                    "merchant_stats": {
                        "terms": {"field": "merchant_name.keyword", "size": 10},
                        "aggs": {
                            "total_spent": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "transaction_id"}}
                        }
                    }
                },
                "page_size": 20
            },
            ["results", "aggregations.merchant_stats"]
        )
        self.results.append(result)
        
        # 1.5 Recherche par catégorie
        result = await self.test_endpoint(
            "Recherche par Catégorie",
            "Transactionnelles",
            "SEARCH_BY_CATEGORY",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "category_name": "restaurant"
                },
                "aggregations": {
                    "category_breakdown": {
                        "terms": {"field": "category_name.keyword", "size": 10}
                    }
                },
                "page_size": 30
            },
            ["results", "aggregations.category_breakdown"]
        )
        self.results.append(result)
        
        # 1.6 Recherche textuelle libre
        result = await self.test_endpoint(
            "Recherche Textuelle Libre",
            "Transactionnelles",
            "SEARCH_BY_TEXT",
            {
                "user_id": TEST_USER_ID,
                "query": "supermarché courses",
                "filters": {},
                "aggregations": {
                    "text_matches": {
                        "terms": {"field": "merchant_name.keyword", "size": 10}
                    }
                },
                "page_size": 25,
                "sort": [{"_score": {"order": "desc"}}]
            },
            ["results", "aggregations.text_matches", "results.0._score"]
        )
        self.results.append(result)
        
        # 1.7 Comptage de transactions
        result = await self.test_endpoint(
            "Comptage de Transactions",
            "Transactionnelles",
            "COUNT_TRANSACTIONS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "aggregations": {
                    "total_count": {"value_count": {"field": "transaction_id"}},
                    "count_by_type": {
                        "terms": {"field": "transaction_type.keyword"}
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.total_count", "aggregations.count_by_type"]
        )
        self.results.append(result)

    async def _test_financial_analysis(self):
        """Tests des 7 types d'analyses financières."""
        print("\n💰 === ANALYSES FINANCIÈRES ===")
        
        # 2.1 Analyse globale des dépenses
        result = await self.test_endpoint(
            "Analyse Globale Dépenses",
            "Analyses Financières",
            "SPENDING_ANALYSIS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "transaction_type": "debit"
                },
                "aggregations": {
                    "total_spending": {"sum": {"field": "amount_abs"}},
                    "spending_by_category": {
                        "terms": {"field": "category_name.keyword", "size": 15},
                        "aggs": {"category_total": {"sum": {"field": "amount_abs"}}}
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.total_spending", "aggregations.spending_by_category"]
        )
        self.results.append(result)
        
        # 2.2 Analyse temporelle avec pipeline
        result = await self.test_endpoint(
            "Analyse Temporelle Pipeline",
            "Analyses Financières",
            "TREND_ANALYSIS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "transaction_type": "debit",
                    "date": {"gte": "2025-01-01"}
                },
                "aggregations": {
                    "trend_analysis": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month"
                        },
                        "aggs": {
                            "monthly_total": {"sum": {"field": "amount_abs"}},
                            "growth_rate": {
                                "derivative": {"buckets_path": "monthly_total"}
                            }
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.trend_analysis"]
        )
        self.results.append(result)
        
        # 2.3 Comparaison de périodes
        result = await self.test_endpoint(
            "Comparaison Périodes",
            "Analyses Financières", 
            "SPENDING_COMPARISON",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "transaction_type": "debit"
                },
                "aggregations": {
                    "period_comparison": {
                        "date_range": {
                            "field": "date",
                            "ranges": [
                                {"key": "previous_month", "from": "2025-04-01", "to": "2025-05-01"},
                                {"key": "current_month", "from": "2025-05-01", "to": "2025-06-01"}
                            ]
                        },
                        "aggs": {
                            "period_total": {"sum": {"field": "amount_abs"}}
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.period_comparison"]
        )
        self.results.append(result)

    async def _test_balance_analysis(self):
        """Tests des 3 types d'analyses de soldes."""
        print("\n🏦 === ANALYSES DE SOLDES ===")
        
        # 3.1 Solde général actuel
        result = await self.test_endpoint(
            "Solde Général Actuel",
            "Analyses Soldes",
            "BALANCE_INQUIRY",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "aggregations": {
                    "current_balance": {"sum": {"field": "amount"}},
                    "by_account": {
                        "terms": {"field": "account_id"},
                        "aggs": {
                            "account_balance": {"sum": {"field": "amount"}},
                            "account_name": {
                                "terms": {"field": "account_name.keyword"}
                            }
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.current_balance", "aggregations.by_account"]
        )
        self.results.append(result)
        
        # 3.2 Évolution du solde
        result = await self.test_endpoint(
            "Évolution du Solde",
            "Analyses Soldes",
            "BALANCE_EVOLUTION",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {
                    "date": {"gte": "2025-01-01"}
                },
                "aggregations": {
                    "balance_evolution": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "day"
                        },
                        "aggs": {
                            "daily_balance": {"sum": {"field": "amount"}},
                            "cumulative_balance": {
                                "cumulative_sum": {"buckets_path": "daily_balance"}
                            }
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.balance_evolution"]
        )
        self.results.append(result)

    async def _test_specialized_searches(self):
        """Tests des 6 types de recherches spécialisées."""
        print("\n🔍 === RECHERCHES SPÉCIALISÉES ===")
        
        # 4.1 Recherche multi-marchands
        result = await self.test_endpoint(
            "Recherche Multi-Marchands",
            "Recherches Spécialisées",
            "MULTI_MERCHANT_SEARCH",
            {
                "user_id": TEST_USER_ID,
                "query": "Amazon Netflix Spotify",
                "filters": {},
                "aggregations": {
                    "merchant_matches": {
                        "terms": {"field": "merchant_name.keyword", "size": 15}
                    }
                },
                "page_size": 30
            },
            ["results", "aggregations.merchant_matches"]
        )
        self.results.append(result)
        
        # 4.2 Recherche avec exclusions (must_not)
        result = await self.test_endpoint(
            "Recherche avec Exclusions",
            "Recherches Spécialisées",
            "SEARCH_WITH_EXCLUSIONS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "advanced_query": {
                    "bool": {
                        "must": [{"term": {"transaction_type": "debit"}}],
                        "must_not": [
                            {"terms": {"category_name.keyword": ["restaurant", "bar"]}}
                        ]
                    }
                },
                "aggregations": {
                    "excluded_analysis": {
                        "terms": {"field": "category_name.keyword", "size": 15}
                    }
                },
                "page_size": 40
            },
            ["results", "aggregations.excluded_analysis"]
        )
        self.results.append(result)
        
        # 4.3 Analyse multi-comptes
        result = await self.test_endpoint(
            "Analyse Multi-Comptes",
            "Recherches Spécialisées",
            "MULTI_ACCOUNT_ANALYSIS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "aggregations": {
                    "account_comparison": {
                        "terms": {"field": "account_id", "size": 10},
                        "aggs": {
                            "account_balance": {"sum": {"field": "amount"}},
                            "account_name": {
                                "terms": {"field": "account_name.keyword"}
                            },
                            "transaction_count": {"value_count": {"field": "transaction_id"}}
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.account_comparison"]
        )
        self.results.append(result)

    async def _test_edge_cases(self):
        """Tests de cas limites et de stress."""
        print("\n⚡ === CAS LIMITES ET STRESS ===")
        
        # 5.1 Requête avec tri personnalisé
        result = await self.test_endpoint(
            "Tri Personnalisé",
            "Cas Limites",
            "CUSTOM_SORT",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {"transaction_type": "debit"},
                "sort": [
                    {"amount_abs": {"order": "desc"}},
                    {"date": {"order": "desc"}}
                ],
                "page_size": 10
            },
            ["results"]
        )
        self.results.append(result)
        
        # 5.2 Agrégations complexes imbriquées
        result = await self.test_endpoint(
            "Agrégations Complexes Imbriquées",
            "Cas Limites",
            "COMPLEX_AGGREGATIONS",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "aggregations": {
                    "complex_analysis": {
                        "terms": {"field": "category_name.keyword", "size": 10},
                        "aggs": {
                            "category_stats": {"stats": {"field": "amount_abs"}},
                            "monthly_breakdown": {
                                "date_histogram": {
                                    "field": "date",
                                    "calendar_interval": "month"
                                },
                                "aggs": {
                                    "monthly_total": {"sum": {"field": "amount_abs"}},
                                    "top_merchants": {
                                        "terms": {"field": "merchant_name.keyword", "size": 3}
                                    }
                                }
                            }
                        }
                    }
                },
                "aggregation_only": True,
                "page_size": 1
            },
            ["aggregations.complex_analysis"]
        )
        self.results.append(result)
        
        # 5.3 Highlighting
        result = await self.test_endpoint(
            "Highlighting Résultats",
            "Cas Limites",
            "SEARCH_HIGHLIGHTING",
            {
                "user_id": TEST_USER_ID,
                "query": "restaurant",
                "filters": {},
                "page_size": 10,
                "highlight": {
                    "fields": {
                        "primary_description": {},
                        "merchant_name": {}
                    }
                }
            },
            ["results", "results.0.highlights"]
        )
        self.results.append(result)
        
        # 5.4 Test avec utilisateur invalide
        result = await self.test_endpoint(
            "Utilisateur Invalide",
            "Cas Limites",
            "INVALID_USER",
            {
                "user_id": 99999,  # Utilisateur inexistant
                "query": "",
                "filters": {},
                "page_size": 10
            },
            ["results"]
        )
        self.results.append(result)
        
        # 5.5 Pagination extrême
        result = await self.test_endpoint(
            "Pagination Extrême",
            "Cas Limites",
            "EXTREME_PAGINATION",
            {
                "user_id": TEST_USER_ID,
                "query": "",
                "filters": {},
                "page_size": 1000  # Maximum autorisé
            },
            ["results", "response_metadata.returned_results"]
        )
        self.results.append(result)

    def generate_report(self):
        """Génère le rapport markdown avec les résultats."""
        print(f"\n📊 Génération du rapport: {OUTPUT_FILE}")
        
        # Statistiques globales
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Temps de réponse moyen
        avg_response_time = sum(r.response_time_ms for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Groupement par catégorie
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"total": 0, "success": 0, "results": []}
            categories[result.category]["total"] += 1
            if result.success:
                categories[result.category]["success"] += 1
            categories[result.category]["results"].append(result)
        
        # Génération du markdown
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# 📊 Rapport de Test Search Service\n\n")
            f.write(f"**Date du test** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**URL testée** : {SEARCH_SERVICE_URL}\n\n")
            f.write(f"**Utilisateur test** : {TEST_USER_ID}\n\n")
            
            # Résumé exécutif
            f.write("## 🎯 Résumé Exécutif\n\n")
            f.write(f"- **Total des tests** : {total_tests}\n")
            f.write(f"- **Tests réussis** : {successful_tests} ✅\n")
            f.write(f"- **Tests échoués** : {failed_tests} ❌\n")
            f.write(f"- **Taux de succès** : {success_rate:.1f}%\n")
            f.write(f"- **Temps de réponse moyen** : {avg_response_time:.1f}ms\n\n")
            
            # Status par catégorie
            f.write("## 📋 Statut par Catégorie\n\n")
            f.write("| Catégorie | Total | Réussis | Taux | Temps Moyen |\n")
            f.write("|-----------|-------|---------|------|-------------|\n")
            
            for category, stats in categories.items():
                cat_success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
                cat_avg_time = sum(r.response_time_ms for r in stats["results"]) / stats["total"] if stats["total"] > 0 else 0
                status_emoji = "✅" if cat_success_rate == 100 else "⚠️" if cat_success_rate > 50 else "❌"
                f.write(f"| {category} | {stats['total']} | {stats['success']} | {cat_success_rate:.1f}% {status_emoji} | {cat_avg_time:.1f}ms |\n")
            
            f.write("\n")
            
            # Détails par catégorie
            f.write("## 🔍 Détails par Catégorie\n\n")
            
            for category, stats in categories.items():
                f.write(f"### {category}\n\n")
                
                for result in stats["results"]:
                    status_emoji = "✅" if result.success else "❌"
                    f.write(f"#### {status_emoji} {result.name}\n")
                    f.write(f"- **Intention** : `{result.intention}`\n")
                    f.write(f"- **Temps de réponse** : {result.response_time_ms:.1f}ms\n")
                    f.write(f"- **Code de statut** : {result.status_code or 'N/A'}\n")
                    
                    if not result.success:
                        f.write(f"- **❌ Erreur** : {result.error_message or 'Erreur inconnue'}\n")
                        if result.missing_fields:
                            f.write(f"- **Champs manquants** : {', '.join(result.missing_fields)}\n")
                    
                    if result.success and result.response_data:
                        # Quelques statistiques sur la réponse
                        data = result.response_data
                        if 'results' in data and isinstance(data['results'], list):
                            f.write(f"- **Résultats retournés** : {len(data['results'])}\n")
                        if 'aggregations' in data and data['aggregations']:
                            f.write(f"- **Agrégations présentes** : Oui\n")
                        if 'response_metadata' in data:
                            metadata = data['response_metadata']
                            if 'total_results' in metadata:
                                f.write(f"- **Total disponible** : {metadata['total_results']}\n")
                            if 'elasticsearch_took' in metadata:
                                f.write(f"- **Temps Elasticsearch** : {metadata['elasticsearch_took']}ms\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Problèmes identifiés
            failed_results = [r for r in self.results if not r.success]
            if failed_results:
                f.write("## 🚨 Problèmes Identifiés\n\n")
                
                # Grouper par type d'erreur
                error_types = {}
                for result in failed_results:
                    error_key = "Champs manquants" if result.missing_fields else "Erreur API"
                    if error_key not in error_types:
                        error_types[error_key] = []
                    error_types[error_key].append(result)
                
                for error_type, results in error_types.items():
                    f.write(f"### {error_type}\n\n")
                    for result in results:
                        f.write(f"- **{result.name}** : {result.error_message or 'Champs: ' + ', '.join(result.missing_fields or [])}\n")
                    f.write("\n")
            
            # Recommandations
            f.write("## 💡 Recommandations\n\n")
            
            if failed_tests == 0:
                f.write("🎉 **Excellent !** Tous les tests passent. Le search_service est prêt pour la production.\n\n")
            else:
                f.write("### Corrections Prioritaires\n\n")
                
                # Analyser les types d'échecs
                missing_field_tests = [r for r in failed_results if r.missing_fields]
                api_error_tests = [r for r in failed_results if not r.missing_fields]
                
                if missing_field_tests:
                    f.write("1. **Champs manquants dans les réponses** :\n")
                    all_missing = set()
                    for result in missing_field_tests:
                        all_missing.update(result.missing_fields or [])
                    for field in sorted(all_missing):
                        f.write(f"   - `{field}`\n")
                    f.write("\n")
                
                if api_error_tests:
                    f.write("2. **Erreurs API à corriger** :\n")
                    error_codes = {}
                    for result in api_error_tests:
                        code = result.status_code or 0
                        if code not in error_codes:
                            error_codes[code] = []
                        error_codes[code].append(result.name)
                    
                    for code, names in error_codes.items():
                        f.write(f"   - **HTTP {code}** : {', '.join(names)}\n")
                    f.write("\n")
                
                f.write("### Actions Recommandées\n\n")
                
                if success_rate < 50:
                    f.write("🔴 **Critique** : Moins de 50% des tests passent. Révision architecturale nécessaire.\n\n")
                elif success_rate < 80:
                    f.write("🟠 **Important** : Plusieurs fonctionnalités manquantes. Implémentation prioritaire nécessaire.\n\n")
                else:
                    f.write("🟡 **Bon** : La majorité des fonctionnalités fonctionnent. Corrections mineures nécessaires.\n\n")
            
            # Payload des tests échoués pour debug
            if failed_results:
                f.write("## 🔧 Payloads pour Debug\n\n")
                f.write("Utilisez ces payloads pour reproduire les erreurs :\n\n")
                
                for result in failed_results[:5]:  # Limiter à 5 pour ne pas surcharger
                    f.write(f"### {result.name}\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(result.payload, indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")

        print(f"✅ Rapport généré : {OUTPUT_FILE}")

async def main():
    """Fonction principale du script."""
    print("🚀 Search Service Comprehensive Test Suite")
    print("=========================================")
    
    try:
        async with SearchServiceTester() as tester:
            await tester.run_all_tests()
            tester.generate_report()
            
        # Afficher résumé
        total = len(tester.results)
        success = len([r for r in tester.results if r.success])
        
        print(f"\n📊 RÉSUMÉ FINAL")
        print(f"Tests réussis : {success}/{total} ({success/total*100:.1f}%)")
        print(f"Rapport détaillé : {OUTPUT_FILE}")
        
        if success == total:
            print("🎉 Tous les tests passent ! Search service est prêt.")
            sys.exit(0)
        else:
            print("⚠️ Certains tests échouent. Consultez le rapport pour les détails.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())