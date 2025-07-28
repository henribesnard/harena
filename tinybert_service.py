#!/usr/bin/env python3
"""
🚀 Service Ultra-Optimisé : Règles Intelligentes + Fallback DeepSeek Optionnel
Architecture: 95% règles (10ms) + 5% DeepSeek si vraiment nécessaire

Objectif: < 50ms pour 95% des requêtes, > 85% précision
"""

import asyncio
import json
import time
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from contextlib import asynccontextmanager

# =====================================
# CONFIGURATION OPTIMISÉE
# =====================================

class Config:
    # DeepSeek (fallback optionnel)
    DEEPSEEK_API_KEY = "sk-6923dd2c9f674a10b78665f3e01f9193"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"
    
    # Seuils intelligents
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # Seuil pour confiance élevée
    DEEPSEEK_THRESHOLD = 0.3          # Seuil pour déclencher DeepSeek
    
    # Performance targets
    TARGET_LATENCY_MS = 50
    TARGET_ACCURACY = 0.85

# =====================================
# MODÈLES DE DONNÉES
# =====================================

class IntentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[int] = None
    use_deepseek_fallback: bool = Field(True, description="Autoriser fallback DeepSeek")

class IntentResponse(BaseModel):
    intent: str
    intent_code: str
    confidence: float
    processing_time_ms: float
    method_used: str
    query: str
    entities: Dict[str, Any] = {}
    suggestions: List[str] = []
    cost_estimate: float = 0.0

@dataclass
class Metrics:
    total_requests: int = 0
    rule_based_success: int = 0
    deepseek_fallback: int = 0
    avg_latency: float = 0.0
    total_cost: float = 0.0

# =====================================
# MOTEUR DE RÈGLES ULTRA-OPTIMISÉ
# =====================================

class IntelligentRuleEngine:
    """Moteur de règles intelligent avec extraction d'entités"""
    
    def __init__(self):
        # Patterns regex optimisés pour français
        self.intent_patterns = {
            "ACCOUNT_BALANCE": {
                "patterns": [
                    r"\b(solde|combien.*ai|argent.*compte|euros?\s+sur|balance|disponible|reste)\b",
                    r"\b(compte.*courant|livret|épargne)\b.*\b(solde|combien)\b",
                    r"\bmon\s+(solde|compte)\b"
                ],
                "entities": [
                    (r"\b(compte\s+courant|livret\s+a|épargne|livret)\b", "account_type"),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month")
                ],
                "confidence_boost": 0.9
            },
            
            "SEARCH_BY_CATEGORY": {
                "patterns": [
                    r"\b(restaurant|resto|repas|dîner|déjeuner)\b",
                    r"\b(courses|alimentation|supermarché|carrefour|leclerc)\b", 
                    r"\b(transport|essence|carburant|taxi|uber)\b",
                    r"\b(shopping|vêtements|achats|boutique)\b",
                    r"\b(loisirs|cinéma|sport|vacances)\b",
                    r"\b(santé|pharmacie|médecin|dentiste)\b",
                    r"\b(dépenses?.*\b(restaurant|courses|transport|shopping|loisirs|santé)\b)"
                ],
                "entities": [
                    (r"\b(restaurant|resto|repas|dîner|déjeuner)\b", "category", "restaurant"),
                    (r"\b(courses|alimentation|supermarché)\b", "category", "alimentation"),
                    (r"\b(transport|essence|taxi|uber)\b", "category", "transport"),
                    (r"\b(shopping|vêtements|achats)\b", "category", "shopping"),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month"),
                    (r"\b(ce\s+mois|mois\s+dernier|cette\s+semaine|semaine\s+dernière)\b", "period")
                ],
                "confidence_boost": 0.85
            },
            
            "BUDGET_ANALYSIS": {
                "patterns": [
                    r"\b(budget|dépensé|combien.*dépensé|coûté|montant)\b",
                    r"\b(analyse|bilan|résumé|total).*\b(dépenses?|budget)\b",
                    r"\b(j'ai\s+dépensé|ça\s+m'a\s+coûté)\b"
                ],
                "entities": [
                    (r"\b(\d+)\s*euros?\b", "amount"),
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month"),
                    (r"\b(ce\s+mois|mois\s+dernier|cette\s+année|année\s+dernière)\b", "period")
                ],
                "confidence_boost": 0.8
            },
            
            "TRANSFER": {
                "patterns": [
                    r"\b(virer|virement|transfert|transférer)\b",
                    r"\b(envoyer|verser|payer).*\b(argent|euros?)\b",
                    r"\b(donner|prêter).*\b(\d+.*euros?)\b"
                ],
                "entities": [
                    (r"\b(\d+(?:[,\.]\d+)?)\s*euros?\b", "amount"),
                    (r"\b(à|vers|pour)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", "recipient")
                ],
                "confidence_boost": 0.9
            },
            
            "SEARCH_BY_DATE": {
                "patterns": [
                    r"\b(historique|transactions|opérations|mouvements)\b",
                    r"\b(hier|avant-hier|semaine|mois).*\b(dernier|dernière|passé)\b",
                    r"\b(récent|dernier|précédent)\b"
                ],
                "entities": [
                    (r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b", "month"),
                    (r"\b(hier|avant-hier|semaine\s+dernière|mois\s+dernier)\b", "period"),
                    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "date")
                ],
                "confidence_boost": 0.75
            },
            
            "CARD_MANAGEMENT": {
                "patterns": [
                    r"\b(carte|cb|visa|mastercard)\b.*\b(bloquer|débloquer|opposition)\b",
                    r"\b(bloquer|annuler|gérer).*\bcarte\b",
                    r"\b(limite|plafond).*\bcarte\b"
                ],
                "entities": [
                    (r"\b(visa|mastercard|cb|carte\s+bleue)\b", "card_type"),
                    (r"\b(\d+)\s*euros?\b", "amount")
                ],
                "confidence_boost": 0.95
            },
            
            "GREETING": {
                "patterns": [
                    r"^\s*(bonjour|salut|hello|bonsoir|coucou|hey)\b",
                    r"\b(bonjour|salut).*\b(comment\s+(ça\s+va|allez-vous))\b"
                ],
                "entities": [],
                "confidence_boost": 0.95
            },
            
            "HELP": {
                "patterns": [
                    r"\b(aide|aidez|help|comment)\b",
                    r"\b(expliquer|ne\s+comprends?\s+pas|aide-moi)\b",
                    r"^\s*(que|qu'est-ce|comment).*\b(faire|fonctionne)\b"
                ],
                "entities": [],
                "confidence_boost": 0.8
            },
            
            "GOODBYE": {
                "patterns": [
                    r"\b(au\s+revoir|bye|ciao|à\s+bientôt|goodbye|salut)\b",
                    r"\b(merci.*bye|c'est\s+tout|terminé)\b"
                ],
                "entities": [],
                "confidence_boost": 0.95
            }
        }
        
        # Mapping vers codes search service
        self.intent_to_search_code = {
            "ACCOUNT_BALANCE": "ACCOUNT_BALANCE",
            "SEARCH_BY_CATEGORY": "SEARCH_BY_CATEGORY", 
            "BUDGET_ANALYSIS": "BUDGET_ANALYSIS",
            "TRANSFER": "TRANSFER",
            "SEARCH_BY_DATE": "SEARCH_BY_DATE",
            "CARD_MANAGEMENT": "CARD_MANAGEMENT",
            "GREETING": "GREETING",
            "HELP": "HELP",
            "GOODBYE": "GOODBYE",
            "UNKNOWN": "UNKNOWN"
        }
        
        # Cache patterns compilés
        self._compiled_patterns = {}
        self._compile_patterns()
        
        self.logger = logging.getLogger(__name__)
    
    def _compile_patterns(self):
        """Pre-compilation des patterns regex pour performance"""
        for intent, config in self.intent_patterns.items():
            self._compiled_patterns[intent] = {
                "patterns": [re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]],
                "entities": [(re.compile(pattern, re.IGNORECASE), name, value if len(entity) > 2 else None) 
                           for entity in config["entities"] 
                           for pattern, name, *value in [entity]],
                "confidence_boost": config["confidence_boost"]
            }
    
    def _extract_entities(self, query: str, intent: str) -> Dict[str, Any]:
        """Extraction intelligente d'entités"""
        entities = {}
        
        if intent in self._compiled_patterns:
            for pattern, name, default_value in self._compiled_patterns[intent]["entities"]:
                matches = pattern.findall(query)
                if matches:
                    if default_value:
                        entities[name] = default_value
                    else:
                        entities[name] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return entities
    
    def detect_intent(self, query: str) -> Tuple[str, float, Dict[str, Any]]:
        """Détection ultra-rapide par règles"""
        query_clean = query.strip().lower()
        
        # Score par intention
        intent_scores = {}
        
        for intent, config in self._compiled_patterns.items():
            score = 0.0
            matches = 0
            
            # Test de chaque pattern
            for pattern in config["patterns"]:
                if pattern.search(query_clean):
                    matches += 1
                    # Score basé sur spécificité du pattern
                    pattern_score = len(pattern.pattern) / len(query_clean)
                    score += pattern_score * config["confidence_boost"]
            
            # Bonus pour multiple matches
            if matches > 1:
                score *= 1.2
            
            intent_scores[intent] = min(score, 1.0)
        
        # Meilleure intention
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_score = intent_scores[best_intent]
            
            if best_score > 0.1:  # Seuil minimum
                entities = self._extract_entities(query, best_intent)
                return best_intent, best_score, entities
        
        return "UNKNOWN", 0.05, {}
    
    def get_suggestions(self, intent: str, entities: Dict[str, Any]) -> List[str]:
        """Suggestions contextuelles intelligentes"""
        suggestions_map = {
            "ACCOUNT_BALANCE": [
                "Voir l'historique des soldes",
                "Comparer avec le mois dernier", 
                "Afficher tous mes comptes"
            ],
            "SEARCH_BY_CATEGORY": [
                f"Détails des dépenses {entities.get('category', '')}".strip(),
                "Comparer avec la période précédente",
                "Voir le budget par catégorie"
            ],
            "BUDGET_ANALYSIS": [
                "Analyse détaillée par catégorie",
                "Évolution sur plusieurs mois",
                "Comparaison avec mes objectifs"
            ],
            "TRANSFER": [
                "Voir mes bénéficiaires récents",
                "Programmer un virement récurrent",
                "Vérifier mes limites de virement"
            ],
            "SEARCH_BY_DATE": [
                "Filtrer par montant",
                "Exporter les données",
                "Recherche par marchand"
            ]
        }
        
        base_suggestions = suggestions_map.get(intent, [
            "Que puis-je faire d'autre ?",
            "Voir l'aide complète",
            "Retour au menu principal"
        ])
        
        return base_suggestions[:3]

# =====================================
# SERVICE PRINCIPAL OPTIMISÉ
# =====================================

class OptimizedIntentService:
    """Service optimisé règles + fallback optionnel"""
    
    def __init__(self):
        self.rule_engine = IntelligentRuleEngine()
        self.deepseek_client = None
        self.metrics = Metrics()
        
        # Cache mémoire ultra-rapide
        self.cache = {}
        self.cache_max_size = 200
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialisation"""
        self.logger.info("🚀 Initialisation service optimisé")
        
        # Client DeepSeek optionnel
        try:
            self.deepseek_client = AsyncOpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL,
                timeout=6.0
            )
            self.logger.info("✅ DeepSeek fallback disponible")
        except:
            self.logger.warning("⚠️ DeepSeek fallback indisponible")
    
    async def _deepseek_fallback(self, query: str) -> Tuple[str, float, Dict[str, Any], float]:
        """Fallback DeepSeek optimisé"""
        if not self.deepseek_client:
            return "UNKNOWN", 0.0, {}, 0.0
        
        start_time = time.time()
        
        try:
            response = await self.deepseek_client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "Classificateur intentions financières. JSON uniquement: {\"intent\":\"ACCOUNT_BALANCE|SEARCH_BY_CATEGORY|BUDGET_ANALYSIS|TRANSFER|SEARCH_BY_DATE|CARD_MANAGEMENT|GREETING|HELP|GOODBYE|UNKNOWN\",\"confidence\":0.0-1.0,\"entities\":{}}"
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.05,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            if "```" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0] if "```json" in result_text else result_text
            
            data = json.loads(result_text)
            
            processing_time = (time.time() - start_time) * 1000
            tokens = response.usage.total_tokens if response.usage else 100
            cost = tokens * 0.00014 / 1000
            
            return (
                data.get("intent", "UNKNOWN"),
                data.get("confidence", 0.0), 
                data.get("entities", {}),
                cost
            )
            
        except Exception as e:
            self.logger.error(f"Erreur DeepSeek: {e}")
            return "UNKNOWN", 0.0, {}, 0.0
    
    async def detect_intent(self, request: IntentRequest) -> Dict[str, Any]:
        """Pipeline de détection optimisé"""
        start_time = time.time()
        
        # Cache ultra-rapide
        cache_key = request.query.lower().strip()
        if cache_key in self.cache:
            cached = self.cache[cache_key].copy()
            cached["processing_time_ms"] = (time.time() - start_time) * 1000
            cached["method_used"] = "cache"
            return cached
        
        # 1. Règles intelligentes (ultra-rapide)
        rule_intent, rule_confidence, rule_entities = self.rule_engine.detect_intent(request.query)
        
        # 2. Décision fallback
        final_intent = rule_intent
        final_confidence = rule_confidence  
        final_entities = rule_entities
        method_used = "rules"
        cost = 0.0
        
        # Fallback DeepSeek seulement si vraiment nécessaire
        should_use_deepseek = (
            request.use_deepseek_fallback and
            rule_confidence < Config.DEEPSEEK_THRESHOLD and
            rule_intent == "UNKNOWN"
        )
        
        if should_use_deepseek:
            self.logger.info(f"🔄 Fallback DeepSeek (confiance règles: {rule_confidence:.3f})")
            ds_intent, ds_confidence, ds_entities, ds_cost = await self._deepseek_fallback(request.query)
            
            if ds_confidence > rule_confidence:
                final_intent = ds_intent
                final_confidence = ds_confidence
                final_entities = {**rule_entities, **ds_entities}  # Merge entities
                method_used = "deepseek_fallback"
                cost = ds_cost
                self.metrics.deepseek_fallback += 1
            else:
                method_used = "rules_vs_deepseek"
                self.metrics.rule_based_success += 1
        else:
            self.metrics.rule_based_success += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        # Résultat final
        result = {
            "intent": final_intent,
            "intent_code": self.rule_engine.intent_to_search_code.get(final_intent, "UNKNOWN"),
            "confidence": final_confidence,
            "processing_time_ms": processing_time,
            "method_used": method_used,
            "query": request.query,
            "entities": final_entities,
            "suggestions": self.rule_engine.get_suggestions(final_intent, final_entities),
            "cost_estimate": cost
        }
        
        # Cache si confiance élevée
        if final_confidence > 0.6 and len(self.cache) < self.cache_max_size:
            cache_result = result.copy()
            cache_result.pop("processing_time_ms")
            self.cache[cache_key] = cache_result
        
        # Métriques
        self.metrics.total_requests += 1
        self.metrics.total_cost += cost
        self.metrics.avg_latency = (
            (self.metrics.avg_latency * (self.metrics.total_requests - 1) + processing_time) / 
            self.metrics.total_requests
        )
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Métriques complètes"""
        if self.metrics.total_requests == 0:
            return {"total_requests": 0, "service_ready": True}
        
        rule_success_rate = self.metrics.rule_based_success / self.metrics.total_requests
        deepseek_usage_rate = self.metrics.deepseek_fallback / self.metrics.total_requests
        
        return {
            "total_requests": self.metrics.total_requests,
            "avg_latency_ms": round(self.metrics.avg_latency, 2),
            "total_cost": round(self.metrics.total_cost, 4),
            "performance": {
                "rule_success_rate": round(rule_success_rate, 3),
                "deepseek_usage_rate": round(deepseek_usage_rate, 3),
                "meets_latency_target": self.metrics.avg_latency <= Config.TARGET_LATENCY_MS,
                "target_latency_ms": Config.TARGET_LATENCY_MS
            },
            "distribution": {
                "rules_success": self.metrics.rule_based_success,
                "deepseek_fallback": self.metrics.deepseek_fallback
            },
            "cache_size": len(self.cache),
            "efficiency": {
                "cost_per_request": round(self.metrics.total_cost / self.metrics.total_requests, 6) if self.metrics.total_requests > 0 else 0,
                "fast_responses_percent": round(rule_success_rate * 100, 1)
            }
        }

# =====================================
# APPLICATION FASTAPI
# =====================================

service = OptimizedIntentService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Démarrage Service Ultra-Optimisé")
    await service.initialize()
    print("✅ Service prêt - Règles intelligentes activées")
    yield

app = FastAPI(
    title="Ultra-Optimized Intent Detection",
    description="Règles intelligentes + Fallback DeepSeek optionnel",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.post("/detect-intent", response_model=IntentResponse)
async def detect_intent_endpoint(request: IntentRequest):
    result = await service.detect_intent(request)
    return IntentResponse(**result)

@app.get("/metrics")
async def get_metrics():
    return service.get_metrics()

@app.get("/health") 
async def health():
    metrics = service.get_metrics()
    return {
        "status": "healthy",
        "total_requests": metrics.get("total_requests", 0),
        "avg_latency_ms": metrics.get("avg_latency_ms", 0)
    }

@app.get("/test-comprehensive")
async def comprehensive_test():
    """Test complet avec métriques détaillées"""
    test_cases = [
        # Tests basiques
        ("bonjour comment ça va", "GREETING"),
        ("quel est mon solde compte courant", "ACCOUNT_BALANCE"), 
        ("mes dépenses restaurant ce mois", "SEARCH_BY_CATEGORY"),
        ("faire un virement de 100 euros", "TRANSFER"),
        ("au revoir merci", "GOODBYE"),
        
        # Tests avec entités
        ("mes courses chez carrefour en janvier", "SEARCH_BY_CATEGORY"),
        ("virer 250 euros à Marie", "TRANSFER"),
        ("combien j'ai dépensé en transport", "BUDGET_ANALYSIS"),
        ("bloquer ma carte visa", "CARD_MANAGEMENT"),
        ("historique des transactions de décembre", "SEARCH_BY_DATE"),
        
        # Tests ambigus
        ("aide moi", "HELP"),
        ("quelque chose de très complexe et ambigu", "UNKNOWN")
    ]
    
    results = []
    start_test = time.time()
    
    for query, expected in test_cases:
        request = IntentRequest(query=query, user_id=1)
        result = await service.detect_intent(request)
        
        is_correct = result["intent"] == expected or (expected == "UNKNOWN" and result["confidence"] < 0.5)
        
        results.append({
            "query": query,
            "expected": expected,
            "detected": result["intent"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "latency_ms": result["processing_time_ms"],
            "method": result["method_used"],
            "entities": result["entities"]
        })
    
    total_test_time = (time.time() - start_test) * 1000
    
    # Statistiques
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    fast_responses = sum(1 for r in results if r["latency_ms"] < Config.TARGET_LATENCY_MS)
    
    return {
        "test_results": results,
        "statistics": {
            "total_tests": len(results),
            "correct_predictions": correct_count,
            "accuracy_rate": round(accuracy, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "fast_responses": fast_responses,
            "fast_response_rate": round(fast_responses / len(results), 3),
            "total_test_time_ms": round(total_test_time, 2),
            "meets_targets": {
                "latency": avg_latency <= Config.TARGET_LATENCY_MS,
                "accuracy": accuracy >= Config.TARGET_ACCURACY
            }
        },
        "service_metrics": service.get_metrics()
    }

# Test direct
async def run_performance_test():
    await service.initialize()
    
    test_queries = [
        "bonjour comment allez vous",
        "quel est mon solde", 
        "mes dépenses restaurant ce mois",
        "virement 100 euros à Paul",
        "historique janvier",
        "bloquer ma carte",
        "au revoir",
        "aide moi s'il te plaît",
        "requête très ambiguë pour tester"
    ]
    
    print("🧪 Test Performance Service Ultra-Optimisé")
    print("=" * 80)
    
    for query in test_queries:
        request = IntentRequest(query=query, use_deepseek_fallback=True)
        result = await service.detect_intent(request)
        
        entities_str = ", ".join([f"{k}:{v}" for k, v in result["entities"].items()]) if result["entities"] else "none"
        
        print(f"{query[:35]:<35} | {result['intent']:<18} | {result['confidence']:.3f} | {result['processing_time_ms']:5.1f}ms | {result['method_used']:<15} | {entities_str}")
    
    print("\n📊 Métriques finales:")
    metrics = service.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(run_performance_test())