"""
🧠 Niveau L1 - Lightweight Classifier TinyBERT

Classification TinyBERT optimisée performance 15-30ms pour requêtes
non matchées L0 avec embeddings 384-dim et similarité cosinus.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import json

from conversation_service.intent_detection.models import (
    IntentResult, IntentType, IntentLevel, IntentConfidence
)
from conversation_service.intent_detection.cache_manager import CacheManager
from conversation_service.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    🤖 Wrapper modèle embeddings TinyBERT optimisé
    
    Utilise sentence-transformers avec modèle léger pour français
    et cache embeddings pour éviter recalculs fréquents.
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ~120MB
        self.embedding_dim = 384
        self.max_sequence_length = 256
        
        # Cache embeddings local
        self._embedding_cache = {}
        self._cache_max_size = 1000
        
        # Thread pool pour calculs embeddings
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")
        
        # Métriques
        self._total_embeddings = 0
        self._cache_hits = 0
        
        logger.info("🤖 EmbeddingModel initialisé")
    
    async def initialize(self):
        """Initialisation asynchrone modèle avec fallback gracieux"""
        try:
            logger.info(f"📥 Chargement modèle embeddings: {self.model_name}")
            
            # Import et chargement dans thread pool pour éviter blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self._thread_pool,
                self._load_model_sync
            )
            
            # Test modèle avec phrase simple
            test_embedding = await self.encode_text("test modèle")
            if test_embedding is not None and len(test_embedding) == self.embedding_dim:
                logger.info(f"✅ Modèle embeddings chargé - dim: {self.embedding_dim}")
                return True
            else:
                raise ValueError("Test encoding failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger modèle embeddings: {e}")
            logger.info("💔 Lightweight Classifier L1 indisponible")
            self.model = None
            return False
    
    def _load_model_sync(self):
        """Chargement synchrone modèle (exécuté dans thread pool)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Chargement avec optimisations
            model = SentenceTransformer(
                self.model_name,
                cache_folder="/tmp/sentence_transformers",  # Cache local
                device="cpu"  # Force CPU pour compatibilité Heroku
            )
            
            # Configuration optimisations
            model.max_seq_length = self.max_sequence_length
            
            return model
            
        except ImportError:
            logger.error("❌ sentence-transformers non installé")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    async def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode texte en embedding avec cache local
        
        Args:
            text: Texte à encoder
            
        Returns:
            np.ndarray: Embedding 384-dim ou None si erreur
        """
        if not self.model or not text:
            return None
        
        # Vérification cache local
        text_hash = hash(text.lower().strip())
        if text_hash in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text_hash]
        
        try:
            self._total_embeddings += 1
            
            # Preprocessing texte
            preprocessed_text = self._preprocess_text(text)
            
            # Encoding dans thread pool pour non-blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._thread_pool,
                self._encode_sync,
                preprocessed_text
            )
            
            # Mise en cache si embedding valide
            if embedding is not None:
                self._cache_embedding(text_hash, embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur encoding texte: {e}")
            return None
    
    def _encode_sync(self, text: str) -> Optional[np.ndarray]:
        """Encoding synchrone (exécuté dans thread pool)"""
        try:
            # Encoding avec normalisation L2
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return embedding.astype(np.float32)  # Optimisation mémoire
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur encoding sync: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessing texte pour optimiser embeddings"""
        if not text:
            return ""
        
        # Nettoyage basique
        processed = text.lower().strip()
        
        # Troncature si trop long
        if len(processed) > self.max_sequence_length:
            processed = processed[:self.max_sequence_length]
        
        return processed
    
    def _cache_embedding(self, text_hash: int, embedding: np.ndarray):
        """Cache embedding local avec éviction LRU"""
        if len(self._embedding_cache) >= self._cache_max_size:
            # Éviction simple : supprime premier élément
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]
        
        self._embedding_cache[text_hash] = embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques cache embeddings"""
        cache_hit_rate = self._cache_hits / max(1, self._total_embeddings)
        
        return {
            "total_embeddings": self._total_embeddings,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self._cache_max_size,
            "model_loaded": self.model is not None
        }

class IntentEmbeddings:
    """
    📚 Base embeddings intentions pré-calculés
    
    Embeddings de référence pour chaque intention financière
    permettant classification par similarité cosinus rapide.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.intent_embeddings: Dict[IntentType, np.ndarray] = {}
        self.intent_examples: Dict[IntentType, List[str]] = {}
        self._initialize_intent_examples()
        
        logger.info("📚 IntentEmbeddings initialisé")
    
    def _initialize_intent_examples(self):
        """Définition exemples textuels par intention financière"""
        self.intent_examples = {
            # Balance/Solde
            IntentType.BALANCE_CHECK: [
                "quel est mon solde compte",
                "combien d'argent j'ai",
                "voir solde bancaire",
                "consulter mes comptes",
                "position financière actuelle"
            ],
            
            # Dépenses/Analyses
            IntentType.EXPENSE_ANALYSIS: [
                "analyser mes dépenses",
                "où va mon argent",
                "catégories de dépenses",
                "budget mensuel dépensé",
                "répartition des achats"
            ],
            
            # Transferts
            IntentType.TRANSFER: [
                "faire un virement",
                "envoyer de l'argent",
                "transférer fonds",
                "payer quelqu'un",
                "virer vers compte"
            ],
            
            # Factures
            IntentType.BILL_PAYMENT: [
                "payer une facture",
                "régler mes factures",
                "paiement automatique",
                "échéances à venir",
                "factures en attente"
            ],
            
            # Investissements
            IntentType.INVESTMENT_QUERY: [
                "mes investissements",
                "performance portefeuille",
                "actions et obligations",
                "rendements placements",
                "conseils investissement"
            ],
            
            # Gestion carte
            IntentType.CARD_MANAGEMENT: [
                "gérer ma carte bancaire",
                "bloquer carte perdue",
                "limites de paiement",
                "paramètres carte",
                "sécurité carte"
            ],
            
            # Historique
            IntentType.TRANSACTION_HISTORY: [
                "historique des transactions",
                "voir mes opérations",
                "détails des paiements",
                "relevé bancaire",
                "mouvements compte"
            ],
            
            # Budget
            IntentType.BUDGET_PLANNING: [
                "planifier mon budget",
                "objectifs financiers",
                "gestion budget mensuel",
                "prévoir les dépenses",
                "optimiser finances"
            ],
            
            # Épargne
            IntentType.SAVINGS_GOAL: [
                "objectifs d'épargne",
                "mettre de côté",
                "économiser pour projet",
                "plan épargne",
                "atteindre objectif financier"
            ],
            
            # Système
            IntentType.GREETING: [
                "bonjour",
                "salut comment ça va",
                "hello assistant",
                "bonsoir",
                "hey là"
            ],
            
            IntentType.HELP: [
                "j'ai besoin d'aide",
                "comment faire",
                "que puis-je faire",
                "assistance",
                "expliquer fonctionnalités"
            ],
            
            IntentType.UNKNOWN: [
                "texte incompréhensible",
                "requête non financière",
                "sujet hors contexte",
                "phrase sans sens",
                "demande non supportée"
            ]
        }
    
    async def build_intent_embeddings(self):
        """Construction embeddings de référence pour chaque intention"""
        logger.info("🔧 Construction embeddings intentions...")
        
        built_count = 0
        
        for intent_type, examples in self.intent_examples.items():
            try:
                # Calcul embeddings pour tous les exemples
                example_embeddings = []
                
                for example in examples:
                    embedding = await self.embedding_model.encode_text(example)
                    if embedding is not None:
                        example_embeddings.append(embedding)
                
                if example_embeddings:
                    # Moyenne des embeddings exemples pour embedding intention
                    intent_embedding = np.mean(example_embeddings, axis=0)
                    # Normalisation L2
                    intent_embedding = intent_embedding / np.linalg.norm(intent_embedding)
                    
                    self.intent_embeddings[intent_type] = intent_embedding
                    built_count += 1
                    
                    logger.debug(f"✅ Embedding construit: {intent_type.value}")
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur construction embedding {intent_type.value}: {e}")
        
        logger.info(f"✅ {built_count} embeddings intentions construits")
        return built_count > 0
    
    def classify_by_similarity(self, query_embedding: np.ndarray, confidence_threshold: float = 0.7) -> Optional[Tuple[IntentType, float]]:
        """
        Classification par similarité cosinus avec embeddings intentions
        
        Args:
            query_embedding: Embedding requête utilisateur
            confidence_threshold: Seuil confiance minimum
            
        Returns:
            Tuple[IntentType, float]: (intention, score_similarité) ou None
        """
        if not self.intent_embeddings or query_embedding is None:
            return None
        
        best_intent = None
        best_similarity = 0.0
        
        # Calcul similarité avec chaque intention
        for intent_type, intent_embedding in self.intent_embeddings.items():
            try:
                # Similarité cosinus (embeddings normalisés)
                similarity = np.dot(query_embedding, intent_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_intent = intent_type
                    
            except Exception as e:
                logger.debug(f"⚠️ Erreur calcul similarité {intent_type}: {e}")
        
        # Vérification seuil confiance
        if best_intent and best_similarity >= confidence_threshold:
            return best_intent, best_similarity
        
        return None
    
    def get_similarity_scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Scores similarité avec toutes les intentions (debug)"""
        scores = {}
        
        for intent_type, intent_embedding in self.intent_embeddings.items():
            try:
                similarity = np.dot(query_embedding, intent_embedding)
                scores[intent_type.value] = round(float(similarity), 3)
            except Exception:
                scores[intent_type.value] = 0.0
        
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

class LightweightClassifier:
    """
    🧠 Classificateur principal L1 TinyBERT
    
    Objectif: 15-30ms pour requêtes non matchées L0
    avec classification embeddings et extraction entités basique.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.embedding_model = EmbeddingModel()
        self.intent_embeddings = IntentEmbeddings(self.embedding_model)
        
        # Configuration seuils
        self.confidence_threshold = 0.75
        self.high_confidence_threshold = 0.85
        
        # Métriques performance
        self._total_classifications = 0
        self._successful_classifications = 0
        self._average_latency = 0.0
        
        # Status initialisation
        self._initialized = False
        
        logger.info("🧠 Lightweight Classifier L1 initialisé")
    
    async def initialize(self):
        """Initialisation avec chargement modèle et construction embeddings"""
        if self._initialized:
            return True
        
        try:
            logger.info("🔧 Initialisation Lightweight Classifier L1...")
            
            # 1. Initialisation modèle embeddings
            model_loaded = await self.embedding_model.initialize()
            if not model_loaded:
                logger.warning("⚠️ Modèle embeddings indisponible")
                return False
            
            # 2. Construction embeddings intentions
            embeddings_built = await self.intent_embeddings.build_intent_embeddings()
            if not embeddings_built:
                logger.warning("⚠️ Impossible de construire embeddings intentions")
                return False
            
            self._initialized = True
            logger.info("✅ Lightweight Classifier L1 initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Lightweight Classifier: {e}")
            return False
    
    async def classify_intent(self, query: str, user_id: str = "anonymous") -> Optional[IntentResult]:
        """
        Classification intention via embeddings TinyBERT
        
        Args:
            query: Requête utilisateur normalisée
            user_id: ID utilisateur pour contexte
            
        Returns:
            IntentResult: Résultat classification ou None si échec
        """
        if not self._initialized:
            logger.warning("⚠️ Lightweight Classifier non initialisé")
            return None
        
        start_time = time.time()
        self._total_classifications += 1
        
        try:
            # 1. Génération embedding requête
            query_embedding = await self.embedding_model.encode_text(query)
            if query_embedding is None:
                return None
            
            # 2. Classification par similarité
            classification_result = self.intent_embeddings.classify_by_similarity(
                query_embedding, 
                self.confidence_threshold
            )
            
            if not classification_result:
                return None
            
            intent_type, similarity_score = classification_result
            
            # 3. Extraction entités basique selon intention
            entities = await self._extract_basic_entities(query, intent_type)
            
            # 4. Construction résultat
            confidence = IntentConfidence.from_embedding_similarity(
                similarity=similarity_score,
                model_name="TinyBERT-multilingual"
            )
            
            result = IntentResult(
                intent_type=intent_type,
                confidence=confidence,
                level=IntentLevel.L1_LIGHTWEIGHT,
                latency_ms=0.0,  # Sera défini par appelant
                entities=entities,
                user_id=user_id,
                processing_details={
                    "similarity_score": similarity_score,
                    "embedding_model": self.embedding_model.model_name,
                    "confidence_threshold": self.confidence_threshold
                }
            )
            
            # 5. Métriques performance
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, success=True)
            
            self._successful_classifications += 1
            
            logger.debug(
                f"✅ L1 Classification: {intent_type.value} "
                f"(similarity: {similarity_score:.3f}, "
                f"latency: {processing_time:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time, success=False)
            
            logger.warning(f"⚠️ Erreur classification L1: {e}")
            return None
    
    async def _extract_basic_entities(self, query: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extraction entités basique selon type intention"""
        entities = {}
        
        try:
            query_lower = query.lower()
            
            # Extraction montants (regex simple)
            import re
            amount_pattern = r'(\d+(?:[,\.]\d{1,2})?)\s*(?:euros?|€|eur)'
            amount_matches = re.findall(amount_pattern, query_lower)
            
            if amount_matches:
                # Conversion première occurrence
                amount_str = amount_matches[0].replace(',', '.')
                entities["amount"] = float(amount_str)
                entities["currency"] = "EUR"
            
            # Extraction périodes temporelles
            time_patterns = {
                r'\b(?:ce\s+)?mois\b': 'month',
                r'\b(?:cette\s+)?semaine\b': 'week', 
                r'\b(?:cette\s+)?année\b': 'year',
                r'\baujou?rd\'?hui\b': 'today',
                r'\bhier\b': 'yesterday'
            }
            
            for pattern, period in time_patterns.items():
                if re.search(pattern, query_lower):
                    entities["time_period"] = period
                    break
            
            # Extraction catégories (mots-clés financiers)
            if intent_type == IntentType.EXPENSE_ANALYSIS:
                categories = {
                    'restaurant': ['restaurant', 'resto', 'nourriture', 'repas'],
                    'transport': ['transport', 'essence', 'métro', 'bus', 'taxi'],
                    'shopping': ['shopping', 'achat', 'vêtement', 'magasin'],
                    'logement': ['loyer', 'logement', 'maison', 'appartement'],
                    'loisirs': ['loisir', 'cinéma', 'sport', 'vacances']
                }
                
                for category, keywords in categories.items():
                    if any(keyword in query_lower for keyword in keywords):
                        entities["expense_category"] = category
                        break
            
            # Extraction types de comptes
            if intent_type == IntentType.BALANCE_CHECK:
                account_types = {
                    'courant': ['courant', 'chèque'],
                    'épargne': ['épargne', 'livret', 'pel'],
                    'investissement': ['investissement', 'action', 'bourse']
                }
                
                for account_type, keywords in account_types.items():
                    if any(keyword in query_lower for keyword in keywords):
                        entities["account_type"] = account_type
                        break
            
        except Exception as e:
            logger.debug(f"⚠️ Erreur extraction entités: {e}")
        
        return entities
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """Mise à jour métriques performance"""
        if self._average_latency == 0.0:
            self._average_latency = latency_ms
        else:
            # Smoothing exponentiel
            self._average_latency = 0.9 * self._average_latency + 0.1 * latency_ms
    
    # ==========================================
    # MÉTRIQUES ET STATUS
    # ==========================================
    
    async def get_status(self) -> Dict[str, Any]:
        """Status détaillé Lightweight Classifier"""
        success_rate = (
            self._successful_classifications / max(1, self._total_classifications)
        )
        
        status = {
            "initialized": self._initialized,
            "model_loaded": self.embedding_model.model is not None,
            "intent_embeddings_count": len(self.intent_embeddings.intent_embeddings),
            "total_classifications": self._total_classifications,
            "successful_classifications": self._successful_classifications,
            "success_rate": round(success_rate, 3),
            "average_latency_ms": round(self._average_latency, 2),
            "confidence_threshold": self.confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold
        }
        
        # Ajout stats modèle embeddings
        if self.embedding_model:
            embedding_stats = self.embedding_model.get_cache_stats()
            status["embedding_cache"] = embedding_stats
        
        return status
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Métriques détaillées pour monitoring"""
        base_status = await self.get_status()
        
        detailed = {
            **base_status,
            "model_info": {
                "model_name": self.embedding_model.model_name,
                "embedding_dim": self.embedding_model.embedding_dim,
                "max_sequence_length": self.embedding_model.max_sequence_length
            },
            "intent_examples_count": {
                intent.value: len(examples)
                for intent, examples in self.intent_embeddings.intent_examples.items()
            }
        }
        
        return detailed
    
    # ==========================================
    # MÉTHODES DEBUG ET TESTING
    # ==========================================
    
    async def test_classification(self, query: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test classification avec détails debug"""
        start_time = time.time()
        
        # Classification
        result = await self.classify_intent(query)
        
        # Génération embedding pour analyse détaillée
        query_embedding = await self.embedding_model.encode_text(query)
        
        test_result = {
            "query": query,
            "expected_intent": expected_intent,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        if result:
            test_result["classification"] = {
                "intent": result.intent_type.value,
                "confidence": result.confidence.score,
                "entities": result.entities,
                "processing_details": result.processing_details
            }
            
            # Vérification attente
            if expected_intent:
                test_result["expectation_met"] = (result.intent_type.value == expected_intent)
        else:
            test_result["classification"] = None
        
        # Scores similarité avec toutes intentions (debug)
        if query_embedding is not None:
            similarity_scores = self.intent_embeddings.get_similarity_scores(query_embedding)
            test_result["similarity_scores"] = similarity_scores
        
        return test_result
    
    async def benchmark_classification(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Benchmark classification sur cas de test"""
        logger.info(f"🏁 Benchmark L1 sur {len(test_cases)} cas...")
        
        start_time = time.time()
        results = []
        correct_predictions = 0
        total_latency = 0.0
        
        for case in test_cases:
            query = case["query"]
            expected = case.get("expected_intent")
            
            case_start = time.time()
            
            try:
                result = await self.classify_intent(query)
                case_latency = (time.time() - case_start) * 1000
                
                if result:
                    predicted_intent = result.intent_type.value
                    confidence = result.confidence.score
                    
                    # Vérification exactitude si attendu fourni
                    is_correct = (predicted_intent == expected) if expected else None
                    if is_correct:
                        correct_predictions += 1
                    
                    results.append({
                        "query": query,
                        "expected": expected,
                        "predicted": predicted_intent,
                        "confidence": confidence,
                        "latency_ms": case_latency,
                        "correct": is_correct,
                        "success": True
                    })
                else:
                    results.append({
                        "query": query,
                        "expected": expected,
                        "predicted": None,
                        "confidence": 0.0,
                        "latency_ms": case_latency,
                        "correct": False if expected else None,
                        "success": False
                    })
                
                total_latency += case_latency
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        total_time = (time.time() - start_time) * 1000
        
        # Calcul métriques benchmark
        successful_cases = sum(1 for r in results if r.get("success", False))
        cases_with_expected = sum(1 for case in test_cases if case.get("expected_intent"))
        
        accuracy = (correct_predictions / cases_with_expected) if cases_with_expected > 0 else 0.0
        
        benchmark_results = {
            "total_cases": len(test_cases),
            "successful_classifications": successful_cases,
            "cases_with_expected": cases_with_expected,
            "correct_predictions": correct_predictions,
            "accuracy": round(accuracy, 3),
            "success_rate": round(successful_cases / len(test_cases), 3),
            "total_time_ms": round(total_time, 2),
            "average_latency_ms": round(total_latency / len(test_cases), 2),
            "target_latency_met": (total_latency / len(test_cases)) < 30.0,
            "results_sample": results[:10]  # Échantillon pour éviter spam
        }
        
        logger.info(f"🏁 Benchmark L1 terminé - Accuracy: {accuracy:.1%}, Avg latency: {benchmark_results['average_latency_ms']:.1f}ms")
        return benchmark_results
    
    async def analyze_similarity_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyse patterns de similarité pour optimisation"""
        similarity_analysis = {
            "intent_distributions": {},
            "confidence_distributions": {},
            "low_confidence_queries": [],
            "high_confidence_queries": []
        }
        
        for query in queries:
            query_embedding = await self.embedding_model.encode_text(query)
            if query_embedding is None:
                continue
            
            # Classification
            classification_result = self.intent_embeddings.classify_by_similarity(
                query_embedding, confidence_threshold=0.0  # Pas de seuil pour analyse
            )
            
            if classification_result:
                intent, confidence = classification_result
                
                # Distribution intentions
                intent_name = intent.value
                similarity_analysis["intent_distributions"][intent_name] = \
                    similarity_analysis["intent_distributions"].get(intent_name, 0) + 1
                
                # Distribution confiances
                conf_bucket = f"{int(confidence * 10) / 10:.1f}"
                similarity_analysis["confidence_distributions"][conf_bucket] = \
                    similarity_analysis["confidence_distributions"].get(conf_bucket, 0) + 1
                
                # Queries problématiques
                if confidence < 0.7:
                    similarity_analysis["low_confidence_queries"].append({
                        "query": query,
                        "intent": intent_name,
                        "confidence": round(confidence, 3)
                    })
                elif confidence > 0.9:
                    similarity_analysis["high_confidence_queries"].append({
                        "query": query,
                        "intent": intent_name,
                        "confidence": round(confidence, 3)
                    })
        
        return similarity_analysis
    
    async def optimize_thresholds(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Optimisation seuils confiance basée sur cas de test"""
        logger.info("🎯 Optimisation seuils confiance...")
        
        threshold_candidates = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        optimization_results = {}
        
        for threshold in threshold_candidates:
            # Test temporaire avec ce seuil
            original_threshold = self.confidence_threshold
            self.confidence_threshold = threshold
            
            # Benchmark avec ce seuil
            benchmark = await self.benchmark_classification(test_cases)
            
            optimization_results[threshold] = {
                "accuracy": benchmark["accuracy"],
                "success_rate": benchmark["success_rate"],
                "average_latency_ms": benchmark["average_latency_ms"],
                "score": benchmark["accuracy"] * 0.7 + benchmark["success_rate"] * 0.3  # Score pondéré
            }
            
            # Restauration seuil original
            self.confidence_threshold = original_threshold
        
        # Seuil optimal = meilleur score pondéré
        best_threshold = max(optimization_results.keys(), key=lambda t: optimization_results[t]["score"])
        
        return {
            "current_threshold": self.confidence_threshold,
            "optimal_threshold": best_threshold,
            "threshold_analysis": optimization_results,
            "improvement_potential": optimization_results[best_threshold]["score"] - optimization_results[self.confidence_threshold]["score"]
        }
    
    async def shutdown(self):
        """Arrêt propre Lightweight Classifier"""
        logger.info("🛑 Arrêt Lightweight Classifier L1...")
        
        try:
            # Stats finales
            final_status = await self.get_status()
            logger.info(f"📊 Stats finales L1: "
                       f"Success rate = {final_status['success_rate']:.1%}, "
                       f"Avg latency = {final_status['average_latency_ms']:.1f}ms")
            
            # Fermeture thread pool
            if hasattr(self.embedding_model, '_thread_pool'):
                self.embedding_model._thread_pool.shutdown(wait=True)
            
            # Clear caches
            self.embedding_model._embedding_cache.clear()
            
            logger.info("✅ Lightweight Classifier L1 arrêté")
            
        except Exception as e:
            logger.error(f"❌ Erreur arrêt Lightweight Classifier: {e}")


# ==========================================
# UTILITAIRES ET HELPERS
# ==========================================

def create_l1_test_cases() -> List[Dict[str, str]]:
    """Cas de test pour validation Lightweight Classifier"""
    return [
        # Balance checks
        {"query": "combien d'argent me reste-t-il", "expected_intent": "BALANCE_CHECK"},
        {"query": "voir position de mes comptes", "expected_intent": "BALANCE_CHECK"},
        {"query": "solde compte épargne actuel", "expected_intent": "BALANCE_CHECK"},
        
        # Expense analysis
        {"query": "analyse de mes dépenses shopping", "expected_intent": "EXPENSE_ANALYSIS"},
        {"query": "où va mon budget ce mois", "expected_intent": "EXPENSE_ANALYSIS"},
        {"query": "combien dépensé en restaurant", "expected_intent": "EXPENSE_ANALYSIS"},
        
        # Transfers
        {"query": "envoyer 200 euros à marie", "expected_intent": "TRANSFER"},
        {"query": "faire virement vers compte ami", "expected_intent": "TRANSFER"},
        {"query": "transférer fonds urgents", "expected_intent": "TRANSFER"},
        
        # Card management
        {"query": "limites de ma carte bleue", "expected_intent": "CARD_MANAGEMENT"},
        {"query": "sécuriser carte bancaire", "expected_intent": "CARD_MANAGEMENT"},
        {"query": "paramètres carte de crédit", "expected_intent": "CARD_MANAGEMENT"},
        
        # Investment
        {"query": "performance de mon portefeuille", "expected_intent": "INVESTMENT_QUERY"},
        {"query": "rendement des placements", "expected_intent": "INVESTMENT_QUERY"},
        {"query": "conseils pour investir", "expected_intent": "INVESTMENT_QUERY"},
        
        # System
        {"query": "salut comment ça marche", "expected_intent": "GREETING"},
        {"query": "j'ai besoin d'assistance", "expected_intent": "HELP"},
        {"query": "merci et bonne soirée", "expected_intent": "GOODBYE"},
        
        # Edge cases  
        {"query": "xyz abc 123", "expected_intent": "UNKNOWN"},
        {"query": "météo aujourd'hui", "expected_intent": "UNKNOWN"}
    ]

async def validate_l1_performance(classifier: LightweightClassifier) -> Dict[str, Any]:
    """Validation performance L1 selon targets"""
    test_cases = create_l1_test_cases()
    
    # Benchmark principal
    benchmark = await classifier.benchmark_classification(test_cases)
    
    # Optimisation seuils
    optimization = await classifier.optimize_thresholds(test_cases)
    
    # Validation targets L1
    target_latency = 30.0  # ms
    target_accuracy = 0.80
    
    validation = {
        "performance_validation": {
            "target_latency_ms": target_latency,
            "actual_avg_latency_ms": benchmark["average_latency_ms"],
            "latency_target_met": benchmark["average_latency_ms"] < target_latency,
            
            "target_accuracy": target_accuracy,
            "actual_accuracy": benchmark["accuracy"],
            "accuracy_target_met": benchmark["accuracy"] >= target_accuracy
        },
        "benchmark_results": benchmark,
        "threshold_optimization": optimization,
        "classifier_status": await classifier.get_detailed_metrics()
    }
    
    return validation