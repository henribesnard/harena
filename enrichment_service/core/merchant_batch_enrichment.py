"""
Service d'enrichissement par lot des marchands avec Deepseek LLM.

Ce service traite spécifiquement l'extraction de noms de marchands 
par lots pour optimiser les coûts et les performances.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session
from db_service.models.sync import RawTransaction
from enrichment_service.core.llm_merchant_service import get_merchant_service
from enrichment_service.storage.elasticsearch_client import ElasticsearchClient

logger = logging.getLogger(__name__)

@dataclass
class MerchantEnrichmentResult:
    """Résultat d'enrichissement de marchand pour une transaction."""
    transaction_id: int
    original_description: str
    merchant_name: Optional[str]
    confidence: float
    processing_time: float
    status: str  # "success", "skipped", "error"
    error_message: Optional[str] = None

@dataclass
class BatchMerchantEnrichmentResult:
    """Résultat d'enrichissement par lot."""
    user_id: int
    total_transactions: int
    processed: int
    successful_extractions: int
    skipped: int
    errors: int
    total_processing_time: float
    average_time_per_transaction: float
    results: List[MerchantEnrichmentResult]
    cost_estimate: Dict[str, Any]

class MerchantBatchEnrichmentService:
    """Service d'enrichissement de marchands par lot."""
    
    def __init__(self, db: Session, elasticsearch_client: ElasticsearchClient):
        self.db = db
        self.es_client = elasticsearch_client
        self.merchant_service = get_merchant_service()
        
        # Configuration par défaut
        self.confidence_threshold = 0.3
        self.batch_size = 50  # Traiter par petits lots pour éviter les timeouts
        self.delay_between_requests = 0.1  # Délai entre appels LLM pour éviter rate limiting
        
    async def enrich_user_merchants(
        self,
        user_id: int,
        limit: Optional[int] = None,
        only_missing: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> BatchMerchantEnrichmentResult:
        """
        Enrichit les noms de marchands pour un utilisateur.
        
        Args:
            user_id: ID utilisateur
            limit: Limite du nombre de transactions à traiter
            only_missing: Ne traiter que les transactions sans merchant_name
            confidence_threshold: Seuil de confiance (défaut 0.3)
            
        Returns:
            BatchMerchantEnrichmentResult: Résultat du traitement
        """
        start_time = time.perf_counter()
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        logger.info(f"🤖 Démarrage enrichissement marchands pour user {user_id}")
        
        # 1. Récupérer les transactions à traiter
        transactions = await self._get_transactions_to_process(
            user_id, limit, only_missing
        )
        
        if not transactions:
            logger.info(f"Aucune transaction à traiter pour user {user_id}")
            return self._create_empty_result(user_id)
        
        logger.info(f"📋 {len(transactions)} transactions à traiter")
        
        # 2. Traitement par lots
        all_results = []
        processed = 0
        successful = 0
        skipped = 0
        errors = 0
        
        # Traiter par petits lots pour éviter les timeouts
        for i in range(0, len(transactions), self.batch_size):
            batch = transactions[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(transactions) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"🔄 Traitement lot {batch_num}/{total_batches} ({len(batch)} transactions)")
            
            batch_results = await self._process_transaction_batch(batch)
            all_results.extend(batch_results)
            
            # Statistiques du lot
            batch_successful = sum(1 for r in batch_results if r.status == "success")
            batch_skipped = sum(1 for r in batch_results if r.status == "skipped") 
            batch_errors = sum(1 for r in batch_results if r.status == "error")
            
            processed += len(batch)
            successful += batch_successful
            skipped += batch_skipped
            errors += batch_errors
            
            logger.info(f"✅ Lot {batch_num} terminé: {batch_successful} succès, {batch_skipped} ignorés, {batch_errors} erreurs")
        
        # 3. Mise à jour Elasticsearch
        update_count = await self._update_elasticsearch_documents(all_results, user_id)
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / len(transactions) if transactions else 0
        
        # 4. Estimation des coûts
        cost_estimate = self._estimate_costs(successful)
        
        logger.info(
            f"🎉 Enrichissement terminé: {successful}/{len(transactions)} extractions réussies en {total_time:.2f}s"
        )
        
        return BatchMerchantEnrichmentResult(
            user_id=user_id,
            total_transactions=len(transactions),
            processed=processed,
            successful_extractions=successful,
            skipped=skipped,
            errors=errors,
            total_processing_time=total_time,
            average_time_per_transaction=avg_time,
            results=all_results,
            cost_estimate=cost_estimate
        )
    
    async def _get_transactions_to_process(
        self, 
        user_id: int, 
        limit: Optional[int], 
        only_missing: bool
    ) -> List[RawTransaction]:
        """Récupère les transactions à traiter."""
        
        query = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.deleted == False
        )
        
        # Filtrer les transactions sans description
        query = query.filter(
            (RawTransaction.clean_description.isnot(None)) |
            (RawTransaction.provider_description.isnot(None))
        )
        
        if only_missing:
            # TODO: Ajouter filtre pour ne prendre que celles sans merchant_name dans ES
            # Pour l'instant, on prend toutes les transactions
            pass
        
        query = query.order_by(RawTransaction.date.desc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    async def _process_transaction_batch(
        self, 
        transactions: List[RawTransaction]
    ) -> List[MerchantEnrichmentResult]:
        """Traite un lot de transactions."""
        
        results = []
        
        for tx in transactions:
            # Délai entre requêtes pour éviter rate limiting
            if results:  # Pas de délai pour la première requête
                await asyncio.sleep(self.delay_between_requests)
            
            start_time = time.perf_counter()
            description = tx.clean_description or tx.provider_description or ""
            
            try:
                if not description.strip():
                    results.append(MerchantEnrichmentResult(
                        transaction_id=tx.bridge_transaction_id,
                        original_description=description,
                        merchant_name=None,
                        confidence=0.0,
                        processing_time=0,
                        status="skipped",
                        error_message="Empty description"
                    ))
                    continue
                
                # Extraction LLM
                extraction_result = await self.merchant_service.extract_merchant_name(
                    description=description,
                    amount=tx.amount
                )
                
                processing_time = time.perf_counter() - start_time
                
                if extraction_result.merchant_name and extraction_result.confidence >= self.confidence_threshold:
                    status = "success"
                    merchant_name = extraction_result.merchant_name
                    logger.debug(f"✅ TX {tx.bridge_transaction_id}: {merchant_name} (conf: {extraction_result.confidence:.2f})")
                else:
                    status = "skipped"
                    merchant_name = None
                    logger.debug(f"⏭️ TX {tx.bridge_transaction_id}: confiance trop faible ({extraction_result.confidence:.2f})")
                
                results.append(MerchantEnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    original_description=description,
                    merchant_name=merchant_name,
                    confidence=extraction_result.confidence,
                    processing_time=processing_time,
                    status=status
                ))
                
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                logger.error(f"❌ Erreur TX {tx.bridge_transaction_id}: {e}")
                
                results.append(MerchantEnrichmentResult(
                    transaction_id=tx.bridge_transaction_id,
                    original_description=description,
                    merchant_name=None,
                    confidence=0.0,
                    processing_time=processing_time,
                    status="error",
                    error_message=str(e)
                ))
        
        return results
    
    async def _update_elasticsearch_documents(
        self, 
        results: List[MerchantEnrichmentResult],
        user_id: int
    ) -> int:
        """Met à jour les documents Elasticsearch avec les noms de marchands."""
        
        successful_results = [r for r in results if r.status == "success" and r.merchant_name]
        
        if not successful_results:
            return 0
        
        logger.info(f"📝 Mise à jour de {len(successful_results)} documents Elasticsearch")
        
        try:
            # Préparer les mises à jour
            updates = []
            for result in successful_results:
                # CORRECTION: Format cohérent avec le système d'indexation  
                doc_id = f"user_{user_id}_tx_{result.transaction_id}"
                updates.append({
                    "id": doc_id,
                    "update": {
                        "merchant_name": result.merchant_name
                    }
                })
            
            # Appel bulk update Elasticsearch - maintenant implémenté!
            update_result = await self.es_client.bulk_update_documents(updates)
            
            logger.info(f"✅ Mis à jour {update_result['updated']}/{update_result['total']} documents Elasticsearch")
            
            return update_result['updated']
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour Elasticsearch: {e}")
            return 0
    
    def _estimate_costs(self, successful_extractions: int) -> Dict[str, Any]:
        """Estime les coûts de l'enrichissement."""
        
        # Estimation basée sur les coûts Deepseek (approximatif)
        cost_per_request = 0.0001  # ~0.01 centime par requête
        total_cost = successful_extractions * cost_per_request
        
        return {
            "successful_extractions": successful_extractions,
            "estimated_cost_usd": round(total_cost, 4),
            "cost_per_extraction": cost_per_request,
            "currency": "USD"
        }
    
    def _create_empty_result(self, user_id: int) -> BatchMerchantEnrichmentResult:
        """Crée un résultat vide."""
        return BatchMerchantEnrichmentResult(
            user_id=user_id,
            total_transactions=0,
            processed=0,
            successful_extractions=0,
            skipped=0,
            errors=0,
            total_processing_time=0.0,
            average_time_per_transaction=0.0,
            results=[],
            cost_estimate=self._estimate_costs(0)
        )