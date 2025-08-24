"""Service d'enrichissement de compte avec métriques et logging structuré."""
import logging
import time
import uuid
from typing import List

from prometheus_client import Counter, Gauge, Histogram

from enrichment_service.core.processor import ElasticsearchTransactionProcessor
from enrichment_service.models import (
    BatchEnrichmentResult,
    BatchTransactionInput,
    TransactionInput,
)

logger = logging.getLogger(__name__)

# Métriques spécifiques au service d'enrichissement de compte
ACCOUNT_ENRICH_TIME = Histogram(
    "account_enrichment_seconds",
    "Temps de traitement d'un compte"
)
ACCOUNT_CACHE_HITS = Counter(
    "account_enrichment_cache_hits_total",
    "Nombre de transactions traitées depuis le cache"
)
ACCOUNT_ERRORS = Counter(
    "account_enrichment_errors_total",
    "Nombre d'erreurs lors de l'enrichissement de compte"
)
DATA_QUALITY_SCORE = Gauge(
    "account_enrichment_data_quality_score",
    "Score de qualité des données pour un lot de transactions"
)


class AccountEnrichmentService:
    """Service orchestrant l'enrichissement des transactions d'un compte."""

    def __init__(self, processor: ElasticsearchTransactionProcessor):
        self.processor = processor

    async def enrich_account(
        self,
        user_id: int,
        transactions: List[TransactionInput],
        force_update: bool = False,
    ) -> BatchEnrichmentResult:
        """Enrichit toutes les transactions d'un compte utilisateur."""
        correlation_id = str(uuid.uuid4())
        log = logging.LoggerAdapter(
            logger, {"correlation_id": correlation_id, "user_id": user_id}
        )
        start_time = time.perf_counter()

        try:
            batch_input = BatchTransactionInput(
                user_id=user_id, transactions=transactions
            )
            result = await self.processor.process_transactions_batch(
                batch_input, force_update=force_update
            )

            elapsed = time.perf_counter() - start_time
            ACCOUNT_ENRICH_TIME.observe(elapsed)

            cache_hits = len([r for r in result.results if r.status == "skipped"])
            if cache_hits:
                ACCOUNT_CACHE_HITS.inc(cache_hits)

            quality = (
                result.successful / result.total_transactions
                if result.total_transactions
                else 0
            )
            DATA_QUALITY_SCORE.set(quality)

            log.info(
                "Enrichissement de compte terminé",
                extra={"cache_hits": cache_hits, "data_quality_score": quality},
            )
            return result
        except Exception as e:
            ACCOUNT_ERRORS.inc()
            ACCOUNT_ENRICH_TIME.observe(time.perf_counter() - start_time)
            log.error(f"Erreur enrichissement compte: {e}")
            raise
