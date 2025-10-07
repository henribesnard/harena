"""
Modèles de données pour le service d'enrichissement - Elasticsearch uniquement.

Ce module définit les structures de données utilisées pour l'enrichissement
et l'indexation des transactions financières dans Elasticsearch.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Modèles Pydantic pour l'API
class TransactionInput(BaseModel):
    """Modèle d'entrée pour une transaction à traiter.

    Note: ``account_id`` correspond au ``bridge_account_id`` (identifiant métier)
    du compte dans PostgreSQL, et non à l'ID interne.
    """
    bridge_transaction_id: int
    user_id: int
    account_id: int  # bridge_account_id
    account_name: Optional[str] = None
    account_type: Optional[str] = None
    account_balance: Optional[float] = None
    account_currency: Optional[str] = None
    account_last_sync: Optional[datetime] = None
    clean_description: Optional[str] = None
    provider_description: Optional[str] = None
    amount: float
    date: datetime
    booking_date: Optional[datetime] = None
    transaction_date: Optional[datetime] = None
    value_date: Optional[datetime] = None
    currency_code: Optional[str] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    merchant_name: Optional[str] = None
    operation_type: Optional[str] = None
    deleted: bool = False
    future: bool = False
    recent_transactions: List[float] = Field(default_factory=list)


class BatchTransactionInput(BaseModel):
    """Modèle pour le traitement en lot de transactions."""
    user_id: int
    transactions: List[TransactionInput]

class ElasticsearchEnrichmentResult(BaseModel):
    """Résultat d'un enrichissement de transaction pour Elasticsearch."""
    transaction_id: int
    user_id: int
    searchable_text: str
    document_id: str  # Format: user_{user_id}_tx_{transaction_id}
    indexed: bool
    metadata: Dict[str, Any]
    processing_time: float
    status: str = "success"
    error_message: Optional[str] = None

class BatchEnrichmentResult(BaseModel):
    """Résultat d'un enrichissement en lot."""
    user_id: int
    total_transactions: int
    successful: int
    failed: int
    processing_time: float
    results: List[ElasticsearchEnrichmentResult]
    errors: List[str] = []

class UserSyncResult(BaseModel):
    """Résultat de synchronisation utilisateur."""
    user_id: int
    total_transactions: int
    transactions_indexed: int
    accounts_indexed: int = 0
    updated: int
    errors: int
    with_account_metadata: int = 0
    accounts_synced: int = 0
    processing_time: float
    status: str = "success"
    error_details: List[str] = []

    @property
    def indexed(self) -> int:
        return self.transactions_indexed

@dataclass
class StructuredTransaction:
    """Transaction structurée pour l'indexation Elasticsearch."""

    # Identifiants
    transaction_id: int
    user_id: int
    account_id: int

    # Contenu principal
    searchable_text: str
    primary_description: str

    # Données financières
    amount: float
    amount_abs: float
    transaction_type: str  # "debit" ou "credit"
    currency_code: str

    # Dates
    date: datetime
    date_str: str
    month_year: str
    weekday: str

    # Catégorisation
    category_id: Optional[int]
    operation_type: Optional[str]

    # Flags
    is_future: bool
    is_deleted: bool

    # Résultats qualité et métadonnées
    balance_check_passed: Optional[bool] = None
    quality_score: Optional[float] = None

    # SUPPRIMÉ : Informations de compte (maintenant dans index accounts séparé)
    # Seul account_id reste pour le lien

    # Information sur la catégorie
    category_name: Optional[str] = None
    merchant_name: Optional[str] = None

    @classmethod
    def from_transaction_input(cls, tx: TransactionInput) -> "StructuredTransaction":
        """Crée une StructuredTransaction à partir d'une TransactionInput."""
        primary_desc = tx.clean_description or tx.provider_description or "Transaction"
        tx_type = "debit" if tx.amount < 0 else "credit"

        date_str = tx.date.strftime("%Y-%m-%d")
        month_year = tx.date.strftime("%Y-%m")
        weekday = tx.date.strftime("%A")

        # 🔄 SYNCHRONISATION AUTOMATIQUE: Récupérer category_name depuis PostgreSQL
        category_name = tx.category_name  # Utiliser d'abord la valeur fournie
        if tx.category_id and not category_name:
            # Si category_id existe mais pas de nom, récupérer depuis PostgreSQL
            try:
                from .category_service import get_category_service
                category_service = get_category_service()
                category_name = category_service.get_category_name(tx.category_id)
            except Exception as e:
                # En cas d'erreur, continuer sans bloquer l'enrichissement
                category_name = None

        searchable_parts = [
            f"Description: {primary_desc}",
            f"Montant: {abs(tx.amount):.2f} {tx.currency_code or 'EUR'}",
            f"Type: {tx_type}",
            f"Date: {date_str}",
        ]
        if tx.operation_type:
            searchable_parts.append(f"Opération: {tx.operation_type}")
        if tx.category_id:
            searchable_parts.append(f"Catégorie: {tx.category_id}")
        if category_name:
            searchable_parts.append(f"Nom catégorie: {category_name}")
        searchable_text = " | ".join(searchable_parts)

        balance_check_passed = None
        quality_score = None
        if tx.account_balance is not None and tx.recent_transactions:
            from .data_quality import DataQualityValidator

            validator = DataQualityValidator()
            result = validator.validate_account_balance_consistency(
                tx.account_balance, tx.recent_transactions
            )
            balance_check_passed = result.get("balance_check_passed")
            quality_score = result.get("quality_score")

        return cls(
            transaction_id=tx.bridge_transaction_id,
            user_id=tx.user_id,
            account_id=tx.account_id,
            searchable_text=searchable_text,
            primary_description=primary_desc,
            amount=tx.amount,
            amount_abs=abs(tx.amount),
            transaction_type=tx_type,
            currency_code=tx.currency_code or "EUR",
            date=tx.date,
            date_str=date_str,
            month_year=month_year,
            weekday=weekday,
            category_id=tx.category_id,
            operation_type=tx.operation_type,
            is_future=tx.future,
            is_deleted=tx.deleted,
            category_name=category_name,  # ✅ Nom récupéré automatiquement
            balance_check_passed=balance_check_passed,
            quality_score=quality_score,

        )

    def to_elasticsearch_document(self) -> Dict[str, Any]:
        """Convertit en document Elasticsearch."""
        doc = {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "account_id": self.account_id,  # 🔗 Lien vers index accounts
            "searchable_text": self.searchable_text,
            "primary_description": self.primary_description,
            "amount": self.amount,
            "amount_abs": self.amount_abs,
            "transaction_type": self.transaction_type,
            "currency_code": self.currency_code,

            # Dates (optimisées pour les requêtes Elasticsearch)
            "date": self.date.isoformat(),
            "date_str": self.date_str,
            "month_year": self.month_year,
            "weekday": self.weekday,
            "timestamp": self.date.timestamp(),
            "category_id": self.category_id,
            "operation_type": self.operation_type,
            "category_name": self.category_name,
            "merchant_name": self.merchant_name,

            # Flags

            "is_future": self.is_future,
            "is_deleted": self.is_deleted,
            "indexed_at": datetime.now().isoformat(),
            "version": "2.0-elasticsearch",
        }
        if self.balance_check_passed is not None:
            doc["balance_check_passed"] = self.balance_check_passed
        if self.quality_score is not None:
            doc["quality_score"] = self.quality_score
        return doc
    
    def get_document_id(self) -> str:
        """Génère l'ID du document Elasticsearch."""
        return f"user_{self.user_id}_tx_{self.transaction_id}"

class MerchantEnrichmentResult(BaseModel):
    """Résultat d'enrichissement de marchand pour une transaction."""
    transaction_id: int
    original_description: str
    merchant_name: Optional[str]
    confidence: float
    processing_time: float
    status: str  # "success", "skipped", "error"
    error_message: Optional[str] = None

class BatchMerchantEnrichmentResult(BaseModel):
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

class ElasticsearchHealthStatus(BaseModel):
    """Statut de santé du service Elasticsearch."""
    service: str = "enrichment_service_elasticsearch"
    version: str = "2.0.0"
    status: str
    timestamp: str
    elasticsearch: Dict[str, Any]
    database: Optional[Dict[str, Any]] = None
    capabilities: Dict[str, bool]
    performance_metrics: Optional[Dict[str, Any]] = None
