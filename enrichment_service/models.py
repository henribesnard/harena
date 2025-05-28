"""
Modèles de données pour le service d'enrichissement.

Ce module définit les structures de données utilisées pour l'enrichissement
et le stockage vectoriel des transactions financières.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Modèles Pydantic pour l'API
class TransactionInput(BaseModel):
    """Modèle d'entrée pour une transaction à traiter."""
    bridge_transaction_id: int
    user_id: int
    account_id: int
    clean_description: Optional[str] = None
    provider_description: Optional[str] = None
    amount: float
    date: datetime
    booking_date: Optional[datetime] = None
    transaction_date: Optional[datetime] = None
    value_date: Optional[datetime] = None
    currency_code: Optional[str] = None
    category_id: Optional[int] = None
    operation_type: Optional[str] = None
    deleted: bool = False
    future: bool = False

class BatchTransactionInput(BaseModel):
    """Modèle pour le traitement en lot de transactions."""
    user_id: int
    transactions: List[TransactionInput]

class EnrichmentResult(BaseModel):
    """Résultat d'un enrichissement de transaction."""
    transaction_id: int
    user_id: int
    searchable_text: str
    vector_id: str
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
    results: List[EnrichmentResult]
    errors: List[str] = []

# Modèles pour le stockage vectoriel
@dataclass
class VectorizedTransaction:
    """Structure pour une transaction vectorisée prête pour Qdrant."""
    id: str  # Format: user_{user_id}_tx_{transaction_id}
    vector: List[float]
    payload: Dict[str, Any]
    
    def to_qdrant_point(self) -> Dict[str, Any]:
        """Convertit en format point Qdrant."""
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload
        }

@dataclass 
class StructuredTransaction:
    """Transaction structurée pour la recherche."""
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
    
    # Métadonnées supplémentaires
    is_future: bool
    is_deleted: bool
    
    @classmethod
    def from_transaction_input(cls, tx: TransactionInput) -> 'StructuredTransaction':
        """Crée une StructuredTransaction à partir d'une TransactionInput."""
        # Description principale
        primary_desc = tx.clean_description or tx.provider_description or "Transaction"
        
        # Type de transaction
        tx_type = "debit" if tx.amount < 0 else "credit"
        
        # Formatage des dates
        date_str = tx.date.strftime('%Y-%m-%d')
        month_year = tx.date.strftime('%Y-%m')
        weekday = tx.date.strftime('%A')
        
        # Création du texte recherchable
        searchable_parts = [
            f"Description: {primary_desc}",
            f"Montant: {abs(tx.amount):.2f} {tx.currency_code or 'EUR'}",
            f"Type: {tx_type}",
            f"Date: {date_str}"
        ]
        
        if tx.operation_type:
            searchable_parts.append(f"Opération: {tx.operation_type}")
            
        if tx.category_id:
            searchable_parts.append(f"Catégorie: {tx.category_id}")
        
        searchable_text = " | ".join(searchable_parts)
        
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
            is_deleted=tx.deleted
        )
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convertit en payload pour Qdrant (métadonnées + texte)."""
        return {
            # Identifiants
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "account_id": self.account_id,
            
            # Contenu
            "searchable_text": self.searchable_text,
            "primary_description": self.primary_description,
            
            # Données financières
            "amount": self.amount,
            "amount_abs": self.amount_abs,
            "transaction_type": self.transaction_type,
            "currency_code": self.currency_code,
            
            # Dates (stockées en string pour les filtres)
            "date": self.date_str,
            "month_year": self.month_year,
            "weekday": self.weekday,
            "timestamp": self.date.timestamp(),
            
            # Catégorisation
            "category_id": self.category_id,
            "operation_type": self.operation_type,
            
            # Flags
            "is_future": self.is_future,
            "is_deleted": self.is_deleted,
            
            # Métadonnées de traitement
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }

# Modèles pour les réponses de recherche
class SearchResult(BaseModel):
    """Résultat d'une recherche vectorielle."""
    transaction_id: int
    user_id: int
    score: float
    primary_description: str
    amount: float
    date: str
    transaction_type: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    """Réponse complète d'une recherche."""
    query: str
    results: List[SearchResult]
    total_found: int
    processing_time: float