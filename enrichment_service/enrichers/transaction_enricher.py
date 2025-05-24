"""
Enrichisseur de transactions bancaires.

Ce module traite les transactions brutes pour en extraire du contexte,
normaliser les descriptions et générer des embeddings vectoriels.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy.orm import Session
from uuid import uuid4

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import TransactionEnrichmentError, handle_enrichment_error
from enrichment_service.db.models import RawTransaction, SyncAccount, BridgeCategory

logger = logging.getLogger(__name__)

class TransactionEnricher:
    """
    Enrichisseur principal pour les transactions bancaires.
    
    Cette classe transforme les transactions brutes en données enrichies
    avec contexte, métadonnées et embeddings vectoriels.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service, merchant_normalizer):
        """
        Initialise l'enrichisseur de transactions.
        
        Args:
            db: Session de base de données
            embedding_service: Service de génération d'embeddings
            qdrant_service: Service Qdrant pour le stockage vectoriel
            merchant_normalizer: Service de normalisation des marchands
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.merchant_normalizer = merchant_normalizer
        
        # Initialiser les catégories en cache
        self._categories_cache = {}
        self._load_categories_cache()
    
    def _load_categories_cache(self):
        """Charge les catégories Bridge en cache pour un accès rapide."""
        try:
            categories = self.db.query(BridgeCategory).all()
            self._categories_cache = {cat.bridge_category_id: cat for cat in categories}
            logger.debug(f"Cache des catégories chargé: {len(self._categories_cache)} catégories")
        except Exception as e:
            logger.warning(f"Impossible de charger le cache des catégories: {e}")
    
    @log_performance
    async def enrich_transaction(self, transaction: RawTransaction) -> Dict[str, Any]:
        """
        Enrichit une transaction unique.
        
        Args:
            transaction: Transaction brute à enrichir
            
        Returns:
            Dict: Résultat de l'enrichissement
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="transaction"
        )
        
        try:
            # 1. Enrichir les métadonnées de base
            enriched_data = await self._extract_base_metadata(transaction)
            
            # 2. Normaliser la description
            enriched_data.update(await self._normalize_description(transaction))
            
            # 3. Extraire et normaliser le marchand
            if self.merchant_normalizer:
                merchant_info = await self.merchant_normalizer.normalize_merchant(
                    transaction.clean_description or transaction.provider_description
                )
                enriched_data.update(merchant_info)
            
            # 4. Classifier la transaction
            enriched_data.update(await self._classify_transaction(transaction, enriched_data))
            
            # 5. Générer la description enrichie
            enriched_description = await self._generate_enriched_description(transaction, enriched_data)
            enriched_data["enriched_description"] = enriched_description
            
            # 6. Générer l'embedding vectoriel
            vector = await self._generate_embedding(enriched_description, enriched_data)
            
            # 7. Stocker dans Qdrant
            point_id = str(uuid4())
            qdrant_payload = await self._build_qdrant_payload(transaction, enriched_data, point_id)
            
            await self.qdrant_service.upsert_point(
                collection_name="enriched_transactions",
                point_id=point_id,
                vector=vector,
                payload=qdrant_payload
            )
            
            ctx_logger.info(f"Transaction enrichie et vectorisée avec succès (point_id: {point_id})")
            
            return {
                "status": "success",
                "point_id": point_id,
                "enriched_data": enriched_data,
                "vector_dimensions": len(vector) if vector else 0
            }
            
        except Exception as e:
            error_msg = f"Erreur lors de l'enrichissement de la transaction {transaction.id}: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise TransactionEnrichmentError(error_msg, transaction.id, {"user_id": transaction.user_id})
    
    async def _extract_base_metadata(self, transaction: RawTransaction) -> Dict[str, Any]:
        """
        Extrait les métadonnées de base de la transaction.
        
        Args:
            transaction: Transaction à analyser
            
        Returns:
            Dict: Métadonnées de base
        """
        metadata = {
            "user_id": transaction.user_id,
            "raw_transaction_id": transaction.id,
            "bridge_transaction_id": transaction.bridge_transaction_id,
            "amount": transaction.amount,
            "currency": transaction.currency_code or "EUR",
            "date": transaction.date.isoformat() if transaction.date else None,
            "operation_type": transaction.operation_type
        }
        
        # Métadonnées temporelles
        if transaction.date:
            metadata.update({
                "day_of_week": transaction.date.weekday(),
                "week_of_year": transaction.date.isocalendar()[1],
                "month": transaction.date.month,
                "year": transaction.date.year,
                "is_weekend": transaction.date.weekday() >= 5,
                "hour": transaction.date.hour if hasattr(transaction.date, 'hour') else None
            })
        
        # Informations du compte
        account = self.db.query(SyncAccount).filter(
            SyncAccount.id == transaction.account_id
        ).first()
        
        if account:
            metadata.update({
                "account_id": account.bridge_account_id,
                "account_name": account.account_name,
                "account_type": account.account_type
            })
        
        # Informations de catégorie
        if transaction.category_id and transaction.category_id in self._categories_cache:
            category = self._categories_cache[transaction.category_id]
            metadata.update({
                "category_id": category.bridge_category_id,
                "category_name": category.name,
                "parent_category_id": category.parent_id,
                "parent_category_name": category.parent_name
            })
        
        return metadata
    
    async def _normalize_description(self, transaction: RawTransaction) -> Dict[str, Any]:
        """
        Normalise et nettoie la description de la transaction.
        
        Args:
            transaction: Transaction à analyser
            
        Returns:
            Dict: Description normalisée
        """
        raw_description = transaction.clean_description or transaction.provider_description or ""
        
        # Nettoyer la description
        cleaned_description = self._clean_description_text(raw_description)
        
        # Extraire des informations structurées
        extracted_info = self._extract_description_info(cleaned_description)
        
        return {
            "description": raw_description,
            "cleaned_description": cleaned_description,
            **extracted_info
        }
    
    def _clean_description_text(self, description: str) -> str:
        """
        Nettoie le texte de description.
        
        Args:
            description: Description brute
            
        Returns:
            str: Description nettoyée
        """
        if not description:
            return ""
        
        # Supprimer les caractères spéciaux et normaliser
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', description)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().title()
        
        # Supprimer les codes bancaires communs
        patterns_to_remove = [
            r'\b\d{8,}\b',  # Codes longs
            r'\bCB\s*\d+\b',  # Codes CB
            r'\b\d{2}/\d{2}\b',  # Dates format MM/DD
            r'\b\d{4}\*+\d+\b'  # Numéros de carte masqués
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    def _extract_description_info(self, description: str) -> Dict[str, Any]:
        """
        Extrait des informations structurées de la description.
        
        Args:
            description: Description nettoyée
            
        Returns:
            Dict: Informations extraites
        """
        info = {
            "contains_card_payment": False,
            "contains_transfer": False,
            "contains_subscription_indicators": False,
            "contains_location": False,
            "extracted_keywords": []
        }
        
        description_lower = description.lower()
        
        # Détecter les types de paiement
        card_indicators = ['cb', 'carte', 'card', 'paiement']
        if any(indicator in description_lower for indicator in card_indicators):
            info["contains_card_payment"] = True
        
        transfer_indicators = ['virement', 'transfer', 'vir.', 'sepa']
        if any(indicator in description_lower for indicator in transfer_indicators):
            info["contains_transfer"] = True
        
        subscription_indicators = ['abonnement', 'subscription', 'mensuel', 'monthly', 'recurring']
        if any(indicator in description_lower for indicator in subscription_indicators):
            info["contains_subscription_indicators"] = True
        
        # Extraire des mots-clés significatifs
        words = description.split()
        keywords = [word for word in words if len(word) > 3 and word.isalpha()]
        info["extracted_keywords"] = keywords[:5]  # Limiter à 5 mots-clés
        
        return info
    
    async def _classify_transaction(self, transaction: RawTransaction, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifie la transaction selon différents critères.
        
        Args:
            transaction: Transaction à classifier
            enriched_data: Données déjà enrichies
            
        Returns:
            Dict: Classifications
        """
        classification = {
            "is_expense": transaction.amount < 0,
            "is_income": transaction.amount > 0,
            "is_transfer": False,
            "is_recurring": False,
            "is_subscription": False,
            "is_bill_payment": False,
            "is_cash_withdrawal": False,
            "confidence_score": 0.0
        }
        
        amount = abs(transaction.amount)
        description = enriched_data.get("cleaned_description", "").lower()
        
        # Classification basée sur la description
        if any(keyword in description for keyword in ['virement', 'transfer', 'vir.']):
            classification["is_transfer"] = True
            classification["confidence_score"] += 0.3
        
        if any(keyword in description for keyword in ['retrait', 'dab', 'atm', 'distributeur']):
            classification["is_cash_withdrawal"] = True
            classification["confidence_score"] += 0.4
        
        if any(keyword in description for keyword in ['abonnement', 'subscription', 'netflix', 'spotify']):
            classification["is_subscription"] = True
            classification["confidence_score"] += 0.3
        
        # Classification basée sur le montant
        if amount > 1000:
            classification["is_large_transaction"] = True
        elif amount < 5:
            classification["is_small_transaction"] = True
        
        # Classification basée sur la fréquence (nécessiterait une analyse historique)
        # TODO: Implémenter la détection de récurrence basée sur l'historique
        
        return classification
    
    async def _generate_enriched_description(self, transaction: RawTransaction, enriched_data: Dict[str, Any]) -> str:
        """
        Génère une description enrichie en langage naturel.
        
        Args:
            transaction: Transaction originale
            enriched_data: Données enrichies
            
        Returns:
            str: Description enrichie
        """
        amount = abs(transaction.amount)
        currency = enriched_data.get("currency", "EUR")
        date = enriched_data.get("date", "")
        account_name = enriched_data.get("account_name", "compte bancaire")
        
        # Base de la description
        if transaction.amount < 0:
            base = f"Dépense de {amount:.2f} {currency}"
        else:
            base = f"Crédit de {amount:.2f} {currency}"
        
        # Ajouter le contexte
        context_parts = [base]
        
        if enriched_data.get("merchant_name"):
            context_parts.append(f"chez {enriched_data['merchant_name']}")
        elif enriched_data.get("cleaned_description"):
            context_parts.append(f"pour {enriched_data['cleaned_description']}")
        
        if enriched_data.get("category_name"):
            context_parts.append(f"dans la catégorie {enriched_data['category_name']}")
        
        context_parts.append(f"sur le compte {account_name}")
        
        if date:
            try:
                dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
                context_parts.append(f"le {dt.strftime('%d/%m/%Y')}")
            except:
                pass
        
        # Ajouter des qualificatifs
        qualifiers = []
        if enriched_data.get("is_recurring"):
            qualifiers.append("récurrente")
        if enriched_data.get("is_subscription"):
            qualifiers.append("abonnement")
        if enriched_data.get("is_transfer"):
            qualifiers.append("virement")
        
        if qualifiers:
            context_parts.insert(1, f"({', '.join(qualifiers)})")
        
        return " ".join(context_parts)
    
    @log_performance
    async def _generate_embedding(self, enriched_description: str, enriched_data: Dict[str, Any]) -> List[float]:
        """
        Génère l'embedding vectoriel pour la transaction.
        
        Args:
            enriched_description: Description enrichie
            enriched_data: Données enrichies
            
        Returns:
            List[float]: Vecteur d'embedding
        """
        # Construire le texte complet pour l'embedding
        embedding_text_parts = [enriched_description]
        
        # Ajouter des informations contextuelles
        if enriched_data.get("category_name"):
            embedding_text_parts.append(f"Catégorie: {enriched_data['category_name']}")
        
        if enriched_data.get("merchant_name"):
            embedding_text_parts.append(f"Marchand: {enriched_data['merchant_name']}")
        
        # Ajouter des mots-clés
        keywords = enriched_data.get("extracted_keywords", [])
        if keywords:
            embedding_text_parts.append(f"Mots-clés: {' '.join(keywords)}")
        
        # Ajouter des caractéristiques financières
        amount = enriched_data.get("amount", 0)
        if abs(amount) > 100:
            embedding_text_parts.append("transaction importante")
        elif abs(amount) < 10:
            embedding_text_parts.append("petite transaction")
        
        full_text = " | ".join(embedding_text_parts)
        
        try:
            vector = await self.embedding_service.generate_embedding(full_text)
            return vector
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding: {e}")
            # Retourner un vecteur par défaut ou re-lever l'exception
            raise
    
    async def _build_qdrant_payload(self, transaction: RawTransaction, enriched_data: Dict[str, Any], point_id: str) -> Dict[str, Any]:
        """
        Construit le payload pour Qdrant.
        
        Args:
            transaction: Transaction originale
            enriched_data: Données enrichies
            point_id: ID du point Qdrant
            
        Returns:
            Dict: Payload pour Qdrant
        """
        return {
            "id": point_id,
            "user_id": transaction.user_id,
            "raw_transaction_id": transaction.id,
            "bridge_transaction_id": transaction.bridge_transaction_id,
            "account_id": enriched_data.get("account_id"),
            "account_name": enriched_data.get("account_name"),
            "account_type": enriched_data.get("account_type"),
            
            # Données principales
            "description": enriched_data.get("description"),
            "enriched_description": enriched_data.get("enriched_description"),
            "amount": enriched_data.get("amount"),
            "currency": enriched_data.get("currency"),
            "date": enriched_data.get("date"),
            "day_of_week": enriched_data.get("day_of_week"),
            "week_of_year": enriched_data.get("week_of_year"),
            "month": enriched_data.get("month"),
            "year": enriched_data.get("year"),
            
            # Catégorisation
            "category_id": enriched_data.get("category_id"),
            "category_name": enriched_data.get("category_name"),
            "subcategory_id": enriched_data.get("parent_category_id"),
            "subcategory_name": enriched_data.get("parent_category_name"),
            
            # Marchands
            "merchant_id": enriched_data.get("merchant_id"),
            "merchant_name": enriched_data.get("merchant_name"),
            "merchant_type": enriched_data.get("merchant_type"),
            
            # Classifications
            "is_expense": enriched_data.get("is_expense", False),
            "is_income": enriched_data.get("is_income", False),
            "is_transfer": enriched_data.get("is_transfer", False),
            "is_recurring": enriched_data.get("is_recurring", False),
            "is_subscription": enriched_data.get("is_subscription", False),
            "is_bill_payment": enriched_data.get("is_bill_payment", False),
            
            # Contexte
            "tags": enriched_data.get("extracted_keywords", []),
            "context_summary": enriched_data.get("enriched_description")
        }
    
    @log_performance
    async def enrich_transactions_batch(self, transactions: List[RawTransaction], batch_size: int = 50) -> Dict[str, Any]:
        """
        Enrichit un lot de transactions.
        
        Args:
            transactions: Liste de transactions à enrichir
            batch_size: Taille des sous-lots pour le traitement
            
        Returns:
            Dict: Résultat de l'enrichissement en lot
        """
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="batch_transaction",
            batch_id=str(uuid4())[:8]
        )
        
        ctx_logger.info(f"Début de l'enrichissement de {len(transactions)} transactions")
        
        result = {
            "status": "pending",
            "total_transactions": len(transactions),
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Traiter par sous-lots
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            ctx_logger.info(f"Traitement du sous-lot {i//batch_size + 1}: {len(batch)} transactions")
            
            for transaction in batch:
                try:
                    await self.enrich_transaction(transaction)
                    result["processed"] += 1
                except Exception as e:
                    result["failed"] += 1
                    error_info = {
                        "transaction_id": transaction.id,
                        "error": str(e)
                    }
                    result["errors"].append(error_info)
                    ctx_logger.warning(f"Échec de l'enrichissement de la transaction {transaction.id}: {e}")
        
        # Déterminer le statut final
        if result["failed"] == 0:
            result["status"] = "success"
        elif result["processed"] > 0:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
        
        ctx_logger.info(f"Enrichissement terminé: {result['processed']}/{result['total_transactions']} succès, {result['failed']} échecs")
        
        return result
    
    async def update_existing_transaction(self, transaction: RawTransaction) -> Dict[str, Any]:
        """
        Met à jour une transaction déjà enrichie.
        
        Args:
            transaction: Transaction mise à jour
            
        Returns:
            Dict: Résultat de la mise à jour
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="transaction_update"
        )
        
        try:
            # Rechercher le point existant dans Qdrant
            existing_points = await self.qdrant_service.search_points(
                collection_name="enriched_transactions",
                filter_conditions={"bridge_transaction_id": transaction.bridge_transaction_id}
            )
            
            if existing_points:
                # Mettre à jour le point existant
                point_id = existing_points[0]["id"]
                ctx_logger.info(f"Mise à jour du point existant: {point_id}")
                return await self.enrich_transaction(transaction)
            else:
                # Créer un nouveau point
                ctx_logger.info("Aucun point existant trouvé, création d'un nouveau point")
                return await self.enrich_transaction(transaction)
                
        except Exception as e:
            error_msg = f"Erreur lors de la mise à jour de la transaction {transaction.id}: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            raise TransactionEnrichmentError(error_msg, transaction.id)
    
    async def delete_transaction_vector(self, bridge_transaction_id: int, user_id: int) -> bool:
        """
        Supprime le vecteur d'une transaction de Qdrant.
        
        Args:
            bridge_transaction_id: ID Bridge de la transaction
            user_id: ID de l'utilisateur
            
        Returns:
            bool: True si supprimé avec succès
        """
        try:
            return await self.qdrant_service.delete_points(
                collection_name="enriched_transactions",
                filter_conditions={
                    "bridge_transaction_id": bridge_transaction_id,
                    "user_id": user_id
                }
            )
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du vecteur de transaction {bridge_transaction_id}: {e}")
            return False