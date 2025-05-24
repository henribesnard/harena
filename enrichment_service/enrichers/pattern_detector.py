"""
Détecteur de patterns financiers automatique.

Ce module analyse les transactions pour détecter automatiquement des patterns
récurrents comme les abonnements, salaires, factures, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func
from uuid import uuid4

from enrichment_service.core.logging import get_contextual_logger, log_performance
from enrichment_service.core.exceptions import PatternDetectionError
from enrichment_service.core.config import enrichment_settings
from enrichment_service.db.models import RawTransaction

logger = logging.getLogger(__name__)

@dataclass
class TransactionPattern:
    """Représente un pattern de transaction détecté."""
    pattern_id: str
    user_id: int
    pattern_type: str
    pattern_name: str
    merchant_name: Optional[str] = None
    amount_range: Tuple[float, float] = (0.0, 0.0)
    typical_amount: float = 0.0
    frequency_days: float = 0.0
    confidence: float = 0.0
    regularity_score: float = 0.0
    transaction_count: int = 0
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    next_predicted: Optional[datetime] = None
    transactions: List[int] = field(default_factory=list)
    
@dataclass
class PatternCandidate:
    """Candidat pour un pattern en cours d'analyse."""
    transactions: List[RawTransaction]
    merchant_group: str
    amount_group: float
    intervals: List[float]
    avg_interval: float
    std_interval: float
    confidence: float

class PatternDetector:
    """
    Détecteur automatique de patterns dans les transactions financières.
    
    Cette classe analyse les transactions pour identifier automatiquement
    des patterns récurrents sans patterns prédéfinis.
    """
    
    def __init__(self, db: Session, embedding_service, qdrant_service):
        """
        Initialise le détecteur de patterns.
        
        Args:
            db: Session de base de données
            embedding_service: Service d'embedding
            qdrant_service: Service Qdrant
        """
        self.db = db
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        
        # Paramètres de détection
        self.min_occurrences = enrichment_settings.pattern_minimum_occurrences
        self.confidence_threshold = enrichment_settings.pattern_confidence_threshold
        self.window_days = enrichment_settings.pattern_detection_window_days
        
        # Seuils pour la détection automatique
        self.amount_tolerance = 0.15  # 15% de tolérance sur les montants
        self.interval_tolerance = 0.25  # 25% de tolérance sur les intervalles
        self.min_regularity_score = 0.6  # Score minimum de régularité
    
    @log_performance
    async def detect_patterns_for_user(self, user_id: int) -> Dict[str, Any]:
        """
        Détecte tous les patterns pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Dict: Résultat de la détection
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=user_id,
            enrichment_type="pattern_detection"
        )
        
        ctx_logger.info(f"Début de la détection de patterns pour l'utilisateur {user_id}")
        
        result = {
            "user_id": user_id,
            "status": "pending",
            "patterns_detected": 0,
            "patterns_updated": 0,
            "candidates_analyzed": 0,
            "patterns": []
        }
        
        try:
            # 1. Récupérer les transactions récentes
            transactions = await self._get_user_transactions(user_id)
            
            if len(transactions) < self.min_occurrences:
                ctx_logger.info(f"Pas assez de transactions ({len(transactions)}) pour détecter des patterns")
                result["status"] = "insufficient_data"
                return result
            
            # 2. Grouper les transactions par similarité
            candidate_groups = await self._group_similar_transactions(transactions)
            result["candidates_analyzed"] = len(candidate_groups)
            
            # 3. Analyser chaque groupe pour détecter des patterns
            detected_patterns = []
            for group in candidate_groups:
                pattern = await self._analyze_transaction_group(user_id, group)
                if pattern and pattern.confidence >= self.confidence_threshold:
                    detected_patterns.append(pattern)
            
            # 4. Filtrer et affiner les patterns
            filtered_patterns = await self._filter_and_merge_patterns(detected_patterns)
            
            # 5. Générer les embeddings et stocker dans Qdrant
            for pattern in filtered_patterns:
                await self._store_pattern_in_qdrant(pattern)
                result["patterns"].append(self._pattern_to_dict(pattern))
            
            result["patterns_detected"] = len(filtered_patterns)
            result["status"] = "success"
            
            ctx_logger.info(f"Détection terminée: {len(filtered_patterns)} patterns détectés sur {len(candidate_groups)} candidats")
            
        except Exception as e:
            error_msg = f"Erreur lors de la détection de patterns: {str(e)}"
            ctx_logger.error(error_msg, exc_info=True)
            result["status"] = "error"
            result["error"] = error_msg
        
        return result
    
    async def _get_user_transactions(self, user_id: int, days: Optional[int] = None) -> List[RawTransaction]:
        """
        Récupère les transactions d'un utilisateur pour l'analyse.
        
        Args:
            user_id: ID de l'utilisateur
            days: Nombre de jours à récupérer (par défaut: window_days)
            
        Returns:
            List[RawTransaction]: Liste des transactions
        """
        days = days or self.window_days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        transactions = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == user_id,
            RawTransaction.date >= cutoff_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.amount != 0  # Exclure les transactions nulles
        ).order_by(RawTransaction.date.asc()).all()
        
        return transactions
    
    async def _group_similar_transactions(self, transactions: List[RawTransaction]) -> List[List[RawTransaction]]:
        """
        Groupe les transactions similaires par description et montant.
        
        Args:
            transactions: Liste des transactions à grouper
            
        Returns:
            List[List[RawTransaction]]: Groupes de transactions similaires
        """
        # Grouper par similarité de description et de montant
        groups = defaultdict(list)
        
        for transaction in transactions:
            # Normaliser la description pour le groupement
            description_key = self._normalize_description_for_grouping(
                transaction.clean_description or transaction.provider_description or ""
            )
            
            # Arrondir le montant pour créer des groupes de montants similaires
            amount_group = self._get_amount_group(abs(transaction.amount))
            
            # Créer une clé composite
            group_key = (description_key, amount_group, transaction.amount > 0)
            groups[group_key].append(transaction)
        
        # Filtrer les groupes avec suffisamment de transactions
        filtered_groups = [
            group for group in groups.values() 
            if len(group) >= self.min_occurrences
        ]
        
        return filtered_groups
    
    def _normalize_description_for_grouping(self, description: str) -> str:
        """
        Normalise une description pour le groupement.
        
        Args:
            description: Description à normaliser
            
        Returns:
            str: Description normalisée
        """
        if not description:
            return "unknown"
        
        # Supprimer les éléments variables
        import re
        
        normalized = description.lower()
        
        # Supprimer les dates, numéros, références
        patterns_to_remove = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b\d{8,}\b',  # Numéros longs
            r'\bref\s*:?\s*\w+\b',  # Références
            r'\b\d{4}\*+\d+\b',  # Numéros masqués
        ]
        
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized)
        
        # Nettoyer et normaliser
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Prendre les premiers mots significatifs
        words = [w for w in normalized.split() if len(w) > 2]
        return ' '.join(words[:3])  # Limiter à 3 mots
    
    def _get_amount_group(self, amount: float) -> float:
        """
        Détermine le groupe de montant pour une transaction.
        
        Args:
            amount: Montant de la transaction
            
        Returns:
            float: Groupe de montant
        """
        if amount < 10:
            return round(amount, 1)  # Précision au dixième pour les petits montants
        elif amount < 100:
            return round(amount / 5) * 5  # Groupes de 5€
        elif amount < 1000:
            return round(amount / 10) * 10  # Groupes de 10€
        else:
            return round(amount / 50) * 50  # Groupes de 50€
    
    async def _analyze_transaction_group(self, user_id: int, transactions: List[RawTransaction]) -> Optional[TransactionPattern]:
        """
        Analyse un groupe de transactions pour détecter un pattern.
        
        Args:
            user_id: ID de l'utilisateur
            transactions: Groupe de transactions à analyser
            
        Returns:
            Optional[TransactionPattern]: Pattern détecté ou None
        """
        if len(transactions) < self.min_occurrences:
            return None
        
        # Trier par date
        transactions.sort(key=lambda t: t.date)
        
        # Calculer les intervalles entre transactions
        intervals = []
        for i in range(1, len(transactions)):
            interval = (transactions[i].date - transactions[i-1].date).days
            intervals.append(interval)
        
        if not intervals:
            return None
        
        # Analyser la régularité
        avg_interval = sum(intervals) / len(intervals)
        
        # Calculer l'écart-type des intervalles
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_interval = variance ** 0.5
        
        # Score de régularité (plus l'écart-type est faible, plus c'est régulier)
        regularity_score = max(0, 1 - (std_interval / avg_interval)) if avg_interval > 0 else 0
        
        # Vérifier si c'est suffisamment régulier
        if regularity_score < self.min_regularity_score:
            return None
        
        # Analyser les montants
        amounts = [abs(t.amount) for t in transactions]
        avg_amount = sum(amounts) / len(amounts)
        amount_std = (sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)) ** 0.5
        
        # Score de consistance des montants
        amount_consistency = max(0, 1 - (amount_std / avg_amount)) if avg_amount > 0 else 0
        
        # Score de confiance global
        confidence = (regularity_score + amount_consistency) / 2
        
        if confidence < self.confidence_threshold:
            return None
        
        # Déterminer le type de pattern automatiquement
        pattern_type = self._determine_pattern_type(transactions, avg_interval, avg_amount)
        
        # Prédire la prochaine occurrence
        next_predicted = transactions[-1].date + timedelta(days=avg_interval)
        
        # Créer le pattern
        pattern = TransactionPattern(
            pattern_id=str(uuid4()),
            user_id=user_id,
            pattern_type=pattern_type,
            pattern_name=self._generate_pattern_name(transactions, pattern_type),
            merchant_name=self._extract_merchant_name(transactions),
            amount_range=(min(amounts), max(amounts)),
            typical_amount=avg_amount,
            frequency_days=avg_interval,
            confidence=confidence,
            regularity_score=regularity_score,
            transaction_count=len(transactions),
            first_occurrence=transactions[0].date,
            last_occurrence=transactions[-1].date,
            next_predicted=next_predicted,
            transactions=[t.id for t in transactions]
        )
        
        return pattern
    
    def _determine_pattern_type(self, transactions: List[RawTransaction], avg_interval: float, avg_amount: float) -> str:
        """
        Détermine automatiquement le type de pattern.
        
        Args:
            transactions: Liste des transactions
            avg_interval: Intervalle moyen entre transactions
            avg_amount: Montant moyen
            
        Returns:
            str: Type de pattern déterminé
        """
        # Analyser les caractéristiques pour déterminer le type
        
        # Revenus (montants positifs, intervalles réguliers)
        if all(t.amount > 0 for t in transactions):
            if 25 <= avg_interval <= 35:  # Environ mensuel
                if avg_amount > 800:
                    return "salary"
                else:
                    return "recurring_income"
            elif 6 <= avg_interval <= 8:  # Environ hebdomadaire
                return "weekly_income"
            else:
                return "irregular_income"
        
        # Dépenses (montants négatifs)
        else:
            # Analyser la description pour plus de contexte
            descriptions = [
                (t.clean_description or t.provider_description or "").lower() 
                for t in transactions
            ]
            combined_description = " ".join(descriptions)
            
            # Patterns basés sur les mots-clés dans les descriptions
            if any(keyword in combined_description for keyword in ['netflix', 'spotify', 'amazon prime', 'subscription']):
                return "subscription"
            elif any(keyword in combined_description for keyword in ['edf', 'gdf', 'orange', 'sfr', 'bouygues', 'free']):
                return "utility_bill"
            elif any(keyword in combined_description for keyword in ['assurance', 'mutuelle']):
                return "insurance"
            elif any(keyword in combined_description for keyword in ['loyer', 'rent']):
                return "rent"
            elif any(keyword in combined_description for keyword in ['credit', 'pret', 'loan']):
                return "loan_payment"
            elif 25 <= avg_interval <= 35:  # Mensuel
                if avg_amount > 500:
                    return "large_monthly_expense"
                else:
                    return "monthly_bill"
            elif 6 <= avg_interval <= 8:  # Hebdomadaire
                return "weekly_expense"
            elif 80 <= avg_interval <= 100:  # Trimestriel
                return "quarterly_payment"
            else:
                return "recurring_expense"
    
    def _generate_pattern_name(self, transactions: List[RawTransaction], pattern_type: str) -> str:
        """
        Génère un nom descriptif pour le pattern.
        
        Args:
            transactions: Liste des transactions
            pattern_type: Type de pattern
            
        Returns:
            str: Nom du pattern
        """
        # Extraire le nom le plus commun
        descriptions = [
            (t.clean_description or t.provider_description or "").strip()
            for t in transactions if t.clean_description or t.provider_description
        ]
        
        if descriptions:
            # Trouver la description la plus fréquente
            from collections import Counter
            most_common = Counter(descriptions).most_common(1)[0][0]
            
            # Nettoyer et raccourcir
            cleaned_name = self._clean_pattern_name(most_common)
            if cleaned_name:
                return cleaned_name
        
        # Nom par défaut basé sur le type
        type_names = {
            "salary": "Salaire",
            "recurring_income": "Revenu récurrent",
            "subscription": "Abonnement",
            "utility_bill": "Facture",
            "insurance": "Assurance",
            "rent": "Loyer",
            "loan_payment": "Remboursement crédit",
            "monthly_bill": "Facture mensuelle",
            "weekly_expense": "Dépense hebdomadaire",
            "quarterly_payment": "Paiement trimestriel",
            "recurring_expense": "Dépense récurrente"
        }
        
        return type_names.get(pattern_type, "Pattern récurrent")
    
    def _clean_pattern_name(self, name: str) -> str:
        """
        Nettoie un nom de pattern.
        
        Args:
            name: Nom brut
            
        Returns:
            str: Nom nettoyé
        """
        if not name:
            return ""
        
        import re
        
        # Supprimer les éléments variables
        cleaned = re.sub(r'\b\d{8,}\b', '', name)  # Numéros longs
        cleaned = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', cleaned)  # Dates
        cleaned = re.sub(r'\bref\s*:?\s*\w+\b', '', cleaned, flags=re.IGNORECASE)  # Références
        
        # Nettoyer les caractères
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Limiter la longueur
        if len(cleaned) > 30:
            words = cleaned.split()
            cleaned = ' '.join(words[:4])
        
        return cleaned.title() if cleaned else ""
    
    def _extract_merchant_name(self, transactions: List[RawTransaction]) -> Optional[str]:
        """
        Extrait le nom de marchand le plus probable.
        
        Args:
            transactions: Liste des transactions
            
        Returns:
            Optional[str]: Nom du marchand
        """
        # Utiliser la description la plus fréquente comme nom de marchand
        descriptions = []
        for t in transactions:
            desc = t.clean_description or t.provider_description
            if desc:
                cleaned = self._clean_pattern_name(desc)
                if cleaned:
                    descriptions.append(cleaned)
        
        if descriptions:
            from collections import Counter
            most_common = Counter(descriptions).most_common(1)[0][0]
            return most_common
        
        return None
    
    async def _filter_and_merge_patterns(self, patterns: List[TransactionPattern]) -> List[TransactionPattern]:
        """
        Filtre et fusionne les patterns similaires.
        
        Args:
            patterns: Liste des patterns détectés
            
        Returns:
            List[TransactionPattern]: Patterns filtrés et fusionnés
        """
        if not patterns:
            return []
        
        # Trier par confiance décroissante
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        filtered_patterns = []
        
        for pattern in patterns:
            # Vérifier s'il existe déjà un pattern similaire
            similar_pattern = None
            for existing in filtered_patterns:
                if self._are_patterns_similar(pattern, existing):
                    similar_pattern = existing
                    break
            
            if similar_pattern:
                # Fusionner avec le pattern existant
                await self._merge_patterns(similar_pattern, pattern)
            else:
                # Ajouter comme nouveau pattern
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _are_patterns_similar(self, pattern1: TransactionPattern, pattern2: TransactionPattern) -> bool:
        """
        Vérifie si deux patterns sont similaires.
        
        Args:
            pattern1: Premier pattern
            pattern2: Deuxième pattern
            
        Returns:
            bool: True si les patterns sont similaires
        """
        # Vérifier la similarité du nom/marchand
        if pattern1.merchant_name and pattern2.merchant_name:
            from difflib import SequenceMatcher
            name_similarity = SequenceMatcher(None, pattern1.merchant_name.lower(), pattern2.merchant_name.lower()).ratio()
            if name_similarity < 0.8:
                return False
        
        # Vérifier la similarité des montants
        amount_diff = abs(pattern1.typical_amount - pattern2.typical_amount)
        amount_tolerance = max(pattern1.typical_amount, pattern2.typical_amount) * self.amount_tolerance
        if amount_diff > amount_tolerance:
            return False
        
        # Vérifier la similarité des intervalles
        interval_diff = abs(pattern1.frequency_days - pattern2.frequency_days)
        interval_tolerance = max(pattern1.frequency_days, pattern2.frequency_days) * self.interval_tolerance
        if interval_diff > interval_tolerance:
            return False
        
        return True
    
    async def _merge_patterns(self, existing_pattern: TransactionPattern, new_pattern: TransactionPattern):
        """
        Fusionne deux patterns similaires.
        
        Args:
            existing_pattern: Pattern existant (sera modifié)
            new_pattern: Nouveau pattern à fusionner
        """
        # Combiner les transactions
        all_transactions = existing_pattern.transactions + new_pattern.transactions
        existing_pattern.transactions = list(set(all_transactions))  # Supprimer les doublons
        existing_pattern.transaction_count = len(existing_pattern.transactions)
        
        # Recalculer les métriques avec toutes les transactions
        total_amount = (existing_pattern.typical_amount * len(existing_pattern.transactions) + 
                       new_pattern.typical_amount * len(new_pattern.transactions))
        total_transactions = len(existing_pattern.transactions) + len(new_pattern.transactions)
        existing_pattern.typical_amount = total_amount / total_transactions
        
        # Prendre les meilleures valeurs
        existing_pattern.confidence = max(existing_pattern.confidence, new_pattern.confidence)
        existing_pattern.regularity_score = max(existing_pattern.regularity_score, new_pattern.regularity_score)
        
        # Mettre à jour les dates
        if new_pattern.first_occurrence < existing_pattern.first_occurrence:
            existing_pattern.first_occurrence = new_pattern.first_occurrence
        if new_pattern.last_occurrence > existing_pattern.last_occurrence:
            existing_pattern.last_occurrence = new_pattern.last_occurrence
            existing_pattern.next_predicted = new_pattern.next_predicted
        
        # Étendre la plage de montants
        existing_pattern.amount_range = (
            min(existing_pattern.amount_range[0], new_pattern.amount_range[0]),
            max(existing_pattern.amount_range[1], new_pattern.amount_range[1])
        )
    
    async def _store_pattern_in_qdrant(self, pattern: TransactionPattern):
        """
        Stocke un pattern dans Qdrant.
        
        Args:
            pattern: Pattern à stocker
        """
        # Générer la description du pattern
        pattern_description = self._generate_pattern_description(pattern)
        
        # Générer l'embedding
        vector = await self.embedding_service.generate_embedding(pattern_description)
        
        # Construire le payload
        payload = {
            "id": pattern.pattern_id,
            "user_id": pattern.user_id,
            "pattern_type": pattern.pattern_type,
            "pattern_name": pattern.pattern_name,
            "pattern_description": pattern_description,
            
            # Caractéristiques financières
            "amount": pattern.typical_amount,
            "amount_range": list(pattern.amount_range),
            "frequency": self._frequency_to_text(pattern.frequency_days),
            "regularity_score": pattern.regularity_score,
            
            # Attributs
            "merchant_id": pattern.merchant_name,  # Utiliser le nom comme ID pour l'instant
            "merchant_name": pattern.merchant_name,
            "category_id": None,  # À enrichir si nécessaire
            "category_name": None,
            
            # Classification
            "is_income": pattern.typical_amount > 0,
            "is_expense": pattern.typical_amount < 0,
            "is_subscription": pattern.pattern_type == "subscription",
            "is_bill": pattern.pattern_type in ["utility_bill", "monthly_bill"],
            
            # Chronologie
            "last_occurrence_date": pattern.last_occurrence.isoformat() if pattern.last_occurrence else None,
            "next_expected_date": pattern.next_predicted.isoformat() if pattern.next_predicted else None,
            "expected_dates": [pattern.next_predicted.isoformat()] if pattern.next_predicted else [],
            
            # Importance
            "importance_score": self._calculate_importance_score(pattern),
            "monthly_impact": self._calculate_monthly_impact(pattern),
            "yearly_impact": self._calculate_yearly_impact(pattern),
            "budget_percentage": 0.0,  # À calculer avec le budget total
            
            # Contexte
            "recent_transactions": pattern.transactions[-5:],  # 5 dernières transactions
            "status": "active",
            "trend": "stable",  # À analyser
            "narrative": pattern_description,
            "tags": self._generate_pattern_tags(pattern)
        }
        
        # Stocker dans Qdrant
        await self.qdrant_service.upsert_point(
            collection_name="financial_patterns",
            point_id=pattern.pattern_id,
            vector=vector,
            payload=payload
        )
    
    def _generate_pattern_description(self, pattern: TransactionPattern) -> str:
        """
        Génère une description textuelle du pattern.
        
        Args:
            pattern: Pattern à décrire
            
        Returns:
            str: Description du pattern
        """
        frequency_text = self._frequency_to_text(pattern.frequency_days)
        amount_text = f"{pattern.typical_amount:.2f}€"
        
        if pattern.typical_amount > 0:
            direction = "revenus"
        else:
            direction = "dépenses"
            amount_text = f"{abs(pattern.typical_amount):.2f}€"
        
        base_description = f"{direction} {frequency_text} de {amount_text}"
        
        if pattern.merchant_name:
            base_description += f" pour {pattern.merchant_name}"
        
        base_description += f" (confiance: {pattern.confidence:.0%})"
        
        # Ajouter des détails sur la régularité
        if pattern.regularity_score > 0.9:
            base_description += " - Très régulier"
        elif pattern.regularity_score > 0.7:
            base_description += " - Assez régulier"
        
        return base_description
    
    def _frequency_to_text(self, frequency_days: float) -> str:
        """
        Convertit une fréquence en jours en texte.
        
        Args:
            frequency_days: Fréquence en jours
            
        Returns:
            str: Description textuelle de la fréquence
        """
        if frequency_days <= 7:
            return "hebdomadaires"
        elif frequency_days <= 15:
            return "bi-hebdomadaires"
        elif frequency_days <= 35:
            return "mensuels"
        elif frequency_days <= 70:
            return "bi-mensuels"
        elif frequency_days <= 100:
            return "trimestriels"
        elif frequency_days <= 200:
            return "semestriels"
        else:
            return "annuels"
    
    def _calculate_importance_score(self, pattern: TransactionPattern) -> float:
        """
        Calcule un score d'importance pour le pattern.
        
        Args:
            pattern: Pattern à évaluer
            
        Returns:
            float: Score d'importance (0-1)
        """
        # Facteurs d'importance
        amount_factor = min(abs(pattern.typical_amount) / 1000, 1.0)  # Normaliser sur 1000€
        regularity_factor = pattern.regularity_score
        frequency_factor = 1.0 / (pattern.frequency_days / 30)  # Plus fréquent = plus important
        frequency_factor = min(frequency_factor, 1.0)
        
        # Score pondéré
        importance = (amount_factor * 0.4 + regularity_factor * 0.4 + frequency_factor * 0.2)
        return min(importance, 1.0)
    
    def _calculate_monthly_impact(self, pattern: TransactionPattern) -> float:
        """
        Calcule l'impact mensuel du pattern.
        
        Args:
            pattern: Pattern à analyser
            
        Returns:
            float: Impact mensuel en euros
        """
        monthly_frequency = 30.0 / pattern.frequency_days
        return abs(pattern.typical_amount) * monthly_frequency
    
    def _calculate_yearly_impact(self, pattern: TransactionPattern) -> float:
        """
        Calcule l'impact annuel du pattern.
        
        Args:
            pattern: Pattern à analyser
            
        Returns:
            float: Impact annuel en euros
        """
        return self._calculate_monthly_impact(pattern) * 12
    
    def _generate_pattern_tags(self, pattern: TransactionPattern) -> List[str]:
        """
        Génère des tags pour le pattern.
        
        Args:
            pattern: Pattern à analyser
            
        Returns:
            List[str]: Liste de tags
        """
        tags = [pattern.pattern_type]
        
        # Tags basés sur le montant
        if abs(pattern.typical_amount) > 1000:
            tags.append("montant_élevé")
        elif abs(pattern.typical_amount) < 50:
            tags.append("petit_montant")
        
        # Tags basés sur la fréquence
        if pattern.frequency_days <= 7:
            tags.append("fréquent")
        elif pattern.frequency_days >= 90:
            tags.append("rare")
        
        # Tags basés sur la régularité
        if pattern.regularity_score > 0.9:
            tags.append("très_régulier")
        elif pattern.regularity_score < 0.7:
            tags.append("irrégulier")
        
        # Tags basés sur le type
        if pattern.typical_amount > 0:
            tags.append("revenu")
        else:
            tags.append("dépense")
        
        return tags
    
    def _pattern_to_dict(self, pattern: TransactionPattern) -> Dict[str, Any]:
        """
        Convertit un pattern en dictionnaire.
        
        Args:
            pattern: Pattern à convertir
            
        Returns:
            Dict: Dictionnaire du pattern
        """
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "pattern_name": pattern.pattern_name,
            "merchant_name": pattern.merchant_name,
            "typical_amount": pattern.typical_amount,
            "amount_range": pattern.amount_range,
            "frequency_days": pattern.frequency_days,
            "confidence": pattern.confidence,
            "regularity_score": pattern.regularity_score,
            "transaction_count": pattern.transaction_count,
            "importance_score": self._calculate_importance_score(pattern),
            "monthly_impact": self._calculate_monthly_impact(pattern),
            "yearly_impact": self._calculate_yearly_impact(pattern),
            "first_occurrence": pattern.first_occurrence.isoformat() if pattern.first_occurrence else None,
            "last_occurrence": pattern.last_occurrence.isoformat() if pattern.last_occurrence else None,
            "next_predicted": pattern.next_predicted.isoformat() if pattern.next_predicted else None
        }
    
    async def update_patterns_for_transaction(self, transaction: RawTransaction):
        """
        Met à jour les patterns affectés par une nouvelle transaction.
        
        Args:
            transaction: Nouvelle transaction
        """
        ctx_logger = get_contextual_logger(
            __name__,
            user_id=transaction.user_id,
            transaction_id=transaction.id,
            enrichment_type="pattern_update"
        )
        
        try:
            # Rechercher les patterns existants qui pourraient être affectés
            existing_patterns = await self.qdrant_service.search_points(
                collection_name="financial_patterns",
                filter_conditions={"user_id": transaction.user_id},
                limit=100
            )
            
            # Vérifier si la transaction correspond à un pattern existant
            for pattern_point in existing_patterns:
                pattern_data = pattern_point["payload"]
                
                # Vérifier la similarité avec les critères du pattern
                if self._transaction_matches_pattern(transaction, pattern_data):
                    ctx_logger.debug(f"Transaction correspond au pattern {pattern_data['pattern_id']}")
                    await self._update_pattern_with_transaction(pattern_data, transaction)
            
            # Si aucun pattern ne correspond, relancer une détection limitée
            # pour voir si cette transaction peut créer un nouveau pattern
            await self._check_for_new_patterns_with_transaction(transaction)
            
        except Exception as e:
            ctx_logger.error(f"Erreur lors de la mise à jour des patterns: {e}", exc_info=True)
    
    def _transaction_matches_pattern(self, transaction: RawTransaction, pattern_data: Dict[str, Any]) -> bool:
        """
        Vérifie si une transaction correspond à un pattern existant.
        
        Args:
            transaction: Transaction à vérifier
            pattern_data: Données du pattern
            
        Returns:
            bool: True si la transaction correspond
        """
        # Vérifier le montant
        amount_tolerance = pattern_data["amount"] * self.amount_tolerance
        if abs(abs(transaction.amount) - abs(pattern_data["amount"])) > amount_tolerance:
            return False
        
        # Vérifier le nom du marchand si disponible
        if pattern_data.get("merchant_name"):
            transaction_desc = (transaction.clean_description or transaction.provider_description or "").lower()
            pattern_merchant = pattern_data["merchant_name"].lower()
            
            if pattern_merchant not in transaction_desc and transaction_desc not in pattern_merchant:
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, transaction_desc, pattern_merchant).ratio()
                if similarity < 0.7:
                    return False
        
        return True
    
    async def _update_pattern_with_transaction(self, pattern_data: Dict[str, Any], transaction: RawTransaction):
        """
        Met à jour un pattern avec une nouvelle transaction.
        
        Args:
            pattern_data: Données du pattern existant
            transaction: Nouvelle transaction
        """
        # Recalculer les métriques du pattern avec la nouvelle transaction
        # Pour l'instant, on recrée simplement le pattern
        # Une implémentation plus sophistiquée pourrait faire une mise à jour incrémentale
        pass
    
    async def _check_for_new_patterns_with_transaction(self, transaction: RawTransaction):
        """
        Vérifie si une nouvelle transaction peut créer de nouveaux patterns.
        
        Args:
            transaction: Nouvelle transaction
        """
        # Récupérer les transactions récentes similaires
        similar_transactions = await self._find_similar_recent_transactions(transaction)
        
        if len(similar_transactions) >= self.min_occurrences:
            # Analyser ce groupe pour voir s'il forme un pattern
            pattern = await self._analyze_transaction_group(transaction.user_id, similar_transactions)
            if pattern and pattern.confidence >= self.confidence_threshold:
                await self._store_pattern_in_qdrant(pattern)
    
    async def _find_similar_recent_transactions(self, transaction: RawTransaction, days: int = 180) -> List[RawTransaction]:
        """
        Trouve les transactions récentes similaires à une transaction donnée.
        
        Args:
            transaction: Transaction de référence
            days: Nombre de jours à analyser
            
        Returns:
            List[RawTransaction]: Transactions similaires
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Normaliser la description pour la recherche
        description_key = self._normalize_description_for_grouping(
            transaction.clean_description or transaction.provider_description or ""
        )
        amount_group = self._get_amount_group(abs(transaction.amount))
        
        # Rechercher des transactions similaires
        similar_transactions = self.db.query(RawTransaction).filter(
            RawTransaction.user_id == transaction.user_id,
            RawTransaction.date >= cutoff_date,
            RawTransaction.deleted.is_(False),
            RawTransaction.id != transaction.id  # Exclure la transaction elle-même
        ).all()
        
        # Filtrer par similarité
        filtered_transactions = [transaction]  # Inclure la transaction de référence
        
        for t in similar_transactions:
            t_description_key = self._normalize_description_for_grouping(
                t.clean_description or t.provider_description or ""
            )
            t_amount_group = self._get_amount_group(abs(t.amount))
            
            # Vérifier la similarité
            if (t_description_key == description_key and 
                t_amount_group == amount_group and 
                (t.amount > 0) == (transaction.amount > 0)):
                filtered_transactions.append(t)
        
        return filtered_transactions