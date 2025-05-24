"""
Normalisateur de noms de marchands.

Ce module nettoie et standardise les noms de marchands pour améliorer
la cohérence et permettre une meilleure détection de patterns.
"""

import logging
import re
from typing import Dict, Any, Optional, Set, List
from difflib import SequenceMatcher
from dataclasses import dataclass

from enrichment_service.core.logging import get_contextual_logger
from enrichment_service.core.config import enrichment_settings

logger = logging.getLogger(__name__)

@dataclass
class MerchantInfo:
    """Informations normalisées d'un marchand."""
    original_name: str
    normalized_name: str
    merchant_id: str
    merchant_type: Optional[str] = None
    confidence: float = 0.0
    aliases: List[str] = None

class MerchantNormalizer:
    """
    Normalisateur de noms de marchands avec apprentissage automatique.
    
    Cette classe nettoie les noms de marchands, détecte les variations
    et maintient une base de données normalisée des marchands.
    """
    
    def __init__(self, db_session=None):
        """
        Initialise le normalisateur de marchands.
        
        Args:
            db_session: Session de base de données (optionnelle)
        """
        self.db = db_session
        
        # Cache des marchands normalisés
        self._merchant_cache: Dict[str, MerchantInfo] = {}
        self._normalized_merchants: Dict[str, Set[str]] = {}  # normalized -> {originals}
        
        # Patterns de nettoyage
        self._cleanup_patterns = self._build_cleanup_patterns()
        
        # Types de marchands automatiquement détectés
        self._merchant_types = self._build_merchant_types()
        
        # Seuil de similarité pour la détection d'aliases
        self.similarity_threshold = enrichment_settings.merchant_similarity_threshold
    
    def _build_cleanup_patterns(self) -> List[tuple]:
        """
        Construit les patterns de nettoyage pour les noms de marchands.
        
        Returns:
            List[tuple]: Liste de (pattern, replacement)
        """
        return [
            # Supprimer les codes et références
            (r'\b\d{8,}\b', ''),  # Codes longs
            (r'\b\d{2}/\d{2}\b', ''),  # Dates
            (r'\bCB\s*\d+\b', ''),  # Codes CB
            (r'\b\d{4}\*+\d+\b', ''),  # Numéros masqués
            (r'\bRéf[:\s]*\w+\b', ''),  # Références
            
            # Nettoyer les préfixes/suffixes bancaires
            (r'\bPRLV\s+', ''),  # Prélèvement
            (r'\bVIR\s+', ''),  # Virement
            (r'\bCHQ\s+', ''),  # Chèque
            (r'\bCB\s+', ''),  # Carte bancaire
            (r'\bTPE\s+', ''),  # Terminal de paiement
            
            # Normaliser les formats d'adresse
            (r'\s+\d{5}\s+[A-Z\s]+$', ''),  # Code postal + ville
            (r'\s+FR\s*$', ''),  # Suffixe France
            
            # Normaliser les espaces et caractères
            (r'[^\w\s\-&\.]', ' '),  # Garder seulement alphanumériques et quelques caractères
            (r'\s+', ' '),  # Normaliser les espaces
        ]
    
    def _build_merchant_types(self) -> Dict[str, List[str]]:
        """
        Construit le dictionnaire des types de marchands basés sur des mots-clés.
        
        Returns:
            Dict: Type -> liste de mots-clés
        """
        return {
            "supermarket": [
                "supermarche", "hypermarche", "carrefour", "leclerc", "auchan", 
                "intermarche", "super u", "casino", "monoprix", "franprix"
            ],
            "restaurant": [
                "restaurant", "brasserie", "cafe", "bistrot", "mcdonalds", 
                "quick", "kfc", "subway", "pizza", "sushi"
            ],
            "gas_station": [
                "station", "essence", "total", "bp", "shell", "esso", "agip"
            ],
            "pharmacy": [
                "pharmacie", "parapharmacie"
            ],
            "transport": [
                "sncf", "ratp", "uber", "taxi", "metro", "bus", "parking"
            ],
            "subscription": [
                "netflix", "spotify", "amazon", "apple", "google", "microsoft",
                "orange", "sfr", "bouygues", "free"
            ],
            "bank": [
                "banque", "credit", "societe generale", "bnp", "caisse epargne",
                "mutuel", "banque populaire"
            ],
            "insurance": [
                "assurance", "mutuelle", "maif", "macif", "axa", "allianz"
            ],
            "retail": [
                "fnac", "darty", "boulanger", "conforama", "ikea", "leroy merlin",
                "castorama", "decathlon"
            ],
            "health": [
                "hopital", "clinique", "cabinet", "dentiste", "medecin", "laboratoire"
            ],
            "education": [
                "ecole", "universite", "formation", "cours"
            ]
        }
    
    async def normalize_merchant(self, raw_merchant_name: str) -> Dict[str, Any]:
        """
        Normalise un nom de marchand et retourne les informations enrichies.
        
        Args:
            raw_merchant_name: Nom brut du marchand
            
        Returns:
            Dict: Informations normalisées du marchand
        """
        if not raw_merchant_name or not raw_merchant_name.strip():
            return {
                "merchant_id": "unknown",
                "merchant_name": "Marchand inconnu",
                "merchant_type": None,
                "confidence": 0.0
            }
        
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="merchant_normalization"
        )
        
        # Vérifier le cache
        if raw_merchant_name in self._merchant_cache:
            cached_info = self._merchant_cache[raw_merchant_name]
            return self._merchant_info_to_dict(cached_info)
        
        # Nettoyer le nom
        cleaned_name = self._clean_merchant_name(raw_merchant_name)
        
        # Rechercher des marchands similaires existants
        similar_merchant = self._find_similar_merchant(cleaned_name)
        
        if similar_merchant:
            # Utiliser le marchand similaire existant
            merchant_info = similar_merchant
            # Ajouter cet alias
            merchant_info.aliases = merchant_info.aliases or []
            if raw_merchant_name not in merchant_info.aliases:
                merchant_info.aliases.append(raw_merchant_name)
        else:
            # Créer un nouveau marchand normalisé
            merchant_info = MerchantInfo(
                original_name=raw_merchant_name,
                normalized_name=cleaned_name,
                merchant_id=self._generate_merchant_id(cleaned_name),
                merchant_type=self._detect_merchant_type(cleaned_name),
                confidence=1.0,
                aliases=[raw_merchant_name]
            )
        
        # Mettre en cache
        self._merchant_cache[raw_merchant_name] = merchant_info
        
        # Ajouter aux marchands normalisés
        if merchant_info.normalized_name not in self._normalized_merchants:
            self._normalized_merchants[merchant_info.normalized_name] = set()
        self._normalized_merchants[merchant_info.normalized_name].add(raw_merchant_name)
        
        ctx_logger.debug(f"Marchand normalisé: '{raw_merchant_name}' -> '{merchant_info.normalized_name}'")
        
        return self._merchant_info_to_dict(merchant_info)
    
    def _clean_merchant_name(self, name: str) -> str:
        """
        Nettoie un nom de marchand en appliquant les patterns de nettoyage.
        
        Args:
            name: Nom brut du marchand
            
        Returns:
            str: Nom nettoyé
        """
        cleaned = name.strip()
        
        # Appliquer les patterns de nettoyage
        for pattern, replacement in self._cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Normaliser la casse
        cleaned = cleaned.strip().title()
        
        # Supprimer les mots vides en fin
        stop_words = ['Et', 'Le', 'La', 'Les', 'Du', 'De', 'Des', 'Au', 'Aux']
        words = cleaned.split()
        while words and words[-1] in stop_words:
            words.pop()
        
        cleaned = ' '.join(words) if words else cleaned
        
        # Limiter la longueur
        if len(cleaned) > 50:
            cleaned = cleaned[:50].strip()
        
        return cleaned if cleaned else "Marchand Inconnu"
    
    def _find_similar_merchant(self, cleaned_name: str) -> Optional[MerchantInfo]:
        """
        Recherche un marchand similaire dans le cache.
        
        Args:
            cleaned_name: Nom nettoyé à rechercher
            
        Returns:
            Optional[MerchantInfo]: Marchand similaire trouvé ou None
        """
        if not enrichment_settings.merchant_normalization_enabled:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for normalized_name in self._normalized_merchants.keys():
            similarity = SequenceMatcher(None, cleaned_name.lower(), normalized_name.lower()).ratio()
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                # Récupérer l'info du marchand depuis le cache
                for cached_merchant in self._merchant_cache.values():
                    if cached_merchant.normalized_name == normalized_name:
                        best_match = cached_merchant
                        break
        
        return best_match
    
    def _generate_merchant_id(self, normalized_name: str) -> str:
        """
        Génère un ID unique pour un marchand.
        
        Args:
            normalized_name: Nom normalisé du marchand
            
        Returns:
            str: ID unique du marchand
        """
        # Créer un ID basé sur le nom normalisé
        import hashlib
        
        # Nettoyer le nom pour l'ID
        clean_for_id = re.sub(r'[^\w]', '_', normalized_name.lower())
        clean_for_id = re.sub(r'_+', '_', clean_for_id).strip('_')
        
        # Ajouter un hash pour garantir l'unicité
        hash_suffix = hashlib.md5(normalized_name.encode()).hexdigest()[:8]
        
        return f"merchant_{clean_for_id}_{hash_suffix}"
    
    def _detect_merchant_type(self, merchant_name: str) -> Optional[str]:
        """
        Détecte automatiquement le type de marchand basé sur le nom.
        
        Args:
            merchant_name: Nom du marchand à analyser
            
        Returns:
            Optional[str]: Type de marchand détecté ou None
        """
        name_lower = merchant_name.lower()
        
        for merchant_type, keywords in self._merchant_types.items():
            for keyword in keywords:
                if keyword.lower() in name_lower:
                    return merchant_type
        
        return None
    
    def _merchant_info_to_dict(self, merchant_info: MerchantInfo) -> Dict[str, Any]:
        """
        Convertit un MerchantInfo en dictionnaire.
        
        Args:
            merchant_info: Informations du marchand
            
        Returns:
            Dict: Dictionnaire des informations
        """
        return {
            "merchant_id": merchant_info.merchant_id,
            "merchant_name": merchant_info.normalized_name,
            "merchant_type": merchant_info.merchant_type,
            "confidence": merchant_info.confidence,
            "original_name": merchant_info.original_name,
            "aliases": merchant_info.aliases or []
        }
    
    async def get_merchant_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur les marchands normalisés.
        
        Returns:
            Dict: Statistiques des marchands
        """
        total_merchants = len(self._normalized_merchants)
        total_aliases = sum(len(aliases) for aliases in self._normalized_merchants.values())
        
        # Compter par type
        type_counts = {}
        for merchant_info in self._merchant_cache.values():
            merchant_type = merchant_info.merchant_type or "unknown"
            type_counts[merchant_type] = type_counts.get(merchant_type, 0) + 1
        
        return {
            "total_normalized_merchants": total_merchants,
            "total_aliases": total_aliases,
            "average_aliases_per_merchant": total_aliases / total_merchants if total_merchants > 0 else 0,
            "merchant_types": type_counts,
            "cache_size": len(self._merchant_cache)
        }
    
    async def get_merchant_suggestions(self, partial_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retourne des suggestions de marchands basées sur un nom partiel.
        
        Args:
            partial_name: Nom partiel à rechercher
            limit: Nombre maximum de suggestions
            
        Returns:
            List[Dict]: Liste des suggestions
        """
        suggestions = []
        partial_lower = partial_name.lower()
        
        for merchant_info in self._merchant_cache.values():
            # Vérifier si le nom partiel correspond
            if partial_lower in merchant_info.normalized_name.lower():
                similarity = SequenceMatcher(None, partial_lower, merchant_info.normalized_name.lower()).ratio()
                suggestions.append({
                    **self._merchant_info_to_dict(merchant_info),
                    "similarity": similarity
                })
        
        # Trier par similarité et limiter
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions[:limit]
    
    async def merge_merchants(self, primary_merchant_id: str, secondary_merchant_id: str) -> bool:
        """
        Fusionne deux marchands en un seul.
        
        Args:
            primary_merchant_id: ID du marchand principal
            secondary_merchant_id: ID du marchand à fusionner
            
        Returns:
            bool: True si la fusion a réussi
        """
        primary_merchant = None
        secondary_merchant = None
        
        # Trouver les marchands dans le cache
        for merchant_info in self._merchant_cache.values():
            if merchant_info.merchant_id == primary_merchant_id:
                primary_merchant = merchant_info
            elif merchant_info.merchant_id == secondary_merchant_id:
                secondary_merchant = merchant_info
        
        if not primary_merchant or not secondary_merchant:
            return False
        
        # Fusionner les aliases
        primary_merchant.aliases = primary_merchant.aliases or []
        secondary_merchant.aliases = secondary_merchant.aliases or []
        
        all_aliases = list(set(primary_merchant.aliases + secondary_merchant.aliases))
        primary_merchant.aliases = all_aliases
        
        # Mettre à jour le cache pour tous les aliases du marchand secondaire
        for alias in secondary_merchant.aliases:
            if alias in self._merchant_cache:
                self._merchant_cache[alias] = primary_merchant
        
        # Supprimer le marchand secondaire des marchands normalisés
        if secondary_merchant.normalized_name in self._normalized_merchants:
            # Transférer les aliases vers le marchand principal
            secondary_aliases = self._normalized_merchants[secondary_merchant.normalized_name]
            if primary_merchant.normalized_name not in self._normalized_merchants:
                self._normalized_merchants[primary_merchant.normalized_name] = set()
            self._normalized_merchants[primary_merchant.normalized_name].update(secondary_aliases)
            
            # Supprimer l'entrée secondaire
            del self._normalized_merchants[secondary_merchant.normalized_name]
        
        logger.info(f"Marchands fusionnés: {secondary_merchant.normalized_name} -> {primary_merchant.normalized_name}")
        return True
    
    async def export_merchants(self) -> List[Dict[str, Any]]:
        """
        Exporte tous les marchands normalisés.
        
        Returns:
            List[Dict]: Liste de tous les marchands
        """
        merchants = []
        seen_ids = set()
        
        for merchant_info in self._merchant_cache.values():
            if merchant_info.merchant_id not in seen_ids:
                merchants.append(self._merchant_info_to_dict(merchant_info))
                seen_ids.add(merchant_info.merchant_id)
        
        return merchants
    
    async def import_merchants(self, merchants_data: List[Dict[str, Any]]) -> int:
        """
        Importe des marchands depuis des données externes.
        
        Args:
            merchants_data: Données des marchands à importer
            
        Returns:
            int: Nombre de marchands importés
        """
        imported_count = 0
        
        for merchant_data in merchants_data:
            try:
                merchant_info = MerchantInfo(
                    original_name=merchant_data.get("original_name", ""),
                    normalized_name=merchant_data.get("merchant_name", ""),
                    merchant_id=merchant_data.get("merchant_id", ""),
                    merchant_type=merchant_data.get("merchant_type"),
                    confidence=merchant_data.get("confidence", 1.0),
                    aliases=merchant_data.get("aliases", [])
                )
                
                # Ajouter au cache
                for alias in merchant_info.aliases:
                    self._merchant_cache[alias] = merchant_info
                
                # Ajouter aux marchands normalisés
                if merchant_info.normalized_name not in self._normalized_merchants:
                    self._normalized_merchants[merchant_info.normalized_name] = set()
                self._normalized_merchants[merchant_info.normalized_name].update(merchant_info.aliases)
                
                imported_count += 1
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'import du marchand {merchant_data}: {e}")
        
        logger.info(f"Importé {imported_count} marchands")
        return imported_count
    
    def clear_cache(self):
        """Vide le cache des marchands."""
        self._merchant_cache.clear()
        self._normalized_merchants.clear()
        logger.info("Cache des marchands vidé")
    
    async def optimize_merchants(self) -> Dict[str, Any]:
        """
        Optimise la base de marchands en détectant et fusionnant les doublons.
        
        Returns:
            Dict: Résultat de l'optimisation
        """
        ctx_logger = get_contextual_logger(
            __name__,
            enrichment_type="merchant_optimization"
        )
        
        original_count = len(self._normalized_merchants)
        merges_performed = 0
        
        # Créer une liste des marchands pour éviter la modification pendant l'itération
        merchants_list = list(self._normalized_merchants.keys())
        
        for i, merchant1 in enumerate(merchants_list):
            for merchant2 in merchants_list[i+1:]:
                if merchant1 not in self._normalized_merchants or merchant2 not in self._normalized_merchants:
                    continue  # Déjà fusionné
                
                similarity = SequenceMatcher(None, merchant1.lower(), merchant2.lower()).ratio()
                
                if similarity >= 0.9:  # Seuil élevé pour les fusions automatiques
                    # Trouver les IDs des marchands
                    merchant1_id = None
                    merchant2_id = None
                    
                    for merchant_info in self._merchant_cache.values():
                        if merchant_info.normalized_name == merchant1:
                            merchant1_id = merchant_info.merchant_id
                        elif merchant_info.normalized_name == merchant2:
                            merchant2_id = merchant_info.merchant_id
                    
                    if merchant1_id and merchant2_id:
                        success = await self.merge_merchants(merchant1_id, merchant2_id)
                        if success:
                            merges_performed += 1
                            ctx_logger.debug(f"Fusion automatique: {merchant2} -> {merchant1}")
        
        final_count = len(self._normalized_merchants)
        
        result = {
            "original_count": original_count,
            "final_count": final_count,
            "merges_performed": merges_performed,
            "reduction": original_count - final_count
        }
        
        ctx_logger.info(f"Optimisation terminée: {merges_performed} fusions, {result['reduction']} réduction")
        
        return result