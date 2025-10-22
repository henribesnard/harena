"""
Category & Operation Type Validator - Valide et corrige les filtres avant exécution
Utilise MetadataService pour mapper les catégories/operations approximatives vers les noms exacts
Performance: < 1 seconde (utilise le cache du MetadataService)
"""
import logging
import re
from typing import Dict, Any, Optional, List
from ..services.metadata_service import metadata_service

logger = logging.getLogger(__name__)


class CategoryValidator:
    """
    Valide et corrige automatiquement les filtres de catégories et operation_types

    Responsabilités:
    - Mapper les noms approximatifs vers les noms exacts
    - Corriger les pluriels, accents, synonymes
    - AJOUTER automatiquement category_name si détecté dans le message
    - Valider les operation_types
    - Logger toutes les corrections pour monitoring
    - Performance: < 1 seconde (utilise cache)
    """

    def __init__(self):
        self.metadata_service = metadata_service
        self.stats = {
            "category_corrections": 0,
            "category_additions": 0,
            "operation_type_corrections": 0,
            "total_validations": 0
        }

    def validate_and_correct_filters(
        self,
        filters: Dict[str, Any],
        user_message: str = None
    ) -> Dict[str, Any]:
        """
        Valide et corrige les filtres de catégorie et operation_type

        Args:
            filters: Dictionnaire de filtres Elasticsearch
            user_message: Message utilisateur (pour détection automatique si category manquante)

        Returns:
            Dictionnaire de filtres corrigés
        """
        self.stats["total_validations"] += 1
        corrected_filters = filters.copy() if filters else {}

        # === 1. VALIDATION ET CORRECTION DES CATÉGORIES ===
        corrected_filters = self._validate_category(corrected_filters, user_message)

        # === 2. VALIDATION ET CORRECTION DES OPERATION_TYPES ===
        corrected_filters = self._validate_operation_type(corrected_filters)

        return corrected_filters

    def _validate_category(
        self,
        filters: Dict[str, Any],
        user_message: str = None
    ) -> Dict[str, Any]:
        """
        Valide et corrige le filtre category_name
        Si manquant, tente de le détecter depuis le message utilisateur
        """
        corrected_filters = filters.copy()

        # CAS 1: category_name existe → valider et corriger
        if 'category_name' in filters:
            category_filter = filters['category_name']

            # CAS 1A: Liste de catégories → valider chaque élément
            if isinstance(category_filter, list):
                validated_categories = []
                for cat in category_filter:
                    exact_category = self.metadata_service.search_category_by_name(str(cat))
                    if exact_category:
                        validated_categories.append(exact_category)
                        if exact_category != cat:
                            logger.debug(f"Category in list corrected: '{cat}' → '{exact_category}'")
                            self.stats["category_corrections"] += 1
                    else:
                        # Garder la valeur originale si non trouvée
                        validated_categories.append(cat)
                        logger.warning(f"Category '{cat}' not found in list, keeping original")

                # IMPORTANT: Garder le format liste
                corrected_filters['category_name'] = validated_categories
                logger.info(f"✅ Validated category list: {len(validated_categories)} categories")

            # CAS 1B: Catégorie unique → comportement original
            else:
                search_value = self._extract_filter_value(category_filter)

                if search_value:
                    exact_category = self.metadata_service.search_category_by_name(search_value)

                    if exact_category and exact_category != search_value:
                        logger.info(f"✅ Category corrected: '{search_value}' → '{exact_category}'")
                        self.stats["category_corrections"] += 1

                        # Mettre à jour avec le nom exact
                        corrected_filters['category_name'] = self._rebuild_filter(
                            filters['category_name'],
                            exact_category
                        )

                    elif exact_category:
                        logger.debug(f"✓ Category already exact: '{search_value}'")

                    else:
                        logger.warning(
                            f"⚠️  Category not found: '{search_value}' "
                            f"(will try text search fallback)"
                        )

        # CAS 2: category_name MANQUE → essayer de le détecter
        elif user_message:
            detected_category = self._detect_category_from_message(user_message)

            if detected_category:
                logger.info(f"✨ Category auto-detected from message: '{detected_category}'")
                self.stats["category_additions"] += 1
                corrected_filters['category_name'] = {'match': detected_category}

        return corrected_filters

    def _validate_operation_type(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et corrige le filtre operation_type
        """
        if 'operation_type' not in filters:
            return filters

        corrected_filters = filters.copy()
        search_value = self._extract_filter_value(filters['operation_type'])

        if not search_value:
            return filters

        # Mapper vers operation_type exact
        exact_operation = self._find_exact_operation_type(search_value)

        if exact_operation and exact_operation != search_value:
            logger.info(f"✅ Operation type corrected: '{search_value}' → '{exact_operation}'")
            self.stats["operation_type_corrections"] += 1

            corrected_filters['operation_type'] = self._rebuild_filter(
                filters['operation_type'],
                exact_operation
            )

        elif exact_operation:
            logger.debug(f"✓ Operation type already exact: '{search_value}'")

        else:
            logger.warning(f"⚠️  Unknown operation_type: '{search_value}'")

        return corrected_filters

    def _detect_category_from_message(self, message: str) -> Optional[str]:
        """
        Détecte automatiquement une catégorie depuis le message utilisateur
        Utilise les synonymes du MetadataService

        Performance: O(n*m) où n=nombre de catégories, m=nombre de synonymes
        Avec 37 catégories et ~100 synonymes, prend < 50ms
        """
        if not message:
            return None

        message_lower = message.lower()

        # Chercher dans CATEGORY_SYNONYMS (définis dans MetadataService)
        category_synonyms = self.metadata_service.CATEGORY_SYNONYMS

        # Recherche par correspondance exacte de mots
        for category_name, synonyms in category_synonyms.items():
            for synonym in synonyms:
                # Chercher le synonyme comme mot complet (avec word boundaries)
                pattern = r'\b' + re.escape(synonym.lower()) + r's?\b'  # Accepte pluriel
                # ✅ CORRECTIF: Chercher dans message_lower (pas message_clean)
                # pour préserver les stopwords dans les expressions multi-mots comme "frais de santé"
                if re.search(pattern, message_lower):
                    # Vérifier que cette catégorie existe bien en base
                    exact = self.metadata_service.search_category_by_name(category_name)
                    if exact:
                        logger.debug(f"Detected '{synonym}' → category '{exact}'")
                        return exact

        return None

    def _find_exact_operation_type(self, search_value: str) -> Optional[str]:
        """
        Trouve l'operation_type exact depuis une valeur approximative
        """
        if not search_value:
            return None

        search_lower = search_value.lower().strip()

        # Recherche dans les operation_types définis dans MetadataService
        operation_types = self.metadata_service.OPERATION_TYPES

        # 1. Recherche exacte (insensible à la casse)
        for op_type in operation_types.keys():
            if op_type.lower() == search_lower:
                return op_type

        # 2. Recherche dans les synonymes
        for op_type, synonyms in operation_types.items():
            if search_lower in [s.lower() for s in synonyms]:
                return op_type

        # 3. Recherche partielle
        for op_type in operation_types.keys():
            if search_lower in op_type.lower() or op_type.lower() in search_lower:
                return op_type

        return None

    def _extract_filter_value(self, filter_obj: Any) -> Optional[str]:
        """
        Extrait la valeur string depuis un filtre (peut être str, dict, list)
        """
        if isinstance(filter_obj, str):
            return filter_obj

        elif isinstance(filter_obj, dict):
            # Format: {"match": "value"} ou {"term": "value"}
            value = filter_obj.get('match') or filter_obj.get('term')
            if isinstance(value, str):
                return value
            elif isinstance(value, list) and len(value) > 0:
                return value[0]

        return None

    def _rebuild_filter(self, original_filter: Any, new_value: str) -> Any:
        """
        Reconstruit un filtre avec une nouvelle valeur, en conservant la structure
        """
        if isinstance(original_filter, str):
            return new_value

        elif isinstance(original_filter, dict):
            # Conserver la structure mais avec nouvelle valeur
            if 'match' in original_filter:
                return {'match': new_value}
            elif 'term' in original_filter:
                return {'term': new_value}
            elif 'terms' in original_filter:
                return {'terms': [new_value]}
            else:
                return new_value

        return new_value

    def _clean_message(self, message: str) -> str:
        """
        Nettoie le message en enlevant articles, prépositions communes
        """
        # Articles et prépositions à enlever
        stopwords = [
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de',
            'en', 'dans', 'sur', 'pour', 'avec', 'mes', 'mon', 'ma',
            'ce', 'cette', 'ces', 'chez', 'au', 'aux'
        ]

        words = message.split()
        filtered_words = [w for w in words if w not in stopwords]
        return ' '.join(filtered_words)

    def get_correction_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de correction (pour monitoring)"""
        return {
            "validator": "category_and_operation_type",
            "categories_loaded": len(self.metadata_service.get_all_categories()),
            "operation_types_available": len(self.metadata_service.OPERATION_TYPES),
            "stats": self.stats.copy()
        }

    def reset_stats(self):
        """Reset les statistiques"""
        self.stats = {
            "category_corrections": 0,
            "category_additions": 0,
            "operation_type_corrections": 0,
            "total_validations": 0
        }


# Instance singleton
category_validator = CategoryValidator()
