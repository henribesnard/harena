# conversation_service/__init__.py
"""
Service de conversation intelligente pour Harena.

Ce service gère les conversations avec l'assistant IA, incluant la détection
d'intention, la génération de réponses et l'intégration avec le service de recherche.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]

# ============================================================
# conversation_service/api/__init__.py

"""
Module API pour le service de conversation.

Contient les routes REST et WebSocket pour les conversations
et la gestion de l'historique.
"""

__all__ = []

# ============================================================
# conversation_service/core/__init__.py

"""
Module core contenant la logique métier de la conversation.

Inclut le gestionnaire de conversation, la détection d'intention,
le client DeepSeek et le formateur de requêtes.
"""

__all__ = []

# ============================================================
# conversation_service/storage/__init__.py

"""
Module de stockage pour les conversations.

Fournit l'interface pour la persistance des conversations
et messages dans PostgreSQL.
"""

__all__ = []

# ============================================================
# conversation_service/prompts/__init__.py

"""
Module des prompts pour l'IA.

Contient les prompts système pour la détection d'intention
et la génération de réponses contextuelles.
"""

__all__ = []

# ============================================================
# conversation_service/utils/__init__.py

"""
Utilitaires pour le service de conversation.

Inclut le compteur de tokens, les outils de streaming
et les fonctions d'aide.
"""

__all__ = []