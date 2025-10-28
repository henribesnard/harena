import time
import hashlib
from typing import Any, Optional, Dict, Tuple


class MultiLevelCache:
    """Cache simple en mémoire avec support optionnel du TTL.

    Malgré son nom, cette implémentation ne fournit **qu'une seule couche** de
    cache en mémoire. Elle offre simplement l'interface asynchrone attendue par
    ``SearchEngine`` sans dépendre de ``conversation_service`` ni d'une base
    Redis. Les valeurs sont stockées dans un dictionnaire avec, pour chaque
    entrée, un éventuel horodatage d'expiration. Les clés sont en outre
    préfixées par ``user_id`` afin d'isoler les données entre utilisateurs.
    """

    def __init__(self) -> None:
        # Dictionnaire interne : {"user_id:key": (timestamp_expiration, valeur)}
        self._store: Dict[str, Tuple[Optional[float], Any]] = {}

    def _format_key(self, user_id: int, key: str) -> str:
        return f"{user_id}:{key}"

    async def get(self, user_id: int, key: str) -> Any:
        """Récupère la valeur mise en cache pour ``user_id``.

        Retourne ``None`` si la clé est absente ou expirée."""
        namespaced_key = self._format_key(user_id, key)
        item = self._store.get(namespaced_key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            # Entrée expirée : on la supprime et on se comporte comme un miss
            self._store.pop(namespaced_key, None)
            return None
        return value

    async def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur pour ``user_id`` avec un TTL optionnel (en secondes)."""
        namespaced_key = self._format_key(user_id, key)
        expires_at = time.time() + ttl if ttl is not None else None
        self._store[namespaced_key] = (expires_at, value)

    async def clear(self) -> None:
        """Supprime toutes les entrées du cache."""
        self._store.clear()

    async def clear_user(self, user_id: int) -> int:
        """Supprime toutes les entrées du cache pour un utilisateur spécifique.

        Args:
            user_id: ID de l'utilisateur dont le cache doit être invalidé

        Returns:
            int: Nombre d'entrées supprimées
        """
        prefix = f"{user_id}:"
        keys_to_delete = [key for key in self._store.keys() if key.startswith(prefix)]
        for key in keys_to_delete:
            self._store.pop(key, None)
        return len(keys_to_delete)


def generate_cache_key(prefix: str, **parts: Any) -> str:
    """Génère une clé de cache déterministe à partir des éléments fournis.

    Les éléments sont sérialisés dans un ordre stable puis hachés afin
    d'éviter des clés trop longues."""
    raw = "|".join(f"{k}:{parts[k]}" for k in sorted(parts))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{prefix}:{digest}"
