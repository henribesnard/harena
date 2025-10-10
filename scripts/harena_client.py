#!/usr/bin/env python3
"""
Harena API Client - Best Practices
Gestion intelligente des tokens JWT avec auto-refresh
"""
import requests
import json
import jwt
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HarenaAPIClient:
    """
    Client API Harena avec gestion automatique des tokens JWT

    Features:
    - Auto-refresh du token avant expiration
    - Cache du token dans un fichier sécurisé
    - Validation du token avant chaque requête
    - Retry automatique en cas de 401
    """

    def __init__(
        self,
        base_url: str = "http://52.210.228.191",
        email: str = "henri@example.com",
        password: str = "Henri123456",
        token_file: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.token_file = Path(token_file) if token_file else Path.home() / '.harena_token'

        self._token: Optional[str] = None
        self._token_exp: Optional[datetime] = None

        # Charger le token depuis le cache si disponible
        self._load_token_from_cache()

    def _load_token_from_cache(self) -> bool:
        """Charge le token depuis le fichier cache s'il existe et est valide"""
        if not self.token_file.exists():
            return False

        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                token = data.get('token')

                if token and self._is_token_valid(token):
                    self._token = token
                    self._token_exp = datetime.fromtimestamp(
                        jwt.decode(token, options={"verify_signature": False})['exp'],
                        tz=timezone.utc
                    )
                    logger.info("Token valide charge depuis le cache")
                    return True
                else:
                    logger.info("Token en cache expire ou invalide")
                    return False
        except Exception as e:
            logger.warning(f"Erreur lecture cache token: {e}")
            return False

    def _save_token_to_cache(self, token: str) -> None:
        """Sauvegarde le token dans le fichier cache"""
        try:
            # Créer le fichier avec permissions restreintes (600)
            self.token_file.touch(mode=0o600, exist_ok=True)

            with open(self.token_file, 'w') as f:
                json.dump({
                    'token': token,
                    'email': self.email,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }, f)

            logger.info(f"Token sauvegarde dans {self.token_file}")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde token: {e}")

    def _is_token_valid(self, token: str, buffer_seconds: int = 300) -> bool:
        """
        Vérifie si un token est valide (non expiré avec buffer de sécurité)

        Args:
            token: Le token JWT à vérifier
            buffer_seconds: Marge de sécurité avant expiration (5 min par défaut)
        """
        try:
            # Décoder le token sans vérifier la signature (validation locale)
            payload = jwt.decode(token, options={"verify_signature": False})
            exp_timestamp = payload.get('exp')

            if not exp_timestamp:
                return False

            exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            now = datetime.now(timezone.utc)

            # Vérifier avec buffer de sécurité
            time_until_expiry = (exp_time - now).total_seconds()

            if time_until_expiry <= buffer_seconds:
                logger.info(f"Token expire dans {time_until_expiry:.0f}s (buffer: {buffer_seconds}s)")
                return False

            logger.debug(f"Token valide pour encore {time_until_expiry:.0f}s")
            return True

        except jwt.DecodeError:
            logger.warning("Token invalide (decode error)")
            return False
        except Exception as e:
            logger.warning(f"Erreur validation token: {e}")
            return False

    def _login(self) -> str:
        """
        Authentification et récupération d'un nouveau token

        Returns:
            Le token JWT

        Raises:
            Exception si l'authentification échoue
        """
        url = f"{self.base_url}:8000/api/v1/users/auth/login"

        payload = f"username={self.email}&password={self.password}"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        logger.info(f"Authentification pour {self.email}...")

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)

            if response.status_code == 200:
                token = response.json()['access_token']

                # Sauvegarder le token
                self._token = token
                self._token_exp = datetime.fromtimestamp(
                    jwt.decode(token, options={"verify_signature": False})['exp'],
                    tz=timezone.utc
                )
                self._save_token_to_cache(token)

                logger.info("Authentification reussie")
                return token

            elif response.status_code == 401:
                raise Exception("Identifiants incorrects - Verifiez email/password")
            else:
                raise Exception(f"Erreur authentification: {response.status_code} - {response.text}")

        except requests.RequestException as e:
            raise Exception(f"Erreur reseau lors de l'authentification: {e}")

    def get_valid_token(self, force_refresh: bool = False) -> str:
        """
        Retourne un token valide, en le rafraîchissant si nécessaire

        Args:
            force_refresh: Force le refresh même si le token est valide

        Returns:
            Token JWT valide
        """
        # Si force refresh ou pas de token ou token invalide
        if force_refresh or not self._token or not self._is_token_valid(self._token):
            logger.info("Rafraichissement du token necessaire")
            return self._login()

        return self._token

    def send_message(
        self,
        message: str,
        user_id: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Envoie un message au conversation service

        Args:
            message: Le message utilisateur
            user_id: ID de l'utilisateur
            **kwargs: Paramètres optionnels (message_type, priority, client_info, etc.)

        Returns:
            Réponse JSON complète du serveur
        """
        url = f"{self.base_url}:8001/api/v1/conversation/{user_id}"

        # Construire le payload
        payload = {
            "message": message,
            "message_type": kwargs.get("message_type", "text"),
            "priority": kwargs.get("priority", "normal"),
            "client_info": kwargs.get("client_info", {
                "platform": "api",
                "version": "1.0.0"
            })
        }

        # Ajouter les champs optionnels s'ils sont fournis
        if "context_hints" in kwargs:
            payload["context_hints"] = kwargs["context_hints"]
        if "preferences" in kwargs:
            payload["preferences"] = kwargs["preferences"]

        # Première tentative avec le token actuel
        token = self.get_valid_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        logger.info(f"Envoi message: '{message[:50]}...'")

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        # Si 401, refresh le token et retry une fois
        if response.status_code == 401:
            logger.warning("Token rejete (401) - Tentative de refresh...")
            token = self.get_valid_token(force_refresh=True)
            headers['Authorization'] = f'Bearer {token}'
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        # Traiter la réponse
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Reponse recue - {result['search_summary']['total_results']} resultats")
            return result
        else:
            logger.error(f"Erreur {response.status_code}: {response.text}")
            response.raise_for_status()

    def search_transactions(
        self,
        user_id: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        page_size: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Recherche directe de transactions via le search service

        Args:
            user_id: ID utilisateur
            filters: Filtres de recherche
            page_size: Nombre de résultats par page
            **kwargs: Autres paramètres (sort, aggregations, etc.)
        """
        url = f"{self.base_url}:8005/api/v1/search"

        payload = {
            "user_id": user_id,
            "filters": filters or {},
            "page_size": page_size,
            **kwargs
        }

        token = self.get_valid_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        logger.info(f"Recherche transactions pour user {user_id}")

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # Retry avec refresh si 401
        if response.status_code == 401:
            token = self.get_valid_token(force_refresh=True)
            headers['Authorization'] = f'Bearer {token}'
            response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"{result.get('total_hits', 0)} resultats trouves")
            return result
        else:
            logger.error(f"Erreur {response.status_code}: {response.text}")
            response.raise_for_status()

    def clear_token_cache(self) -> None:
        """Supprime le token en cache (force re-login au prochain appel)"""
        if self.token_file.exists():
            self.token_file.unlink()
            logger.info("Cache token supprime")
        self._token = None
        self._token_exp = None


# ============================================================================
# Exemples d'utilisation
# ============================================================================

def example_basic_usage():
    """Exemple d'utilisation basique"""

    # Créer le client (va automatiquement charger ou créer un token)
    client = HarenaAPIClient(
        email="henri@example.com",
        password="Henri123456"  # ⚠️ Utilisez votre nouveau mot de passe ici
    )

    # Envoyer un message
    result = client.send_message("Mes dépenses de plus de 100 euros")

    # Afficher les résultats
    print(f"\n{'='*80}")
    print(f"Intent: {result['intent']['type']}")
    print(f"Résultats: {result['search_summary']['total_results']}")
    print(f"\n{result['response']['message'][:500]}...")
    print(f"{'='*80}\n")


def example_multiple_requests():
    """Exemple avec plusieurs requêtes (le token est réutilisé automatiquement)"""

    client = HarenaAPIClient(
        email="henri@example.com",
        password="Henri123456"
    )

    questions = [
        "Mes transactions chez Carrefour",
        "Combien j'ai dépensé en mai",
        "Mes achats de plus de 200 euros"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = client.send_message(question)
        print(f"   -> {result['search_summary']['total_results']} resultats trouves")


def example_direct_search():
    """Exemple de recherche directe (sans passer par conversation)"""

    client = HarenaAPIClient(
        email="henri@example.com",
        password="Henri123456"
    )

    # Recherche avec filtres
    result = client.search_transactions(
        filters={
            "transaction_type": "debit",
            "merchant_name": ["Carrefour", "Amazon"]
        },
        page_size=10,
        sort=[{"date": {"order": "desc"}}]
    )

    print(f"\nResultats: {result['total_hits']} transactions")
    for tx in result['results'][:5]:
        print(f"  - {tx['date']}: {tx['merchant_name']} - {tx['amount']}EUR")


if __name__ == "__main__":
    # Choisir l'exemple à exécuter
    print("Harena API Client - Best Practices\n")

    # Exemple basique
    example_basic_usage()

    # Pour forcer un nouveau login (utile si vous avez change le mot de passe):
    # client = HarenaAPIClient(email="henri@example.com", password="NOUVEAU_MOT_DE_PASSE")
    # client.clear_token_cache()  # Supprime l'ancien token
    # result = client.send_message("Test")  # Va se re-authentifier automatiquement
