"""
Token counter utility for managing LLM context size
Prevents exceeding token limits and controls costs
"""
import logging
from typing import List, Dict, Any
import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Utility to count tokens for OpenAI models
    Helps enforce MAX_CONTEXT_TOKENS limits
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize token counter for specific model

        Args:
            model: OpenAI model name (gpt-4o-mini, gpt-4o, etc.)
        """
        self.model = model
        try:
            # Essayer de charger l'encodeur pour le modèle spécifique
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback vers cl100k_base (utilisé par GPT-4 et GPT-3.5-turbo)
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback approximation: ~4 chars per token
            return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens for a list of chat messages (OpenAI format)
        Includes overhead for message formatting

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}

        Returns:
            Total tokens including formatting overhead
        """
        if not messages:
            return 0

        total_tokens = 0

        # Overhead par message selon le format OpenAI
        # Chaque message ajoute ~4 tokens de formatage
        tokens_per_message = 4

        for message in messages:
            total_tokens += tokens_per_message

            # Compter les tokens du rôle
            if "role" in message:
                total_tokens += self.count_tokens(message["role"])

            # Compter les tokens du contenu
            if "content" in message:
                total_tokens += self.count_tokens(message["content"])

        # Overhead global pour la conversation
        total_tokens += 2

        return total_tokens

    def truncate_messages_to_limit(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        keep_system_message: bool = True
    ) -> List[Dict[str, str]]:
        """
        Truncate message history to fit within token limit
        Keeps most recent messages and optionally preserves system message

        Args:
            messages: List of messages in chronological order
            max_tokens: Maximum tokens allowed
            keep_system_message: If True, always keep first message if role=system

        Returns:
            Truncated list of messages that fit within limit
        """
        if not messages:
            return []

        # Séparer le message système si présent
        system_message = None
        conversation_messages = messages

        if keep_system_message and messages[0].get("role") == "system":
            system_message = messages[0]
            conversation_messages = messages[1:]

        # Compter tokens du message système
        system_tokens = 0
        if system_message:
            system_tokens = self.count_messages_tokens([system_message])

        # Budget restant pour la conversation
        remaining_budget = max_tokens - system_tokens

        if remaining_budget <= 0:
            logger.warning(f"System message exceeds token limit: {system_tokens} > {max_tokens}")
            return [system_message] if system_message else []

        # Parcourir les messages du plus récent au plus ancien
        selected_messages = []
        current_tokens = 0

        for message in reversed(conversation_messages):
            message_tokens = self.count_messages_tokens([message])

            if current_tokens + message_tokens <= remaining_budget:
                selected_messages.insert(0, message)  # Insérer au début pour garder l'ordre
                current_tokens += message_tokens
            else:
                # Plus de budget, arrêter
                break

        # Reconstruire avec le message système
        final_messages = []
        if system_message:
            final_messages.append(system_message)
        final_messages.extend(selected_messages)

        logger.debug(
            f"Truncated {len(messages)} messages to {len(final_messages)} "
            f"({current_tokens + system_tokens}/{max_tokens} tokens)"
        )

        return final_messages

    def estimate_response_tokens(self, max_tokens: int = 1000) -> int:
        """
        Estimate tokens needed for LLM response

        Args:
            max_tokens: Maximum tokens to generate

        Returns:
            Estimated tokens for response
        """
        return max_tokens


# Instance globale
_token_counter = None


def get_token_counter(model: str = "gpt-4o-mini") -> TokenCounter:
    """
    Get or create global token counter instance

    Args:
        model: OpenAI model name

    Returns:
        TokenCounter instance
    """
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter(model)
    return _token_counter
