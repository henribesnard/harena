"""Custom exception definitions for Conversation Service."""


class HarenaException(Exception):
    """Base exception carrying an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConversationNotFound(HarenaException):
    """Raised when a conversation cannot be found."""

    def __init__(self, conversation_id: str):
        super().__init__(
            f"Conversation '{conversation_id}' not found", status_code=404
        )
        self.conversation_id = conversation_id


class UnauthorizedAction(HarenaException):
    """Raised when a user is not authorized to perform an action."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)
