# Conversation Service Core

This directory contains the core components that power the conversation
service such as the high level ``ConversationService`` and the
database ``transaction_manager`` helper.

These modules are specific to the conversation domain.  Generic helpers
shared by multiple services live in the top-level `core/` package and are
intentionally kept separate to avoid circular dependencies and to provide
clearer boundaries between shared utilities and service-specific logic.

Projects importing conversation-related primitives should use the
re-exported classes from this package:

```python
from conversation_service.core import ConversationService
```

This keeps the service's public API explicit while retaining a clean
separation from the shared `core/` helpers.
