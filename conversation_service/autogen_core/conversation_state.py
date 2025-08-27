"""Etat de conversation persistant pour gestion multi-tours AutoGen"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ConversationTurn:
    """Représente un échange dans la conversation."""
    role: str
    content: str


@dataclass
class ConversationState:
    """Gère l'historique des messages multi-tours."""
    turns: List[ConversationTurn] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.turns.append(ConversationTurn(role="user", content=content))

    def add_agent_message(self, content: str) -> None:
        self.turns.append(ConversationTurn(role="assistant", content=content))

    def to_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_messages(), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ConversationState":
        messages = json.loads(data)
        state = cls()
        for msg in messages:
            state.turns.append(ConversationTurn(role=msg["role"], content=msg["content"]))
        return state
