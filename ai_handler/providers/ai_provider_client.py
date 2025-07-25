from __future__ import annotations
from abc import ABC, abstractmethod
import typing as t


class AIChat(ABC):

    def __init__(self, chat_id: str):
        self.chat_id = chat_id

    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> str:
        """Send a prompt in this chat context."""
        pass

    @property
    @abstractmethod
    def chat_id(self) -> str:
        """Unique identifier for this chat context. used to retrieve or continue the chat."""
        pass

    @chat_id.setter
    @abstractmethod
    def chat_id(self, value: str):
        """Set the unique identifier for this chat context."""
        pass


class AiProviderClient(ABC):
    """
    Abstract base class for all AI model clients.
    """

    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the AI model and return the response.
        """
        pass

    @property
    def chats(self) -> dict[t.Any, AIChat]:
        """
        List of chat contexts created this session.
        raises NotImplementedError if the provider does not support chat contexts.
        """
        raise NotImplementedError(
            "This provider does not support chat contexts. "
            "Please use a different provider or implement chat support."
        )
