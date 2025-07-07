from typing import Any, Iterable, AsyncIterable, Protocol, runtime_checkable
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# --- Protocol Definitions for Type Safety ---

@runtime_checkable
class ChatCompletionsProtocol(Protocol):
    """Protocol for chat completions interface."""
    def create(self, **kwargs: Any) -> ChatCompletion | str | Iterable[ChatCompletionChunk] | Iterable[str]: ...

@runtime_checkable
class AsyncChatCompletionsProtocol(Protocol):
    """Protocol for async chat completions interface."""
    async def create(self, **kwargs: Any) -> ChatCompletion | str | AsyncIterable[ChatCompletionChunk] | AsyncIterable[str]: ...

@runtime_checkable
class ChatProtocol(Protocol):
    """Protocol for chat interface."""
    completions: ChatCompletionsProtocol

@runtime_checkable
class AsyncChatProtocol(Protocol):
    """Protocol for async chat interface."""
    completions: AsyncChatCompletionsProtocol

@runtime_checkable
class OpenAIProtocol(Protocol):
    """Protocol for OpenAI client interface."""
    chat: ChatProtocol
    def __getattr__(self, name: str) -> Any: ...

@runtime_checkable
class AsyncOpenAIProtocol(Protocol):
    """Protocol for AsyncOpenAI client interface."""
    chat: AsyncChatProtocol
    async def __aenter__(self) -> AsyncOpenAIProtocol: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
