from typing import Any, Iterable, AsyncIterable, Protocol, runtime_checkable
from anthropic.types import Message, RawMessageStreamEvent

# --- Protocol Definitions for Type Safety ---

@runtime_checkable
class MessagesProtocol(Protocol):
    """Protocol for messages interface."""
    def create(self, **kwargs: Any) -> Message | str | Iterable[RawMessageStreamEvent]: ...

@runtime_checkable
class AsyncMessagesProtocol(Protocol):
    """Protocol for async messages interface."""
    async def create(self, **kwargs: Any) -> Message | str | AsyncIterable[RawMessageStreamEvent]: ...

@runtime_checkable
class AnthropicProtocol(Protocol):
    """Protocol for Anthropic client interface."""
    messages: MessagesProtocol
    def __getattr__(self, name: str) -> Any: ...

@runtime_checkable
class AsyncAnthropicProtocol(Protocol):
    """Protocol for AsyncAnthropic client interface."""
    messages: AsyncMessagesProtocol
    async def __aenter__(self) -> "AsyncAnthropicProtocol": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ... 
