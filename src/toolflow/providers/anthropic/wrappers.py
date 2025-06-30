from typing import Any, List, Dict, overload, Iterable, AsyncIterable
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageStreamEvent

from ...core.execution_loops import (
    sync_execution_loop, sync_streaming_execution_loop,
    async_execution_loop, async_streaming_execution_loop
)
from ...core.utils import filter_toolflow_params
from .handler import AnthropicHandler

# --- Synchronous Wrappers ---

class AnthropicWrapper:
    """Wrapped Anthropic client that transparently adds toolflow capabilities."""
    def __init__(self, client: Anthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.messages = MessagesWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

class MessagesWrapper:
    def __init__(self, client: Anthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.messages.create
        self.handler = AnthropicHandler(client, client.messages.create)

    @overload
    def create(self, *, stream=False, **kwargs: Any) -> Message: ...
    @overload
    def create(self, *, stream=True, **kwargs: Any) -> Iterable[RawMessageStreamEvent]: ...

    def create(self, **kwargs: Any) -> Any:
        # merge full_response with kwargs, but allow method-level override
        if "full_response" not in kwargs:
            kwargs["full_response"] = self.full_response
        if kwargs.get("stream", False):
            if kwargs.get("response_format", None):
                raise ValueError("response_format is not supported for streaming")
            return sync_streaming_execution_loop(handler=self.handler, **kwargs)
        else:
            return sync_execution_loop(handler=self.handler, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

# --- Asynchronous Wrappers ---

class AsyncAnthropicWrapper:
    """Wrapped AsyncAnthropic client that transparently adds toolflow capabilities."""
    def __init__(self, client: AsyncAnthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.messages = AsyncMessagesWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

class AsyncMessagesWrapper:
    def __init__(self, client: AsyncAnthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.handler = AnthropicHandler(client, client.messages.create)

    @overload
    async def create(self, *, stream=False, **kwargs: Any) -> Message: ...
    @overload
    async def create(self, *, stream=True, **kwargs: Any) -> AsyncIterable[RawMessageStreamEvent]: ...

    async def create(self, **kwargs: Any) -> Any:
        # merge full_response with kwargs, but allow method-level override
        if "full_response" not in kwargs:
            kwargs["full_response"] = self.full_response
        if kwargs.get("stream", False):
            if kwargs.get("response_format", None):
                raise ValueError("response_format is not supported for streaming")
            return async_streaming_execution_loop(handler=self.handler, **kwargs)
        else:
            return await async_execution_loop(handler=self.handler, **kwargs) 
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
