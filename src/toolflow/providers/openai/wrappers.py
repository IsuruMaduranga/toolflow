# src/toolflow/providers/openai/wrappers.py

from typing import Any, List, Dict, overload, Iterable, AsyncIterable
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from toolflow.core import CreateMixin
from .handler import OpenAIHandler

# --- Synchronous Wrappers ---

class OpenAIWrapper:
    """Wrapped OpenAI client that transparently adds toolflow capabilities."""
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.chat = ChatWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

class ChatWrapper:
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.completions = CompletionsWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat, name)

class CompletionsWrapper(CreateMixin):
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.chat.completions.create
        self.handler = OpenAIHandler(client, client.chat.completions.create)

    @overload
    def create(self, *, stream=False, **kwargs: Any) -> ChatCompletion: ...
    @overload
    def create(self, *, stream=True, **kwargs: Any) -> Iterable[ChatCompletionChunk]: ...

    def create(self, **kwargs: Any) -> Any:
        return self._create_sync(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

# --- Asynchronous Wrappers ---

class AsyncOpenAIWrapper:
    """Wrapped AsyncOpenAI client that transparently adds toolflow capabilities."""
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.chat = AsyncChatWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

class AsyncChatWrapper:
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.completions = AsyncCompletionsWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat, name)

class AsyncCompletionsWrapper(CreateMixin):
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.handler = OpenAIHandler(client, client.chat.completions.create)

    @overload
    async def create(self, *, stream=False, **kwargs: Any) -> ChatCompletion: ...
    @overload
    async def create(self, *, stream=True, **kwargs: Any) -> AsyncIterable[ChatCompletionChunk]: ...

    async def create(self, **kwargs: Any) -> Any:
        return await self._create_async(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
