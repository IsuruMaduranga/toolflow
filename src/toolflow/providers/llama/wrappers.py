# src/toolflow/providers/llama/wrappers.py

from __future__ import annotations

from typing import Any, List, Dict, overload, Iterable, AsyncIterable, Optional, Union, TypeVar
from typing_extensions import Literal
try:
    from collections.abc import AsyncContextManager
except ImportError:
    # Fallback for Python < 3.11
    from typing import AsyncContextManager

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

# Use OpenAI's NOT_GIVEN directly (available in our minimum supported version 1.56.0+)
from openai._types import NOT_GIVEN, NotGiven

from toolflow.core import ExecutorMixin
from .handler import LlamaHandler

# Type variable for response format in parse method
ResponseFormatT = TypeVar('ResponseFormatT')

# --- Synchronous Wrappers ---

class LlamaWrapper:
    """Wrapped OpenAI client for Llama models that transparently adds toolflow capabilities."""
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.chat = ChatWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client) + super().__dir__()))
    
    @property
    def raw(self) -> OpenAI:
        """Access the underlying OpenAI client for debugging or advanced use."""
        return self._client
    
    def unwrap(self) -> OpenAI:
        return self._client
    
    def __enter__(self) -> LlamaWrapper:
        self._client.__enter__()
        return self
    
    def __exit__(self, *args):
        return self._client.__exit__(*args)

class ChatWrapper:
    """Wrapper for chat.completions that adds toolflow execution capabilities."""
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.completions = CompletionsWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat, name)

class CompletionsWrapper(ExecutorMixin):
    """Wrapper for chat.completions.create that handles tool execution and structured outputs."""
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.chat.completions.create
        self.handler = LlamaHandler(client, client.chat.completions.create)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat.completions, name)

    # Basic create method (no structured outputs, no tools)
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> ChatCompletion: ...

    # Create with tools (no structured outputs)
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> ChatCompletion: ...

    # Create with structured outputs (no tools)
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        response_format: ResponseFormatT,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> ParsedChatCompletion[ResponseFormatT]: ...

    # Create with both tools and structured outputs
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        response_format: ResponseFormatT,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> ParsedChatCompletion[ResponseFormatT]: ...

    # Streaming without tools or structured outputs
    @overload 
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[True],
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> Iterable[ChatCompletionChunk]: ...

    # Streaming with tools
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[True],
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> Iterable[ChatCompletionChunk]: ...

    # Main implementation method
    def create(self, **kwargs) -> Any:
        """Create a chat completion with enhanced toolflow capabilities for Llama models."""
        return self._create_sync(**kwargs)

# --- Async Wrappers ---

class AsyncLlamaWrapper:
    """Wrapped AsyncOpenAI client for Llama models that transparently adds toolflow capabilities."""
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.chat = AsyncChatWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client) + super().__dir__()))
    
    @property
    def raw(self) -> AsyncOpenAI:
        """Access the underlying AsyncOpenAI client for debugging or advanced use."""
        return self._client
    
    def unwrap(self) -> AsyncOpenAI:
        return self._client
    
    async def __aenter__(self) -> AsyncLlamaWrapper:
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        return await self._client.__aexit__(*args)

class AsyncChatWrapper:
    """Async wrapper for chat.completions that adds toolflow execution capabilities."""
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.completions = AsyncCompletionsWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat, name)

class AsyncCompletionsWrapper(ExecutorMixin):
    """Async wrapper for chat.completions.create that handles tool execution and structured outputs."""
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.chat.completions.create
        self.handler = LlamaHandler(client, client.chat.completions.create)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.chat.completions, name)

    # Basic async create method (no structured outputs, no tools)
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> ChatCompletion: ...

    # Async create with tools (no structured outputs)
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> ChatCompletion: ...

    # Async create with structured outputs (no tools)
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        response_format: ResponseFormatT,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> ParsedChatCompletion[ResponseFormatT]: ...

    # Async create with both tools and structured outputs
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        response_format: ResponseFormatT,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[False] = False,
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> ParsedChatCompletion[ResponseFormatT]: ...

    # Async streaming without tools or structured outputs
    @overload 
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[True],
        temperature: Optional[float] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
    ) -> AsyncIterable[ChatCompletionChunk]: ...

    # Async streaming with tools
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Iterable[Any],
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[bool] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] = NOT_GIVEN,
        stream: Literal[True],
        temperature: Optional[float] = NOT_GIVEN,
        tool_choice: Optional[str] = NOT_GIVEN,
        top_logprobs: Optional[int] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Union[float, Any] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: int = 10,
        parallel_tool_calls: bool = True,
    ) -> AsyncIterable[ChatCompletionChunk]: ...

    # Main async implementation method
    async def create(self, **kwargs) -> Any:
        """Create a chat completion with enhanced toolflow capabilities for Llama models."""
        return await self._create_async(**kwargs)

__all__ = ['LlamaWrapper', 'AsyncLlamaWrapper']
