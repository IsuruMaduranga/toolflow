# src/toolflow/providers/openai/wrappers.py

import textwrap
from functools import wraps
from typing import Any, List, Dict, overload, Iterable, AsyncIterable
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ...core.execution_loops import (
    sync_execution_loop, sync_streaming_execution_loop,
    async_execution_loop, async_streaming_execution_loop
)
from ...core.utils import filter_toolflow_params
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

class CompletionsWrapper:
    def __init__(self, client: OpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.chat.completions.create
        self._append_toolflow_docs()

    # def _append_toolflow_docs(self):
    #     original_doc = getattr(self.create, "__doc__", "") or getattr(self.original_create, "__doc__", "")
    #     toolflow_doc_appendix = """
    #     --- Toolflow Additions ---
    #     This method is wrapped by `toolflow` to provide enhanced tool-calling and structured output capabilities.
    #     Additional Parameters: `max_tool_calls`, `parallel_tool_execution`, `max_workers`, `response_format`, `full_response`.
    #     """
    #     # self.create.__doc__ = f"{textwrap.dedent(original_doc or '')}\n{textwrap.dedent(toolflow_doc_appendix)}"

    @overload
    def create(self, *, stream=False, **kwargs: Any) -> ChatCompletion: ...
    @overload
    def create(self, *, stream=True, **kwargs: Any) -> Iterable[ChatCompletionChunk]: ...

    def create(self, **kwargs: Any) -> Any:
        messages: List[Dict] = kwargs.get("messages", [])
        tools: List[Any] = kwargs.get("tools", [])
        stream: bool = kwargs.get("stream", False)
        
        # Toolflow specific params
        max_tool_calls = kwargs.get("max_tool_calls", 5)
        parallel_tool_execution = kwargs.get("parallel_tool_execution", True)
        max_workers = kwargs.get("max_workers", None)
        response_format = kwargs.get("response_format", None)
        
        filtered_kwargs = filter_toolflow_params(kwargs)

        if response_format and not (isinstance(response_format, type) and hasattr(response_format, 'model_json_schema')):
            raise ValueError("response_format must be a Pydantic model.")
        
        if response_format:
            tools = [response_format]
            filtered_kwargs["tool_choice"] = "required"
        
        handler = OpenAIHandler(self._client, self.original_create)

        if stream:
            return sync_streaming_execution_loop(
                handler=handler,
                messages=messages,
                tools=tools,
                **filtered_kwargs,
            )
        else:
            return sync_execution_loop(
                handler=handler,
                messages=messages,
                tools=tools,
                max_tool_calls=max_tool_calls,
                parallel_tool_execution=parallel_tool_execution,
                max_workers=max_workers,
                **filtered_kwargs,
            )

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

class AsyncCompletionsWrapper:
    def __init__(self, client: AsyncOpenAI, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.handler = OpenAIHandler(client, client.chat.completions.create)
        #self._append_toolflow_docs()

    # def _append_toolflow_docs(self):
    #     original_doc = getattr(self.create, "__doc__", "") or getattr(self.original_create, "__doc__", "")
    #     toolflow_doc_appendix = """
    #     --- Toolflow Additions ---
    #     This method is wrapped by `toolflow` to provide enhanced tool-calling and structured output capabilities.
    #     Additional Parameters: `max_tool_calls`, `parallel_tool_execution`, `max_workers`, `response_format`, `full_response`.
    #     """
    #     self.create.__doc__ = f"{textwrap.dedent(original_doc or '')}\n{textwrap.dedent(toolflow_doc_appendix)}"

    @overload
    async def create(self, *, stream=False, **kwargs: Any) -> ChatCompletion: ...
    @overload
    async def create(self, *, stream=True, **kwargs: Any) -> AsyncIterable[ChatCompletionChunk]: ...

    async def create(self, **kwargs: Any) -> Any:
        # merge full_response with kwargs
        kwargs["full_response"] = self.full_response
        if kwargs.get("stream", False):
            return await async_streaming_execution_loop(handler=self.handler, **kwargs)
        else:
            return await async_execution_loop(handler=self.handler, **kwargs)
