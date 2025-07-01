# src/toolflow/providers/openai/wrappers.py

from __future__ import annotations

from typing import Any, List, Dict, overload, Iterable, AsyncIterable, Optional, Union
from typing_extensions import Literal

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# Use OpenAI's NOT_GIVEN directly (available in our minimum supported version 1.56.0+)
from openai._types import NOT_GIVEN, NotGiven

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
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        # Standard OpenAI parameters
        frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        function_call: Union[Dict[str, Any], NotGiven] = NOT_GIVEN,
        functions: Union[Iterable[Dict[str, Any]], NotGiven] = NOT_GIVEN,
        logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
        logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        metadata: Union[Optional[Dict[str, Any]], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
        presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        reasoning_effort: Union[Optional[str], NotGiven] = NOT_GIVEN,
        response_format: Union[Dict[str, Any], NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        service_tier: Union[Optional[Literal["auto", "default", "flex", "scale", "priority"]], NotGiven] = NOT_GIVEN,
        stop: Union[Union[Optional[str], List[str], None], NotGiven] = NOT_GIVEN,
        store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        stream: Union[Optional[Literal[False]], NotGiven] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        tool_choice: Union[Any, NotGiven] = NOT_GIVEN,
        tools: Union[Iterable[Dict[str, Any]], NotGiven] = NOT_GIVEN,
        top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
        user: Union[str, NotGiven] = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> ChatCompletion | str:
        """
        Creates a model response for the given chat conversation. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            stream: Enable streaming response (false for this overload)
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call. (max 128 functions)
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

        Returns:
            ChatCompletion | str: The model's response
        """
        ...
    
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        # Standard OpenAI parameters  
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: Dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[str] | NotGiven = NOT_GIVEN,
        response_format: Dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> Iterable[ChatCompletionChunk] | Iterable[str]:
        """
        Creates a streaming model response for the given chat conversation. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            stream: Enable streaming response (true for this overload)
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call. (max 128 functions)
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            Iterable[ChatCompletionChunk] | Iterable[str]: Stream of response chunks
        """
        ...
    
    @overload
    def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: bool,
        # Standard OpenAI parameters  
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: Dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[str] | NotGiven = NOT_GIVEN,
        response_format: Dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> ChatCompletion | str | Iterable[ChatCompletionChunk] | Iterable[str]:
        """
        Creates a model response for the given chat conversation with dynamic streaming. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call. (max 128 functions)
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            ChatCompletion | Iterable[ChatCompletionChunk]: Complete response or stream based on stream parameter
        """
        ...

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
        self.original_create = client.chat.completions.create
        self.handler = OpenAIHandler(client, client.chat.completions.create)

    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        # Standard OpenAI parameters
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: Dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[str] | NotGiven = NOT_GIVEN,
        response_format: Dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> ChatCompletion | str:
        """
        Asynchronously creates a model response for the given chat conversation. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            stream: Enable streaming response (false for this overload)
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call. (max 128 functions)
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            ChatCompletion | str: The model's response
        """
        ...
    
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        # Standard OpenAI parameters  
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: Dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[str] | NotGiven = NOT_GIVEN,
        response_format: Dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> AsyncIterable[ChatCompletionChunk] | AsyncIterable[str]:
        """
        Asynchronously creates a streaming model response for the given chat conversation. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            stream: Enable streaming response (true for this overload)
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-enhanced parameters
            tools: A list of python functions that the model may call. (max 128 functions)
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            AsyncIterable[ChatCompletionChunk] | AsyncIterable[str]: Async stream of response chunks
        """
        ...
    
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: bool,
        # Standard OpenAI parameters  
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: Dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[str] | NotGiven = NOT_GIVEN,
        response_format: Dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_calls: Optional[int] = None,
        parallel_tool_execution: bool = True,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> ChatCompletion | str | AsyncIterable[ChatCompletionChunk] | AsyncIterable[str]:
        """
        Asynchronously creates a model response for the given chat conversation with dynamic streaming. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            messages: A list of messages comprising the conversation so far
            model: Model ID to use for generation (e.g., 'gpt-4o', 'gpt-4o-mini')
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard OpenAI parameters
            frequency_penalty: Penalize tokens based on frequency (-2.0 to 2.0)
            function_call: [Deprecated] Controls function calling behavior
            functions: [Deprecated] List of functions the model may call
            logit_bias: Modify likelihood of specific tokens appearing
            logprobs: Whether to return log probabilities of output tokens
            max_completion_tokens: Upper bound for tokens generated (includes reasoning tokens)
            max_tokens: [Deprecated] Maximum tokens to generate
            metadata: Key-value pairs for storing additional information
            n: Number of chat completion choices to generate
            parallel_tool_calls: Enable parallel function calling during tool use
            presence_penalty: Penalize tokens based on presence (-2.0 to 2.0)
            reasoning_effort: Effort level for reasoning models ('low', 'medium', 'high')
            response_format: Format specification for model output
            seed: Random seed for deterministic sampling
            service_tier: Processing tier ('auto', 'default', 'flex', 'scale', 'priority')
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store completion for model distillation/evals
            temperature: Sampling temperature (0-2, higher = more random)
            tool_choice: Control which tools are called by the model
            top_logprobs: Number of most likely tokens to return (0-20)
            top_p: Nucleus sampling parameter (0-1)
            user: Stable identifier for end-users
            
            # Toolflow-enhanced parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            max_tool_calls: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full OpenAI response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

            
        Returns:
            ChatCompletion | str | AsyncIterable[ChatCompletionChunk] | AsyncIterable[str]: Complete response or async stream based on stream parameter
        """
        ...

    async def create(self, **kwargs: Any) -> Any:
        return await self._create_async(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
