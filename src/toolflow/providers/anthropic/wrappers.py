from __future__ import annotations

from typing import Any, List, Dict, overload, Iterable, AsyncIterable, Optional
from typing_extensions import Literal
from typing import AsyncContextManager

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageStreamEvent

# Use Anthropic's NOT_GIVEN directly since it's available in minimum supported versions
from anthropic._types import NOT_GIVEN, NotGiven

from toolflow.core import ExecutorMixin
from toolflow.core import filter_toolflow_params
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
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client) + super().__dir__()))
    
    @property
    def raw(self) -> Anthropic:
        """Access the underlying Anthropic client for debugging or advanced use."""
        return self._client
    
    def unwrap(self) -> Anthropic:
        return self._client
    
    def __enter__(self) -> AnthropicWrapper:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.__exit__(exc_type, exc_val, exc_tb)
    
    def __repr__(self) -> str:
        return (
            f"<AnthropicWrapper client={self._client.__class__.__name__} "
            f"full_response={self.full_response}>"
        )

    def __str__(self) -> str:
        return f"Toolflow Anthropic client [sync, full_response={self.full_response}]"

class MessagesWrapper(ExecutorMixin):
    def __init__(self, client: Anthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.messages.create
        self.handler = AnthropicHandler(client, client.messages.create)

    @overload
    def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        stream: Literal[False] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Message | str:
        """
        Creates a message response from Anthropic's Claude. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            system: System prompt to guide model behavior
            temperature: Sampling temperature (0-1, higher = more random)
            thinking: Enable extended thinking mode for deeper reasoning
            tool_choice: Control which tools are called by the model
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)

            # Toolflow-specific parameters 
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular pydantic models.
                Toolflow will handle the rest.
            full_response: Return full Anthropic response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

            Returns:
                Message | str
        """
        ...

    @overload
    def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Iterable[RawMessageStreamEvent]:
        """
        Creates a streaming message response from Anthropic's Claude. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard Anthropic parameters
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            system: System prompt to guide model behavior
            temperature: Sampling temperature (0-1, higher = more random)
            thinking: Enable extended thinking mode for deeper reasoning
            tool_choice: Control which tools are called by the model
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)

            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular pydantic models.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Anthropic response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

            # Extra parameters
            extra_headers: Send extra headers
            extra_query: Add additional query parameters to the request
            extra_body: Add additional JSON properties to the request
            timeout: Override the client-level default timeout for this request, in seconds

            Returns:
                Iterable[RawMessageStreamEvent]
        """
        ...

    @overload
    def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: bool,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Message | Iterable[RawMessageStreamEvent]:
        """
        Creates a message response from Anthropic's Claude with dynamic streaming. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard Anthropic parameters
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            system: System prompt to guide model behavior
            temperature: Sampling temperature (0-1, higher = more random)
            thinking: Enable extended thinking mode for deeper reasoning
            tool_choice: Control which tools are called by the model
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)

            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular pydantic models.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Anthropic response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

            # Extra parameters
            extra_headers: Send extra headers
            extra_query: Add additional query parameters to the request
            extra_body: Add additional JSON properties to the request
            timeout: Override the client-level default timeout for this request, in seconds

            Returns:
                Message | Iterable[RawMessageStreamEvent]
        """
        ...

    def create(self, **kwargs: Any) -> Any:
        return self._create_sync(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client.messages) + super().__dir__()))

# --- Asynchronous Wrappers ---

class AsyncAnthropicWrapper(AsyncContextManager):
    """Wrapped AsyncAnthropic client that transparently adds toolflow capabilities."""
    def __init__(self, client: AsyncAnthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.messages = AsyncMessagesWrapper(client, full_response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client) + super().__dir__()))
    
    @property
    def raw(self) -> AsyncAnthropic:
        """Access the underlying AsyncAnthropic client for debugging or advanced use."""
        return self._client
    
    def unwrap(self) -> AsyncAnthropic:
        return self._client
    
    async def __aenter__(self) -> AsyncAnthropicWrapper:
        """Support async context management."""
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Support async context management."""
        return await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    def __repr__(self) -> str:
        return (
            f"<AsyncAnthropicWrapper client={self._client.__class__.__name__} "
            f"full_response={self.full_response}>"
        )
    
    def __str__(self) -> str:
        return f"Toolflow Anthropic client [async, full_response={self.full_response}]"

class AsyncMessagesWrapper(ExecutorMixin):
    def __init__(self, client: AsyncAnthropic, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_create = client.messages.create
        self.handler = AnthropicHandler(client, client.messages.create)

    @overload
    async def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        stream: Literal[False] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Message | str:
        """
        Asynchronously creates a message response from Anthropic's Claude. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            
            # Standard Anthropic parameters
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            stream: Enable streaming response (false for this overload)
            system: System prompt to guide model behavior
            thinking: Enable extended thinking mode for deeper reasoning
            temperature: Sampling temperature (0-1, higher = more random)
            tool_choice: Control which tools are called by the model
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular pydantic models.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 50)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Anthropic response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
            # Extra parameters
            extra_headers: Send extra headers
            extra_query: Add additional query parameters to the request
            extra_body: Add additional JSON properties to the request
            timeout: Override the client-level default timeout for this request, in seconds

            Returns:
                Message | str
        """
        ...
    
    @overload
    async def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterable[RawMessageStreamEvent]:
        """
        Asynchronously creates a streaming message response from Anthropic's Claude. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            stream: Enable streaming response (true for this overload)
            
            # Standard Anthropic parameters
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            system: System prompt to guide model behavior
            temperature: Sampling temperature (0-1, higher = more random)
            thinking: Enable extended thinking mode for deeper reasoning
            tool_choice: Control which tools are called by the model
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your regular pydantic models.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 50)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Anthropic response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            AsyncIterable[RawMessageStreamEvent]: Async stream of response events
        """
        ...
    
    @overload
    async def create(
        self,
        *,
        max_tokens: int,
        messages: Iterable[Dict[str, Any]],
        model: str,
        stream: bool,
        metadata: Optional[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        stop_sequences: Optional[List[str]] | NotGiven = NOT_GIVEN,
        system: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        tools: Iterable[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        top_k: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        thinking: Optional[bool] = None,
        graceful_error_handling: bool = True,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Message | str | AsyncIterable[RawMessageStreamEvent]:
        """
        Asynchronously creates a message response from Anthropic's Claude with dynamic streaming. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            max_tokens: Maximum number of tokens to generate
            messages: List of message objects comprising the conversation
            model: Anthropic model to use (e.g., 'claude-3-5-sonnet-20241022')
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard Anthropic parameters
            metadata: Key-value pairs for storing additional information
            stop_sequences: Custom text sequences that cause model to stop
            system: System prompt to guide model behavior
            temperature: Sampling temperature (0-1, higher = more random)
            tool_choice: Control which tools are called by the model
            tools: List of tools the model may call
            top_k: Only sample from top K options for each token
            top_p: Nucleus sampling parameter (0-1)
            
            # Toolflow-specific parameters
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 50)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            response_format: Format specification for model output
            full_response: Return full Anthropic response object instead of content only
            thinking: Enable extended thinking mode for deeper reasoning
            
        Returns:
            Message | str | AsyncIterable[RawMessageStreamEvent]: Complete response or async stream based on stream parameter
        """
        ...

    async def create(self, **kwargs: Any) -> Any:
        return await self._create_async(**kwargs) 
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
    
    def __dir__(self) -> List[str]:
        """Improve IDE autocompletion by including client attributes."""
        return list(set(dir(self._client.messages) + super().__dir__()))
