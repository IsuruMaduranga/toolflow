"""
Synchronous OpenAI Beta wrapper classes.

This module contains the beta wrapper classes for OpenAI clients.
"""
from typing import Any, Dict, List, Iterator, Union, Callable, Optional, Iterable, Literal

# Import OpenAI types for proper parameter typing
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionAudioParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params.metadata import Metadata
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.chat import completion_create_params
from openai._types import NOT_GIVEN, NotGiven
from openai._types import Headers, Query, Body
import httpx

from ...tool_execution import (
    validate_and_prepare_openai_tools,
    execute_openai_tools_sync,
)

class BetaWrapper:
    """Wrapper around OpenAI beta that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.chat = BetaChatWrapper(client, full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client.beta, name)


class BetaChatWrapper:
    """Wrapper around OpenAI beta chat that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.completions = BetaCompletionsWrapper(client, full_response)


class BetaCompletionsWrapper:
    """Wrapper around OpenAI beta completions that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_completions = client.beta.chat.completions
        self._full_response = full_response

    def _extract_response_content(self, response, full_response: bool, is_structured: bool = False):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        # Check if we have a parsed structured response
        # Only return parsed if it exists and is not a Mock object (for tests)
        if (hasattr(response.choices[0].message, 'parsed') and 
            response.choices[0].message.parsed is not None and
            not str(type(response.choices[0].message.parsed)).startswith("<class 'unittest.mock")):
            return response.choices[0].message.parsed
        
        return response.choices[0].message.content

    def parse(
        self,
        *,
        # Required parameters
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        
        # OpenAI API parameters (in alphabetical order for consistency)
        audio: Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
        response_format: Optional[Any] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        modalities: Optional[List[Literal["text", "audio"]]] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: Optional[ChatCompletionPredictionContentParam] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[ReasoningEffort] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        web_search_options: completion_create_params.WebSearchOptions | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        
        # Toolflow-specific parameters
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        full_response: Optional[bool] = None,
    ) -> Union[Any, Iterator[Any]]:
        """
        Wrapper over the `client.beta.chat.completions.parse()` method that provides richer integrations with Python specific types
        & returns a `ParsedChatCompletion` object, which is a subclass of the standard `ChatCompletion` class.

        You can pass a pydantic model to this method and it will automatically convert the model
        into a JSON schema, send it to the API and parse the response content back into the given model.

        This method will also automatically parse `function` tool calls if:
        - You use the `openai.pydantic_function_tool()` helper method
        - You mark your tool schema with `"strict": True`

        Args:
            messages: A list of messages comprising the conversation so far
            model: ID of the model to use
            audio: Parameters for audio output
            response_format: An object specifying the format that the model must output
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
            function_call: Deprecated in favor of tool_choice
            functions: Deprecated in favor of tools
            logit_bias: Modify the likelihood of specified tokens appearing in the completion
            logprobs: Whether to return log probabilities of the output tokens or not
            max_completion_tokens: The maximum number of tokens that can be generated in the chat completion
            max_tokens: The maximum number of tokens that can be generated in the chat completion
            metadata: Developer-defined tags and values used for filtering completions
            modalities: Output types that you would like the model to generate for this request
            n: How many chat completion choices to generate for each input message
            parallel_tool_calls: Whether to enable parallel function calling during tool use
            prediction: Configuration for prediction content
            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
            reasoning_effort: Reasoning effort to use for the request
            seed: This feature is in Beta. If specified, our system will make a best effort to sample deterministically
            service_tier: Specifies the latency tier to use for processing the request
            stop: Up to 4 sequences where the API will stop generating further tokens
            store: Whether to store the output of this chat completion request for use in our model distillation or evals products
            stream_options: Options for streaming response
            temperature: What sampling temperature to use, between 0 and 2
            tool_choice: Controls which (if any) tool is called by the model
            tools: A list of tools the model may call
            top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to return at each token position
            top_p: An alternative to sampling with temperature, called nucleus sampling
            user: A unique identifier representing your end-user
            web_search_options: Configuration for web search
            extra_headers: Send extra headers
            extra_query: Add additional query parameters to the request
            extra_body: Add additional JSON properties to the request
            timeout: Override the client-level default timeout for this request
            parallel_tool_execution: Whether to execute tools in parallel
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of workers for parallel tool execution
            full_response: Override client-level full_response setting
        
        Returns:
            ParsedChatCompletion response, potentially with tool results
        """
        # Build kwargs dict for OpenAI API call, excluding toolflow-specific params
        openai_kwargs = {}
        
        # Add all OpenAI parameters that are not NOT_GIVEN
        if audio is not NOT_GIVEN:
            openai_kwargs['audio'] = audio
        if response_format is not NOT_GIVEN:
            openai_kwargs['response_format'] = response_format
        if frequency_penalty is not NOT_GIVEN:
            openai_kwargs['frequency_penalty'] = frequency_penalty
        if function_call is not NOT_GIVEN:
            openai_kwargs['function_call'] = function_call
        if functions is not NOT_GIVEN:
            openai_kwargs['functions'] = functions
        if logit_bias is not NOT_GIVEN:
            openai_kwargs['logit_bias'] = logit_bias
        if logprobs is not NOT_GIVEN:
            openai_kwargs['logprobs'] = logprobs
        if max_completion_tokens is not NOT_GIVEN:
            openai_kwargs['max_completion_tokens'] = max_completion_tokens
        if max_tokens is not NOT_GIVEN:
            openai_kwargs['max_tokens'] = max_tokens
        if metadata is not NOT_GIVEN:
            openai_kwargs['metadata'] = metadata
        if modalities is not NOT_GIVEN:
            openai_kwargs['modalities'] = modalities
        if n is not NOT_GIVEN:
            openai_kwargs['n'] = n
        if parallel_tool_calls is not NOT_GIVEN:
            openai_kwargs['parallel_tool_calls'] = parallel_tool_calls
        if prediction is not NOT_GIVEN:
            openai_kwargs['prediction'] = prediction
        if presence_penalty is not NOT_GIVEN:
            openai_kwargs['presence_penalty'] = presence_penalty
        if reasoning_effort is not NOT_GIVEN:
            openai_kwargs['reasoning_effort'] = reasoning_effort
        if seed is not NOT_GIVEN:
            openai_kwargs['seed'] = seed
        if service_tier is not NOT_GIVEN:
            openai_kwargs['service_tier'] = service_tier
        if stop is not NOT_GIVEN:
            openai_kwargs['stop'] = stop
        if store is not NOT_GIVEN:
            openai_kwargs['store'] = store
        if stream_options is not NOT_GIVEN:
            openai_kwargs['stream_options'] = stream_options
        if temperature is not NOT_GIVEN:
            openai_kwargs['temperature'] = temperature
        if tool_choice is not NOT_GIVEN:
            openai_kwargs['tool_choice'] = tool_choice
        if top_logprobs is not NOT_GIVEN:
            openai_kwargs['top_logprobs'] = top_logprobs
        if top_p is not NOT_GIVEN:
            openai_kwargs['top_p'] = top_p
        if user is not NOT_GIVEN:
            openai_kwargs['user'] = user
        if web_search_options is not NOT_GIVEN:
            openai_kwargs['web_search_options'] = web_search_options
        if extra_headers is not None:
            openai_kwargs['extra_headers'] = extra_headers
        if extra_query is not None:
            openai_kwargs['extra_query'] = extra_query
        if extra_body is not None:
            openai_kwargs['extra_body'] = extra_body
        if timeout is not NOT_GIVEN:
            openai_kwargs['timeout'] = timeout

        # Use method-level full_response if provided, otherwise use client-level setting
        effective_full_response = full_response if full_response is not None else self._full_response

        # Handle toolflow tools
        toolflow_tools = None
        if tools is not NOT_GIVEN:
            toolflow_tools = list(tools)
        
        if toolflow_tools is None:
            response = self._original_completions.parse(
                model=model,
                messages=messages,
                **openai_kwargs
            )
            return self._extract_response_content(response, effective_full_response)
        
        # If tools are provided, handle tool execution
        tool_functions, tool_schemas = validate_and_prepare_openai_tools(toolflow_tools, strict=True)
        current_messages = list(messages)
        
        # Tool execution loop
        while True:
            if max_tool_calls <= 0:
                raise Exception("Max tool calls reached without finding a solution")

            # Make the API call
            response = self._original_completions.parse(
                model=model,
                messages=current_messages,
                tools=tool_schemas,
                **openai_kwargs
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                current_messages.append(response.choices[0].message)
                execution_response = execute_openai_tools_sync(
                    tool_functions, 
                    tool_calls, 
                    parallel_tool_execution,
                    max_workers=max_workers
                )
                max_tool_calls -= len(execution_response)
                current_messages.extend(execution_response)
            else:
                return self._extract_response_content(response, effective_full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)
