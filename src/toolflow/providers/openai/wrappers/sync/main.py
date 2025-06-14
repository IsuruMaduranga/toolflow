"""
Main synchronous OpenAI wrapper classes.

This module contains the core synchronous wrapper classes for OpenAI clients.
"""
from typing import Any, Dict, List, Callable, Iterator, Union, Optional, Iterable, Literal

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

from ...tool_execution import (
    validate_and_prepare_openai_tools,
    execute_openai_tools_sync,
)
from ...streaming import (
    accumulate_openai_streaming_tool_calls,
    convert_accumulated_openai_tool_calls,
    format_openai_tool_calls_for_messages
)
from ...structured_output import (
    create_openai_response_tool,
    handle_openai_structured_response,
    validate_response_format
)

class OpenAIWrapper:
    """Wrapper around OpenAI client that supports tool-py functions."""
    
    def __init__(self, client, full_response: bool = False):
        from .beta import BetaWrapper
        self._client = client
        self._full_response = full_response
        self.chat = ChatWrapper(client, full_response)
        self.beta = BetaWrapper(client, full_response)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class ChatWrapper:
    """Wrapper around OpenAI chat that handles toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._full_response = full_response
        self.completions = CompletionsWrapper(client, full_response)

    def __getattr__(self, name):
        """Delegate all other attributes to the original chat."""
        return getattr(self._client.chat, name)


class CompletionsWrapper:
    """Wrapper around OpenAI completions that processes toolflow functions."""
    
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self._original_completions = client.chat.completions
        self._full_response = full_response

    def _extract_response_content(self, response, full_response: bool, is_structured: bool = False):
        """Extract content from response based on full_response flag."""
        if full_response:
            return response
        
        if is_structured:
            return response.choices[0].message.parsed
        
        return response.choices[0].message.content

    def create(
        self,
        *,
        # Required parameters
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        
        # OpenAI API parameters (in alphabetical order for consistency)
        audio: Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
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
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        web_search_options: completion_create_params.WebSearchOptions | NotGiven = NOT_GIVEN,
        
        # Toolflow-specific parameters
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        graceful_error_handling: bool = True,
        full_response: Optional[bool] = None,
    ) -> Union[Any, Iterator[Any]]:
        """
        Create a chat completion with tool support.
        
        This method supports all OpenAI chat completion parameters plus toolflow-specific enhancements.
        
        Args:
            messages: A list of messages comprising the conversation so far
            model: ID of the model to use
            
            # OpenAI Parameters (see OpenAI documentation for details)
            audio: Parameters for audio output
            frequency_penalty: Number between -2.0 and 2.0 for frequency penalty
            function_call: Deprecated in favor of tool_choice
            functions: Deprecated in favor of tools
            logit_bias: Modify likelihood of specified tokens
            logprobs: Whether to return log probabilities
            max_completion_tokens: Upper bound for tokens generated
            max_tokens: Maximum number of tokens (deprecated, use max_completion_tokens)
            metadata: Set of key-value pairs for additional information
            modalities: Output types the model should generate
            n: Number of chat completion choices to generate
            parallel_tool_calls: Whether to enable parallel function calling
            prediction: Static predicted output content
            presence_penalty: Number between -2.0 and 2.0 for presence penalty
            reasoning_effort: Constrains effort on reasoning (o-series models only)
            response_format: Format that the model must output
            seed: For deterministic sampling
            service_tier: Latency tier for processing
            stop: Up to 4 sequences where API will stop generating
            store: Whether to store output for model distillation/evals
            stream: Whether to stream the response
            stream_options: Options for streaming response
            temperature: Sampling temperature between 0 and 2
            tool_choice: Controls which tool is called by the model
            tools: List of tools the model may call
            top_logprobs: Number of most likely tokens to return
            top_p: Alternative to sampling with temperature
            user: Unique identifier for end-user
            web_search_options: Web search tool options
            
            # Toolflow Parameters
            parallel_tool_execution: Whether to execute multiple tool calls in parallel
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of worker threads for parallel execution
            graceful_error_handling: Whether to handle tool execution errors gracefully
            full_response: Override client-level full_response setting
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results, or Iterator if stream=True
        """
        # Build kwargs dict for OpenAI API call, excluding toolflow-specific params
        openai_kwargs = {}
        
        # Add all OpenAI parameters that are not NOT_GIVEN
        if audio is not NOT_GIVEN:
            openai_kwargs['audio'] = audio
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
        if response_format is not NOT_GIVEN:
            openai_kwargs['response_format'] = response_format
        if seed is not NOT_GIVEN:
            openai_kwargs['seed'] = seed
        if service_tier is not NOT_GIVEN:
            openai_kwargs['service_tier'] = service_tier
        if stop is not NOT_GIVEN:
            openai_kwargs['stop'] = stop
        if store is not NOT_GIVEN:
            openai_kwargs['store'] = store
        if stream is not NOT_GIVEN:
            openai_kwargs['stream'] = stream
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

        # Use method-level full_response if provided, otherwise use client-level setting
        effective_full_response = full_response if full_response is not None else self._full_response

        # Handle toolflow response_format (structured output)
        toolflow_tools = None
        if tools is not NOT_GIVEN:
            toolflow_tools = list(tools)
            
        if response_format is not NOT_GIVEN and response_format is not None:
            if stream is not NOT_GIVEN and stream:
                raise ValueError("response_format is not supported for streaming")
            
            validate_response_format(response_format)
            # Create a dynamic response tool
            response_tool = create_openai_response_tool(response_format)

            toolflow_tools = toolflow_tools or []
            toolflow_tools.append(response_tool)

        # Handle streaming
        if stream is not NOT_GIVEN and stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=toolflow_tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling,
                full_response=effective_full_response,
                **openai_kwargs
            )
        
        if toolflow_tools is None:
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **openai_kwargs
            )
            return self._extract_response_content(response, effective_full_response)
        
        # If tools are provided, handle tool execution
        tool_call_count = 0
        tool_functions, tool_schemas = validate_and_prepare_openai_tools(toolflow_tools)
        current_messages = list(messages)
        
        # Tool execution loop
        while tool_call_count < max_tool_calls:
            # Make the API call
            response = self._original_completions.create(
                model=model,
                messages=current_messages,
                tools=tool_schemas,
                **openai_kwargs
            )

            if response.choices[0].finish_reason == "length":
                raise Exception("Max tokens reached without finding a solution")

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                # No tool calls, we're done
                return self._extract_response_content(response, effective_full_response)
            
            # Handle structured response
            structured_response = handle_openai_structured_response(response, response_format)
            if structured_response:
                return self._extract_response_content(structured_response, effective_full_response, is_structured=True)
            
            # Else we execute rest of the tools
            current_messages.append(response.choices[0].message)
            tool_results = execute_openai_tools_sync(
                tool_functions,
                tool_calls,
                parallel_tool_execution, 
                max_workers=max_workers,
                graceful_error_handling=graceful_error_handling
            )
            tool_call_count += len(tool_results)  
            current_messages.extend(tool_results)

        raise Exception("Max tool calls reached without finding a solution")

    def _create_streaming(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 10,
        max_workers: int = 10,
        graceful_error_handling: bool = True,
        full_response: bool = None,
        **kwargs
    ) -> Iterator[Any]:
        """
        Create a streaming chat completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
        if full_response is None:
            full_response = self._full_response
            
        def streaming_generator():
            current_messages = list(messages)
            remaining_tool_calls = max_tool_calls
            
            while True:
                if remaining_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")
                
                if tools:
                    tool_functions, tool_schemas = validate_and_prepare_openai_tools(tools)
                    
                    # Make streaming API call with tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        **kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        **kwargs
                    )
                
                # Accumulate the streamed response and detect tool calls
                accumulated_content = ""
                accumulated_tool_calls = []
                message_dict = {"role": "assistant", "content": ""}
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        # Yield based on full_response flag
                        if full_response:
                            yield chunk
                        else:
                            # Only yield content if available
                            if delta.content:
                                yield delta.content
                        
                        # Accumulate content
                        if delta.content:
                            accumulated_content += delta.content
                            message_dict["content"] += delta.content
                        
                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                accumulate_openai_streaming_tool_calls(accumulated_tool_calls, tool_call_delta)
                
                # Check if we have tool calls to execute
                if accumulated_tool_calls and tools:
                    tool_calls = convert_accumulated_openai_tool_calls(accumulated_tool_calls)
                    
                    if tool_calls:
                        # Add assistant message with tool calls
                        message_dict["tool_calls"] = format_openai_tool_calls_for_messages(tool_calls)
                        current_messages.append(message_dict)
                        
                        # Execute tools
                        execution_response = execute_openai_tools_sync(
                            tool_functions,
                            tool_calls,
                            parallel_tool_execution,
                            max_workers=max_workers,
                            graceful_error_handling=graceful_error_handling
                        )
                        
                        remaining_tool_calls -= len(tool_calls)
                        current_messages.extend(execution_response)
                        
                        # Continue the loop to get the next response
                        continue
                
                # No tool calls, we're done
                break
        
        return streaming_generator()

    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)
