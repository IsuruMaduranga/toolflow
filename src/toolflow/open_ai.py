from typing import Any, Dict, List, Callable, Iterator, AsyncIterator, Union
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def from_openai(client) -> "OpenAIWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI client.
    
    Args:
        client: An existing OpenAI client instance
    
    Returns:
        OpenAIWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        client = toolflow.from_openai(openai.OpenAI())
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = client.chat.completions.create(
            model="gpt-4",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIWrapper(client)

def from_openai_async(client) -> "OpenAIAsyncWrapper":
    """
    Create a toolflow wrapper around an existing OpenAI async client.
    
    Args:
        client: An existing OpenAI async client instance
    
    Returns:
        OpenAIAsyncWrapper that supports tool-py decorated functions
    
    Example:
        import openai
        import toolflow
        
        client = toolflow.from_openai_async(openai.AsyncOpenAI())
        
        @toolflow.tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        response = await client.chat.completions.create(
            model="gpt-4",
            tools=[get_weather],
            messages=[{"role": "user", "content": "What's the weather in NYC?"}]
        )
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    return OpenAIAsyncWrapper(client)

class OpenAIWrapper:
    """Wrapper around OpenAI client that supports tool-py functions."""
    
    def __init__(self, client):
        self._client = client
        self.chat = ChatWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class ChatWrapper:
    """Wrapper around OpenAI chat that handles toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self.completions = CompletionsWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original chat."""
        return getattr(self._client.chat, name)


class CompletionsWrapper:
    """Wrapper around OpenAI completions that processes toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self._original_completions = client.chat.completions
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Union[Any, Iterator[Any]]:
        """
        Create a chat completion with tool support.
        
        Args:
            tools: List of toolflow decorated functions or OpenAI tool dicts
            parallel_tool_execution: Whether to execute multiple tool calls in parallel (default: False)
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of worker threads to use for parallel execution of sync tools
            stream: Whether to stream the response (default: False)
            **kwargs: All other OpenAI chat completion parameters
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results, or Iterator if stream=True
        """
        tools = kwargs.get('tools', None)
        parallel_tool_execution = kwargs.get('parallel_tool_execution', False)
        max_tool_calls = kwargs.get('max_tool_calls', 5)
        max_workers = kwargs.get('max_workers', 10)
        stream = kwargs.get('stream', False)
        
        # Handle streaming
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
            )
        
        response = None
        if tools:
            tool_functions = {}
            tool_schemas = []
            
            for tool in tools:
                if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
                    tool_schemas.append(tool._tool_metadata)
                    tool_functions[tool._tool_metadata['function']['name']] = tool
                else:
                    raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")

                # Make the API call
                response = self._original_completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
                )

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    messages.append(response.choices[0].message)
                    execution_response = self._execute_tools(
                        tool_functions, 
                        tool_calls, 
                        parallel_tool_execution,
                        max_workers=max_workers
                    )
                    max_tool_calls -= len(execution_response)
                    messages.extend(execution_response)
                else:
                    return response

        else: # No tools, just make the API call
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
            )
        
        return response

    def _execute_tools(
        self, 
        tool_functions: Dict[str, Callable],
        tool_calls: List[Dict[str, Any]],
        parallel_tool_execution: bool = False,
        max_workers: int = 10
    ):
        """Execute tool calls sequentially or in parallel based on the parallel_tool_execution flag."""
        
        def execute_single_tool(tool_call):
            """Execute a single tool call."""
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            tool_function = tool_functions.get(tool_name, None)
            if not tool_function:
                raise ValueError(f"Tool {tool_name} not found")
            
            try:
                # Parse JSON arguments and call the function directly
                parsed_args = json.loads(tool_args) if tool_args else {}
                result = tool_function(**parsed_args)
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result) if not isinstance(result, str) else result
                }
            except Exception as e:
                raise Exception(f"Error executing tool {tool_name}: {e}")
        
        # Sequential execution
        if not parallel_tool_execution or len(tool_calls) == 1:
            return [execute_single_tool(tool_call) for tool_call in tool_calls]
        
        # Parallel execution using ThreadPoolExecutor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tool_calls), max_workers)) as executor:
            future_to_tool_call = {
                executor.submit(execute_single_tool, tool_call): tool_call 
                for tool_call in tool_calls
            }
            
            execution_response = []
            for future in concurrent.futures.as_completed(future_to_tool_call):
                tool_call = future_to_tool_call[future]
                try:
                    result = future.result()
                    execution_response.append(result)
                except Exception as e:
                    # Re-raise with tool context
                    raise Exception(f"Error in parallel tool execution: {e}")
            
            # Sort results to maintain order of tool_calls
            call_id_to_result = {result["tool_call_id"]: result for result in execution_response}
            ordered_results = [call_id_to_result[tool_call.id] for tool_call in tool_calls]
            
            return ordered_results

    def _create_streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 5,
        max_workers: int = 10,
        **kwargs
    ) -> Iterator[Any]:
        """
        Create a streaming chat completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
        def streaming_generator():
            current_messages = messages.copy()
            remaining_tool_calls = max_tool_calls
            
            while True:
                if remaining_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")
                
                if tools:
                    tool_functions = {}
                    tool_schemas = []
                    
                    for tool in tools:
                        if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
                            tool_schemas.append(tool._tool_metadata)
                            tool_functions[tool._tool_metadata['function']['name']] = tool
                        else:
                            raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
                    
                    # Make streaming API call with tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        tools=tool_schemas,
                        stream=True,
                        **kwargs
                    )
                else:
                    # Make streaming API call without tools
                    stream = self._original_completions.create(
                        model=model,
                        messages=current_messages,
                        stream=True,
                        **kwargs
                    )
                
                # Accumulate the streamed response and detect tool calls
                accumulated_content = ""
                accumulated_tool_calls = []
                message_dict = {"role": "assistant", "content": ""}
                
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        # Yield the chunk for streaming output
                        yield chunk
                        
                        # Accumulate content
                        if delta.content:
                            accumulated_content += delta.content
                            message_dict["content"] += delta.content
                        
                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Ensure we have enough space in accumulated_tool_calls
                                while len(accumulated_tool_calls) <= tool_call_delta.index:
                                    accumulated_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                # Update the tool call at the correct index
                                if tool_call_delta.id:
                                    accumulated_tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        accumulated_tool_calls[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        accumulated_tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments
                
                # Check if we have tool calls to execute
                if accumulated_tool_calls and tools:
                    # Convert accumulated tool calls to proper format
                    tool_calls = []
                    for tc in accumulated_tool_calls:
                        if tc["id"] and tc["function"]["name"]:  # Only add complete tool calls
                            tool_call = type('ToolCall', (), {
                                'id': tc["id"],
                                'function': type('Function', (), {
                                    'name': tc["function"]["name"],
                                    'arguments': tc["function"]["arguments"]
                                })()
                            })()
                            tool_calls.append(tool_call)
                    
                    if tool_calls:
                        # Add assistant message with tool calls
                        message_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function", 
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in tool_calls
                        ]
                        current_messages.append(message_dict)
                        
                        # Execute tools
                        execution_response = self._execute_tools(
                            tool_functions,
                            tool_calls,
                            parallel_tool_execution,
                            max_workers=max_workers
                        )
                        remaining_tool_calls -= len(execution_response)
                        current_messages.extend(execution_response)
                        
                        # Continue the loop to get the next response
                        continue
                
                # No tool calls, we're done
                break
        
        return streaming_generator()
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)


# Async implementations

class OpenAIAsyncWrapper:
    """Async wrapper around OpenAI client that supports tool-py functions."""
    
    def __init__(self, client):
        self._client = client
        self.chat = ChatAsyncWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original client."""
        return getattr(self._client, name)


class ChatAsyncWrapper:
    """Async wrapper around OpenAI chat that handles toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self.completions = CompletionsAsyncWrapper(client)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original chat."""
        return getattr(self._client.chat, name)


class CompletionsAsyncWrapper:
    """Async wrapper around OpenAI completions that processes toolflow functions."""
    
    def __init__(self, client):
        self._client = client
        self._original_completions = client.chat.completions
    
    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Union[Any, AsyncIterator[Any]]:
        """
        Create a chat completion with tool support (async).
        
        Args:
            tools: List of toolflow decorated functions or OpenAI tool dicts
            parallel_tool_execution: Whether to execute multiple tool calls in parallel (default: False)
            max_tool_calls: Maximum number of tool calls to execute
            max_workers: Maximum number of worker threads to use for parallel execution of sync tools
            stream: Whether to stream the response (default: False)
            **kwargs: All other OpenAI chat completion parameters
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results, or AsyncIterator if stream=True
        """
        tools = kwargs.get('tools', None)
        parallel_tool_execution = kwargs.get('parallel_tool_execution', False)
        max_tool_calls = kwargs.get('max_tool_calls', 5)
        max_workers = kwargs.get('max_workers', 10)
        stream = kwargs.get('stream', False)

        # Handle streaming
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_execution=parallel_tool_execution,
                max_tool_calls=max_tool_calls,
                max_workers=max_workers,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
            )

        response = None
        if tools:
            tool_functions = {}
            tool_schemas = []
            
            for tool in tools:
                if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
                    tool_schemas.append(tool._tool_metadata)
                    tool_functions[tool._tool_metadata['function']['name']] = tool
                else:
                    raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
            
            # Tool execution loop
            while True:
                if max_tool_calls <= 0:
                    raise Exception("Max tool calls reached without finding a solution")

                # Make the API call
                response = await self._original_completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls', 'max_workers']}
                )

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    messages.append(response.choices[0].message)
                    execution_response = await self._execute_tools(
                        tool_functions, 
                        tool_calls, 
                        parallel_tool_execution,
                        max_workers=max_workers
                    )
                    max_tool_calls -= len(execution_response)
                    messages.extend(execution_response)
                else:
                    return response

        else: # No tools, just make the API call
            response = await self._original_completions.create(
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['tools', 'parallel_tool_execution', 'max_tool_calls']}
            )
        
        return response

    async def _execute_tools(
        self, 
        tool_functions: Dict[str, Callable],
        tool_calls: List[Dict[str, Any]],
        parallel_tool_execution: bool = False,
        max_workers: int = 10
    ):
        """Execute tool calls sequentially or in parallel, separating sync and async tools."""
        import asyncio
        import concurrent.futures
        
        async def execute_single_tool(tool_call):
            """Execute a single tool call (async)."""
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            tool_function = tool_functions.get(tool_name, None)
            if not tool_function:
                raise ValueError(f"Tool {tool_name} not found")
            
            try:
                # Parse JSON arguments and call the function
                parsed_args = json.loads(tool_args) if tool_args else {}
                
                # Check if the tool function is async
                if asyncio.iscoroutinefunction(tool_function):
                    result = await tool_function(**parsed_args)
                else:
                    # Run sync functions in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool_function(**parsed_args))
                
                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result) if not isinstance(result, str) else result
                }
            except Exception as e:
                raise Exception(f"Error executing tool {tool_name}: {e}")
        
        # Sequential execution
        if not parallel_tool_execution or len(tool_calls) == 1:
            return [await execute_single_tool(tool_call) for tool_call in tool_calls]
        
        # Parallel execution: separate sync and async tools for optimal performance
        sync_tools = []
        async_tools = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_function = tool_functions.get(tool_name)
            if tool_function and asyncio.iscoroutinefunction(tool_function):
                async_tools.append(tool_call)
            else:
                sync_tools.append(tool_call)
        
        # Execute sync and async tools in parallel
        tasks = []
        
        # Run sync tools in thread pool
        if sync_tools:
            def execute_sync_tools():
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sync_tools), max_workers)) as executor:
                    futures = []
                    for tool_call in sync_tools:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments
                        tool_function = tool_functions[tool_name]
                        
                        def execute_sync_tool(tc=tool_call, tf=tool_function):
                            try:
                                parsed_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                                result = tf(**parsed_args)
                                return {
                                    "tool_call_id": tc.id,
                                    "role": "tool",
                                    "content": json.dumps(result) if not isinstance(result, str) else result
                                }
                            except Exception as e:
                                raise Exception(f"Error executing tool {tc.function.name}: {e}")
                        
                        futures.append(executor.submit(execute_sync_tool))
                    
                    return [future.result() for future in futures]
            
            # Run sync tools in a separate thread pool
            loop = asyncio.get_event_loop()
            sync_task = loop.run_in_executor(None, execute_sync_tools)
            tasks.append(sync_task)
        
        # Run async tools with asyncio.gather
        if async_tools:
            async def execute_async_tools():
                async_tasks = []
                for tool_call in async_tools:
                    tool_name = tool_call.function.name
                    tool_function = tool_functions[tool_name]
                    
                    async def execute_async_tool(tc=tool_call, tf=tool_function):
                        try:
                            parsed_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            result = await tf(**parsed_args)
                            return {
                                "tool_call_id": tc.id,
                                "role": "tool",
                                "content": json.dumps(result) if not isinstance(result, str) else result
                            }
                        except Exception as e:
                            raise Exception(f"Error executing tool {tc.function.name}: {e}")
                    
                    async_tasks.append(execute_async_tool())
                
                return await asyncio.gather(*async_tasks)
            
            tasks.append(execute_async_tools())
        
        # Wait for both sync and async tools to complete
        if tasks:
            try:
                results = await asyncio.gather(*tasks)
                
                # Flatten results from sync and async execution
                all_results = []
                for result_group in results:
                    if isinstance(result_group, list):
                        all_results.extend(result_group)
                    else:
                        all_results.append(result_group)
                
                # Sort results to maintain order of tool_calls
                call_id_to_result = {result["tool_call_id"]: result for result in all_results}
                ordered_results = [call_id_to_result[tool_call.id] for tool_call in tool_calls]
                
                return ordered_results
            except Exception as e:
                raise Exception(f"Error in parallel tool execution: {e}")
        
        return []

    async def _create_streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Callable] = None,
        parallel_tool_execution: bool = False,
        max_tool_calls: int = 5,
        max_workers: int = 10,
        **kwargs
    ) -> AsyncIterator[Any]:
        """
        Create an async streaming chat completion with tool support.
        
        This method handles streaming responses while still supporting tool calls.
        When tool calls are detected in the stream, they are accumulated, executed,
        and then the conversation continues with a new streaming call.
        """
        current_messages = messages.copy()
        remaining_tool_calls = max_tool_calls
        
        while True:
            if remaining_tool_calls <= 0:
                raise Exception("Max tool calls reached without finding a solution")
            
            if tools:
                tool_functions = {}
                tool_schemas = []
                
                for tool in tools:
                    if isinstance(tool, Callable) and hasattr(tool, '_tool_metadata'):
                        tool_schemas.append(tool._tool_metadata)
                        tool_functions[tool._tool_metadata['function']['name']] = tool
                    else:
                        raise ValueError(f"Only decorated functions via @tool are supported. Got {type(tool)}")
                
                # Make streaming API call with tools
                stream = await self._original_completions.create(
                    model=model,
                    messages=current_messages,
                    tools=tool_schemas,
                    **kwargs
                )
            else:
                # Make streaming API call without tools
                stream = await self._original_completions.create(
                    model=model,
                    messages=current_messages,
                    **kwargs
                )
            
            # Accumulate the streamed response and detect tool calls
            accumulated_content = ""
            accumulated_tool_calls = []
            message_dict = {"role": "assistant", "content": ""}
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # Yield the chunk for streaming output
                    yield chunk
                    
                    # Accumulate content
                    if delta.content:
                        accumulated_content += delta.content
                        message_dict["content"] += delta.content
                    
                    # Accumulate tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            # Ensure we have enough space in accumulated_tool_calls
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # Update the tool call at the correct index
                            if tool_call_delta.id:
                                accumulated_tool_calls[tool_call_delta.index]["id"] = tool_call_delta.id
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    accumulated_tool_calls[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    accumulated_tool_calls[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Check if we have tool calls to execute
            if accumulated_tool_calls and tools:
                # Convert accumulated tool calls to proper format
                tool_calls = []
                for tc in accumulated_tool_calls:
                    if tc["id"] and tc["function"]["name"]:  # Only add complete tool calls
                        tool_call = type('ToolCall', (), {
                            'id': tc["id"],
                            'function': type('Function', (), {
                                'name': tc["function"]["name"],
                                'arguments': tc["function"]["arguments"]
                            })()
                        })()
                        tool_calls.append(tool_call)
                
                if tool_calls:
                    # Add assistant message with tool calls
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function", 
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                    current_messages.append(message_dict)
                    
                    # Execute tools
                    execution_response = await self._execute_tools(
                        tool_functions,
                        tool_calls,
                        parallel_tool_execution,
                        max_workers=max_workers
                    )
                    remaining_tool_calls -= len(execution_response)
                    current_messages.extend(execution_response)
                    
                    # Continue the loop to get the next response
                    continue
            
            # No tool calls, we're done
            break
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)
