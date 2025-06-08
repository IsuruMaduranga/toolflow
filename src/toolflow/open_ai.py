from typing import Any, Dict, List, Callable
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
    ):
        """
        Create a chat completion with tool support.
        
        Args:
            tools: List of toolflow decorated functions or OpenAI tool dicts
            auto_execute: Whether to automatically execute tool calls
            max_tool_calls: Maximum number of tool calls to execute
            **kwargs: All other OpenAI chat completion parameters
        
        Returns:
            OpenAI ChatCompletion response, potentially with tool results
        """
        tools = kwargs.get('tools', None)
        max_tool_calls = kwargs.get('max_tool_calls', 5)
        
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
                    **{k: v for k, v in kwargs.items() if k not in ['tools', 'max_tool_calls']}
                )

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    messages.append(response.choices[0].message)
                    exexution_response = self._execute_tools(
                        tool_functions, tool_calls, max_tool_calls, **kwargs
                    )
                    max_tool_calls -= len(exexution_response)
                    messages.extend(exexution_response)
                else:
                    return response

        else: # No tools, just make the API call
            response = self._original_completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        
        return response

    def _execute_tools(
        self, 
        tool_functions: Dict[str, Callable],
        tool_calls: List[Dict[str, Any]],
        max_tool_calls: int,
        **original_kwargs
    ):
        """Execute tool calls and create a new response with results."""
        
        execution_response = []
        
        # Execute each tool call sequentially
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            tool_result = None
            tool_function = tool_functions.get(tool_name, None)
            if tool_function:
                try:
                    # Parse JSON arguments and call the function directly
                    parsed_args = json.loads(tool_args) if tool_args else {}
                    result = tool_function(**parsed_args)
                    tool_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": json.dumps(result) if not isinstance(result, str) else result
                    }
                except Exception as e:
                    raise Exception(f"Error executing tool {tool_name}: {e}")
            else:
                raise ValueError(f"Tool {tool_name} not found")
            
            execution_response.append(tool_result)
        
        return execution_response
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original completions."""
        return getattr(self._original_completions, name)
