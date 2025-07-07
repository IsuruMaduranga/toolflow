# src/toolflow/providers/gemini/wrappers.py

from __future__ import annotations

from typing import Any, List, Dict, overload, Iterable, Optional, Union, Generator
from typing_extensions import Literal

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    from google.generativeai.types.generation_types import GenerationConfig
    from google.generativeai.types.safety_types import SafetySetting
    from google.generativeai.types.content_types import Content, Part
except ImportError:
    # Mock classes for when google-generativeai is not installed
    class GenerateContentResponse:
        pass
    class GenerationConfig:
        pass
    class SafetySetting:
        pass
    class Content:
        pass
    class Part:
        pass

from toolflow.core import ExecutorMixin
from .handler import GeminiHandler

# --- Synchronous Wrappers ---

class GeminiWrapper(ExecutorMixin):
    """Wrapped Gemini client that transparently adds toolflow capabilities."""
    def __init__(self, client, full_response: bool = False):
        self._client = client
        self.full_response = full_response
        self.original_generate_content = client.generate_content
        self.handler = GeminiHandler(client, client.generate_content)

    @overload
    def generate_content(
        self,
        contents: Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any]],
        *,
        # Standard Gemini parameters
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        stream: Optional[Literal[False]] = None,
        tools: Optional[List[Any]] = None,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> GenerateContentResponse | str:
        """
        Generates content using the Gemini model. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            contents: The content to generate a response for. Can be:
                - str: A simple text prompt
                - List[str|Dict]: Multiple content parts
                - Dict: A single content part with role and parts
            
            # Standard Gemini parameters
            generation_config: Configuration for generation (temperature, max_tokens, etc.)
            safety_settings: Safety settings to filter harmful content
            stream: Enable streaming response (false for this overload)
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your Pydantic model.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Gemini response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)

        Returns:
            GenerateContentResponse | str: The model's response
        """
        ...
    
    @overload
    def generate_content(
        self,
        contents: Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any]],
        *,
        stream: Literal[True],
        # Standard Gemini parameters  
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        tools: Optional[List[Any]] = None,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> Generator[GenerateContentResponse, None, None] | Generator[str, None, None]:
        """
        Generates streaming content using the Gemini model. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            contents: The content to generate a response for
            stream: Enable streaming response (true for this overload)
            
            # Standard Gemini parameters
            generation_config: Configuration for generation
            safety_settings: Safety settings to filter harmful content
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your Pydantic model.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Gemini response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            Generator[GenerateContentResponse, None, None] | Generator[str, None, None]: Stream of response chunks
        """
        ...
    
    @overload
    def generate_content(
        self,
        contents: Union[str, List[Union[str, Dict[str, Any]]], Dict[str, Any]],
        *,
        stream: bool,
        # Standard Gemini parameters  
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[List[SafetySetting]] = None,
        tools: Optional[List[Any]] = None,
        # Toolflow-specific parameters
        max_tool_call_rounds: Optional[int] = None,
        max_response_format_retries: Optional[int] = None,
        parallel_tool_execution: bool = True,
        response_format: Optional[Any] = None,
        full_response: Optional[bool] = None,
        graceful_error_handling: bool = True,
    ) -> GenerateContentResponse | str | Generator[GenerateContentResponse, None, None] | Generator[str, None, None]:
        """
        Generates content using the Gemini model with dynamic streaming. Enhanced by toolflow with auto-parallel tool calling.

        Args:
            contents: The content to generate a response for
            stream: Enable streaming response (dynamic based on runtime value)
            
            # Standard Gemini parameters
            generation_config: Configuration for generation
            safety_settings: Safety settings to filter harmful content
            
            # Toolflow-specific parameters
            tools: A list of python functions that the model may call.
                This is enhanced by toolflow. Just pass your regular python functions as tools.
                Toolflow will handle the rest.
            response_format: A pydantic model that the model will output.
                This is enhanced by toolflow. Just pass your Pydantic model.
                Toolflow will handle the rest.
            max_tool_call_rounds: Maximum number of tool calls to execute (default: 10)
            max_response_format_retries: Maximum number of response format retries (default: 2)
            parallel_tool_execution: Execute tool calls in parallel (default: True)
            full_response: Return full Gemini response object instead of content only
            graceful_error_handling: Handle errors gracefully (default: True)
            
        Returns:
            GenerateContentResponse | str | Generator: Complete response or stream based on stream parameter
        """
        ...

    def generate_content(self, contents: Any = None, **kwargs: Any) -> Any:
        if contents is not None:
            kwargs['contents'] = contents
        return self._create_sync(**kwargs)

    def _prepare_gemini_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert toolflow tools to Gemini function declarations."""
        if not tools:
            return []
        
        tool_schemas, _ = self.handler.prepare_tool_schemas(tools)
        gemini_tools = []
        
        for schema in tool_schemas:
            # Convert OpenAI/Toolflow schema to Gemini format
            function_info = schema["function"]
            gemini_schema = {
                "name": function_info["name"],
                "description": function_info.get("description", ""),
                "parameters": function_info.get("parameters", {})
            }
            gemini_tools.append(gemini_schema)
        
        return gemini_tools

    def _convert_messages_to_gemini_format(self, contents: Any) -> Any:
        """Convert various content formats to Gemini's expected format."""
        # If it's already a string, return as-is
        if isinstance(contents, str):
            return contents
        
        # If it's a list of messages (like OpenAI/Anthropic format), convert
        if isinstance(contents, list) and len(contents) > 0:
            if isinstance(contents[0], dict) and "role" in contents[0]:
                # This looks like OpenAI/Anthropic message format
                # Convert to Gemini format
                gemini_contents = []
                for msg in contents:
                    if msg["role"] == "user":
                        gemini_contents.append({
                            "role": "user",
                            "parts": [{"text": msg.get("content", "")}]
                        })
                    elif msg["role"] == "assistant" or msg["role"] == "model":
                        parts = []
                        if "content" in msg and msg["content"]:
                            parts.append({"text": msg["content"]})
                        gemini_contents.append({
                            "role": "model",
                            "parts": parts
                        })
                    elif msg["role"] == "system":
                        # Gemini doesn't have system messages, prepend to first user message
                        if gemini_contents and gemini_contents[-1]["role"] == "user":
                            existing_text = gemini_contents[-1]["parts"][0]["text"]
                            gemini_contents[-1]["parts"][0]["text"] = f"System: {msg['content']}\n\nUser: {existing_text}"
                        else:
                            gemini_contents.append({
                                "role": "user",
                                "parts": [{"text": f"System: {msg['content']}"}]
                            })
                
                return gemini_contents
        
        # Return as-is for other formats
        return contents

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
