# src/toolflow/providers/gemini/handler.py

from __future__ import annotations

from typing import Any, List, Dict, Optional, Tuple, Union, Generator, AsyncGenerator
import json

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
except ImportError:
    # Mock classes for when google-generativeai is not installed
    class GenerateContentResponse:
        pass

from toolflow.core import TransportAdapter, MessageAdapter, ResponseFormatAdapter
from toolflow.core.utils import get_tool_schema

class GeminiHandler(TransportAdapter, MessageAdapter, ResponseFormatAdapter):
    """Handler for Gemini API that implements all three adapter interfaces."""
    
    def __init__(self, client, original_generate_content):
        self.client = client
        self.original_generate_content = original_generate_content
        self._tool_call_name_map = {}  # Map tool_call_id to tool_name

    def call_api(self, **kwargs: Any) -> GenerateContentResponse:
        """Call the Gemini API with converted parameters."""
        try:
            gemini_kwargs = self._convert_to_gemini_format(**kwargs)
            return self.original_generate_content(**gemini_kwargs)
        except Exception as e:
            self._handle_api_error(e, kwargs.get('tools', []))

    def _convert_to_gemini_format(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert toolflow parameters to Gemini API format."""
        gemini_kwargs = {}
        
        # Handle contents
        contents = kwargs.get('contents')
        if contents:
            gemini_kwargs['contents'] = self._convert_messages_to_gemini_format(contents)
        
        # Handle tools
        tools = kwargs.get('tools', [])
        if tools:
            tool_schemas, _ = self.prepare_tool_schemas(tools)
            gemini_tools = []
            
            for schema in tool_schemas:
                function_info = schema["function"]
                gemini_tool = {
                    "name": function_info["name"],
                    "description": function_info.get("description", ""),
                    "parameters": self._convert_schema_to_gemini_format(function_info.get("parameters", {}))
                }
                gemini_tools.append(gemini_tool)
            
            if gemini_tools:
                gemini_kwargs['tools'] = gemini_tools
        
        # Handle other parameters
        if 'generation_config' in kwargs:
            gemini_kwargs['generation_config'] = kwargs['generation_config']
        if 'safety_settings' in kwargs:
            gemini_kwargs['safety_settings'] = kwargs['safety_settings']
        if 'stream' in kwargs:
            gemini_kwargs['stream'] = kwargs['stream']
            
        return gemini_kwargs

    def _convert_schema_to_gemini_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON schema to Gemini format."""
        gemini_schema = {}
        
        if "type" in schema:
            gemini_schema["type"] = schema["type"].upper()
        
        if "properties" in schema:
            gemini_schema["properties"] = {}
            for prop_name, prop_schema in schema["properties"].items():
                gemini_schema["properties"][prop_name] = self._convert_schema_to_gemini_format(prop_schema)
        
        if "required" in schema:
            gemini_schema["required"] = schema["required"]
            
        if "description" in schema:
            gemini_schema["description"] = schema["description"]
        
        # Handle array items
        if "items" in schema:
            gemini_schema["items"] = self._convert_schema_to_gemini_format(schema["items"])
            
        return gemini_schema

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

    def _handle_api_error(self, error: Exception, tools: List[Any]) -> None:
        """Handle API errors with context."""
        raise error

    def parse_response(self, response: GenerateContentResponse) -> Tuple[Optional[str], List[Dict], Any]:
        """Parse a complete response into (text, tool_calls, raw_response)."""
        text_content = ""
        tool_calls = []
        
        # Safely try to get text content - avoid accessing .text if it will error
        try:
            if hasattr(response, 'text') and response.text:
                text_content = response.text
        except ValueError:
            # This happens when response has no valid text parts (e.g., only function calls)
            text_content = ""
        
        # Extract tool calls from response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    # Check if this part has a function call
                    if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call, 'name'):
                        tool_calls.append(self._format_tool_call(part.function_call))
                    elif hasattr(part, 'text') and part.text:
                        # This is a text part, add to text content if we don't already have it
                        if not text_content:
                            text_content = part.text
        
        return text_content, tool_calls, response

    def check_max_tokens_reached(self, response: GenerateContentResponse) -> bool:
        """Check if max tokens was reached and return True if so."""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                return candidate.finish_reason == "MAX_TOKENS"
        return False

    def extract_text_from_response(self, response: GenerateContentResponse) -> str:
        """Extract text content from response."""
        try:
            if hasattr(response, 'text'):
                return response.text or ""
        except ValueError:
            # This happens when response has no valid text parts
            pass
        return ""

    def _format_tool_call(self, function_call) -> Dict[str, Any]:
        """Format a Gemini function call into standard tool call format."""
        # Generate a unique ID for the tool call
        tool_call_id = f"call_{hash(str(function_call))}"[:12]
        
        # Store the mapping from tool_call_id to tool name
        self._tool_call_name_map[tool_call_id] = function_call.name
        
        # Handle arguments - convert to dict if it's not already
        args = {}
        if hasattr(function_call, 'args') and function_call.args:
            if hasattr(function_call.args, 'keys'):
                # It's already dict-like
                args = dict(function_call.args)
            else:
                # Try to convert to dict
                try:
                    args = dict(function_call.args)
                except (TypeError, ValueError):
                    args = {}
        
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_call.name,
                "arguments": args
            }
        }

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict[str, Any]], original_response: Any = None) -> Dict[str, Any]:
        """Build assistant message in Gemini format."""
        parts = []
        
        if text:
            parts.append({"text": text})
        
        for tool_call in tool_calls:
            parts.append({
                "function_call": {
                    "name": tool_call["function"]["name"],
                    "args": tool_call["function"]["arguments"]
                }
            })
        
        return {
            "role": "model",
            "parts": parts
        }

    def build_tool_result_messages(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tool result messages in Gemini format."""
        messages = []
        
        for result in tool_results:
            # Get tool name from the stored mapping
            tool_call_id = result.get("tool_call_id", "unknown")
            tool_name = self._tool_call_name_map.get(tool_call_id, tool_call_id)
            
            messages.append({
                "role": "function",
                "parts": [{
                    "function_response": {
                        "name": tool_name,
                        "response": {
                            "result": result["output"]
                        }
                    }
                }]
            })
        
        return messages

    def prepare_tool_schemas(self, tools: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare tool schemas for Gemini."""
        tool_schemas = []
        tool_map = {}
        
        for tool in tools:
            # Check if tool is already a schema (dict with 'type' and 'function' keys)
            if isinstance(tool, dict) and tool.get('type') == 'function' and 'function' in tool:
                # It's already a schema, use it directly
                schema = tool
                tool_map[schema["function"]["name"]] = tool  # Keep the schema as the tool
            else:
                # It's a function, generate schema
                schema = get_tool_schema(tool)
                tool_map[schema["function"]["name"]] = tool
            
            tool_schemas.append(schema)
        
        return tool_schemas, tool_map

    def extract_response_format_from_response(self, response: GenerateContentResponse, response_format: Any) -> Any:
        """Extract structured response format from Gemini response."""
        # For now, just return the text and let the parent class handle parsing
        text = self.extract_text_from_response(response)
        return text

    # Abstract methods implementation
    async def call_api_async(self, **kwargs: Any) -> GenerateContentResponse:
        """Call the Gemini API asynchronously with converted parameters."""
        try:
            gemini_kwargs = self._convert_to_gemini_format(**kwargs)
            return await self.original_generate_content(**gemini_kwargs)
        except Exception as e:
            self._handle_api_error(e, kwargs.get('tools', []))

    def stream_response(self, **kwargs: Any) -> Generator[Any, None, None]:
        """Stream response from the API."""
        kwargs['stream'] = True
        stream = self.call_api(**kwargs)
        for chunk in stream:
            yield self.parse_stream_chunk(chunk)

    async def stream_response_async(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Stream response from the API asynchronously."""
        kwargs['stream'] = True
        stream = await self.call_api_async(**kwargs)
        async for chunk in stream:
            yield self.parse_stream_chunk(chunk)

    def parse_stream_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Parse a streaming chunk into a standardized format."""
        text_content = ""
        tool_calls = []
        
        # Safely try to get text content from chunk
        try:
            if hasattr(chunk, 'text') and chunk.text:
                text_content = chunk.text
        except ValueError:
            # This happens when chunk has no valid text parts
            text_content = ""
        
        # Extract tool calls from chunk
        if hasattr(chunk, 'candidates') and chunk.candidates:
            candidate = chunk.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call, 'name'):
                        tool_calls.append(self._format_tool_call(part.function_call))
        
        return {
            "text": text_content,
            "tool_calls": tool_calls,
            "raw": chunk
        }

    def accumulate_streaming_response(self, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict], Any]:
        """Accumulate streaming chunks into a final response."""
        text_parts = []
        all_tool_calls = []
        raw_chunks = []
        
        for chunk in chunks:
            if chunk.get("text"):
                text_parts.append(chunk["text"])
            if chunk.get("tool_calls"):
                all_tool_calls.extend(chunk["tool_calls"])
            raw_chunks.append(chunk.get("raw"))
        
        return "".join(text_parts), all_tool_calls, raw_chunks

    async def accumulate_streaming_response_async(self, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict], Any]:
        """Accumulate streaming chunks into a final response asynchronously."""
        return self.accumulate_streaming_response(chunks)
