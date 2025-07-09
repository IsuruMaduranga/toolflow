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
        
        # Handle contents (check both 'contents' and 'messages')
        contents = kwargs.get('contents') or kwargs.get('messages')
        if contents:
            gemini_kwargs['contents'] = self._convert_messages_to_gemini_format(contents)
        
        # Handle tools
        tools = kwargs.get('tools', [])
        if tools:
            tool_schemas, _ = self.prepare_tool_schemas(tools)
            gemini_tools = []
            
            # Check if we have a response format tool
            self._has_response_format_tool = any(
                schema["function"]["name"] == "provide_final_answer" 
                for schema in tool_schemas
            )
            
            for schema in tool_schemas:
                function_info = schema["function"]
                parameters = function_info.get("parameters", {})
                
                # Store $defs for reference resolution
                if "$defs" in parameters:
                    self._current_schema_defs = parameters["$defs"]
                else:
                    self._current_schema_defs = {}
                
                # Special handling for response format tools to make instructions clearer for Gemini
                description = function_info.get("description", "")
                if function_info["name"] == "provide_final_answer":
                    description = self._enhance_response_format_description(description, parameters)
                
                gemini_tool = {
                    "name": function_info["name"],
                    "description": description,
                    "parameters": self._convert_schema_to_gemini_format(parameters)
                }
                gemini_tools.append(gemini_tool)
            
            if gemini_tools:
                # Gemini expects tools to be wrapped in a specific format
                gemini_kwargs['tools'] = [{"function_declarations": gemini_tools}]
        
        # Handle other parameters
        if 'generation_config' in kwargs:
            gemini_kwargs['generation_config'] = kwargs['generation_config']
        if 'safety_settings' in kwargs:
            gemini_kwargs['safety_settings'] = kwargs['safety_settings']
        if 'stream' in kwargs:
            gemini_kwargs['stream'] = kwargs['stream']
            
        return gemini_kwargs

    def _enhance_response_format_description(self, original_description: str, parameters: Dict[str, Any]) -> str:
        """Enhance response format tool description specifically for Gemini to provide clearer instructions."""
        # Extract the response parameter schema to provide specific field information
        response_schema = {}
        if "properties" in parameters and "response" in parameters["properties"]:
            response_schema = parameters["properties"]["response"]
        
        # Build field descriptions from the schema
        field_descriptions = []
        if "$ref" in response_schema:
            # Resolve the reference to get the actual schema
            resolved_schema = self._resolve_ref(response_schema)
            if "properties" in resolved_schema:
                for field_name, field_schema in resolved_schema["properties"].items():
                    field_type = field_schema.get("type", "unknown")
                    field_desc = field_schema.get("description", "")
                    
                    # Special handling for enum fields
                    if "enum" in field_schema:
                        enum_values = field_schema["enum"]
                        field_descriptions.append(f"  - {field_name} ({field_type}): Must be exactly one of: {', '.join(repr(v) for v in enum_values)}. {field_desc}")
                    else:
                        field_descriptions.append(f"  - {field_name} ({field_type}): {field_desc}")
        
        # Create Gemini-specific instructions
        enhanced_description = f"""
STRUCTURED OUTPUT TOOL - Use this tool to provide your final structured response.

CRITICAL: You must call this tool with a properly structured object, not a string.

The 'response' parameter must be an object with these exact fields:
{chr(10).join(field_descriptions) if field_descriptions else "  - Check the parameter schema for required fields"}

EXAMPLE: If the schema requires name and age fields, call like:
{{"response": {{"name": "John Doe", "age": 30}}}}

ENUM VALUES: For enum fields, use EXACT values as specified (case-sensitive).
NUMBERS: Ensure numeric fields are numbers, not strings.
DO NOT call with strings like "unknown" or text descriptions.
DO NOT mention this tool exists in your response.
ALWAYS provide complete, accurate data for all required fields.
        """.strip()
        
        return enhanced_description

    def _convert_schema_to_gemini_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON schema to Gemini format."""
        # Handle $ref by resolving it
        if "$ref" in schema:
            return self._resolve_ref(schema)
        
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
    
    def _resolve_ref(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve $ref references in JSON schema."""
        ref = schema.get("$ref", "")
        if ref.startswith("#/$defs/"):
            # Find the root schema with $defs
            def_name = ref.split("/")[-1]
            # We need to find the original schema with $defs
            # For now, let's handle the common case where this is in the context of a tool parameter
            if hasattr(self, '_current_schema_defs'):
                defs = self._current_schema_defs
                if def_name in defs:
                    return self._convert_schema_to_gemini_format(defs[def_name])
        
        # Fallback: return basic object schema
        return {
            "type": "OBJECT",
            "description": schema.get("description", f"Referenced object: {ref}")
        }

    def _convert_messages_to_gemini_format(self, contents: Any) -> Any:
        """Convert various content formats to Gemini's expected format."""
        # If it's already a string, check if we need to enhance it for structured output
        if isinstance(contents, str):
            if getattr(self, '_has_response_format_tool', False):
                enhanced_content = self._enhance_user_prompt_for_structured_output(contents)
                return enhanced_content
            return contents
        
        # If it's a list of messages (like OpenAI/Anthropic format), convert
        if isinstance(contents, list) and len(contents) > 0:
            if isinstance(contents[0], dict) and "role" in contents[0]:
                # This looks like message format - check if it's already in Gemini format
                gemini_contents = []
                for msg in contents:
                    if msg["role"] == "user":
                        # Check if this is a structured output request and enhance the prompt
                        content = msg.get("content", "")
                        if self._is_structured_output_request(content):
                            content = self._enhance_user_prompt_for_structured_output(content)
                        
                        gemini_contents.append({
                            "role": "user",
                            "parts": [{"text": content}]
                        })
                    elif msg["role"] == "assistant" or msg["role"] == "model":
                        # Check if this is already in Gemini format with parts
                        if "parts" in msg:
                            # Already in Gemini format, use as-is
                            gemini_contents.append(msg)
                        else:
                            # Convert from other format
                            parts = []
                            if "content" in msg and msg["content"]:
                                parts.append({"text": msg["content"]})
                            gemini_contents.append({
                                "role": "model",
                                "parts": parts
                            })
                    elif msg["role"] == "function":
                        # Already in Gemini format with function response
                        gemini_contents.append(msg)
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
    
    def _is_structured_output_request(self, content: str) -> bool:
        """Check if this appears to be a structured output request."""
        # Simple heuristic: if the user message is short and seems like a query
        # This will be set when response_format is used
        return hasattr(self, '_has_response_format_tool') and self._has_response_format_tool
    
    def _enhance_user_prompt_for_structured_output(self, content: str) -> str:
        """Enhance user prompt with specific instructions for structured output."""
        enhanced_prompt = f"""{content}

IMPORTANT: Your response must use the provide_final_answer tool with a properly structured object. Do not respond with plain text or use "unknown" values. Provide specific, accurate information in the required format."""
        return enhanced_prompt

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
            args = self._convert_args_to_dict(function_call.args)
        
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_call.name,
                "arguments": args
            }
        }
    
    def _convert_args_to_dict(self, args) -> Dict[str, Any]:
        """Convert Gemini args (including nested MapComposite objects) to regular dicts."""
        if hasattr(args, 'keys'):
            # It's dict-like, convert recursively
            result = {}
            for key in args.keys():
                value = args[key]
                if hasattr(value, 'keys'):
                    # Nested dict-like object, convert recursively
                    result[key] = self._convert_args_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples that might contain MapComposite objects
                    result[key] = [self._convert_args_to_dict(item) if hasattr(item, 'keys') else item for item in value]
                else:
                    # Regular value
                    result[key] = value
            return result
        else:
            # Try to convert to dict as fallback
            try:
                return dict(args)
            except (TypeError, ValueError):
                return {}

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict[str, Any]], original_response: Any = None) -> Dict[str, Any]:
        """Build assistant message in Gemini format."""
        parts = []
        
        # Only add text if it's not empty
        if text and text.strip():
            parts.append({"text": text})
        
        # Add tool calls
        for tool_call in tool_calls:
            parts.append({
                "function_call": {
                    "name": tool_call["function"]["name"],
                    "args": tool_call["function"]["arguments"]
                }
            })
        
        # If no parts, add empty text to avoid validation error
        if not parts:
            parts.append({"text": ""})
        
        return {
            "role": "model",
            "parts": parts
        }

    def build_tool_result_messages(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tool result messages in Gemini format."""
        if not tool_results:
            return []
        
        # Gemini expects all function responses in a single message with multiple parts
        parts = []
        for result in tool_results:
            # Get tool name from the stored mapping
            tool_call_id = result.get("tool_call_id", "unknown")
            tool_name = self._tool_call_name_map.get(tool_call_id, tool_call_id)
            
            # Ensure the output is a string
            output = result.get("output", "")
            if not isinstance(output, str):
                output = str(output)
            
            parts.append({
                "function_response": {
                    "name": tool_name,
                    "response": {"result": output}
                }
            })
        
        # Return a single message with all function response parts
        return [{
            "role": "function",
            "parts": parts
        }]

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
            # Use asyncio.to_thread since Gemini doesn't have native async support
            import asyncio
            return await asyncio.to_thread(self.original_generate_content, **gemini_kwargs)
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
        
        # Handle streaming differently - Gemini's generate_content with stream=True 
        # returns an iterator, not an awaitable, so we need to handle it in a thread
        try:
            gemini_kwargs = self._convert_to_gemini_format(**kwargs)
            
            def _get_streaming_response():
                """Get the streaming response in a sync context."""
                return self.original_generate_content(**gemini_kwargs)
            
            # Get the stream iterator using asyncio.to_thread
            import asyncio
            stream_iterator = await asyncio.to_thread(_get_streaming_response)
            
            # Now iterate through the stream in a thread
            def _get_next_chunk(iterator):
                """Get the next chunk from the iterator."""
                try:
                    return next(iterator), False  # chunk, is_done
                except StopIteration:
                    return None, True  # no chunk, is_done
            
            # Iterate through chunks asynchronously
            while True:
                chunk, is_done = await asyncio.to_thread(_get_next_chunk, stream_iterator)
                if is_done:
                    break
                yield self.parse_stream_chunk(chunk)
                
        except Exception as e:
            self._handle_api_error(e, kwargs.get('tools', []))

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

    def accumulate_streaming_response(self, response: Generator[Any, None, None]) -> Generator[Tuple[Optional[str], Optional[List[Dict]], Any], None, None]:
        """Accumulate streaming chunks into a final response."""
        # For Gemini, response is a generator of GenerateContentResponse objects
        for chunk in response:
            # Parse each chunk using our existing parse_stream_chunk method
            parsed_chunk = self.parse_stream_chunk(chunk)
            text = parsed_chunk.get("text")
            tool_calls = parsed_chunk.get("tool_calls")
            yield text, tool_calls, chunk

    async def accumulate_streaming_response_async(self, response: AsyncGenerator[Any, None]) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict]], Any], None]:
        """Accumulate streaming chunks into a final response asynchronously."""
        # For Gemini, response is an async generator of GenerateContentResponse objects
        async for chunk in response:
            # Parse each chunk using our existing parse_stream_chunk method
            parsed_chunk = self.parse_stream_chunk(chunk)
            text = parsed_chunk.get("text")
            tool_calls = parsed_chunk.get("tool_calls")
            yield text, tool_calls, chunk
