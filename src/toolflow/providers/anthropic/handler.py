# src/toolflow/providers/anthropic/handler.py
import json
from typing import Any, List, Dict, Generator, AsyncGenerator
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageStreamEvent

from ...core.handlers import AbstractProviderHandler

class AnthropicHandler(AbstractProviderHandler):
    def __init__(self, client: Anthropic | AsyncAnthropic, original_create):
        self.client = client
        self.original_create = original_create

    def call_api(self, **kwargs) -> Any:
        return self.original_create(**kwargs)

    async def call_api_async(self, **kwargs) -> Any:
        return await self.original_create(**kwargs)

    def handle_response(self, response: Message) -> tuple[str | None, List[Dict], Any]:
        text_content = ""
        tool_calls = []
        
        for content_block in response.content:
            if hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    text_content += content_block.text
                elif content_block.type == 'thinking':
                    text_content += f"\n<THINKING>\n{content_block.thinking}\n</THINKING>\n\n"
                elif content_block.type == 'tool_use':
                    tool_calls.append(self._format_tool_call(content_block))
        
        return text_content if text_content else None, tool_calls, response

    def handle_streaming_response(self, response: Generator[RawMessageStreamEvent, None, None]) -> Generator[tuple[str | None, List[Dict] | None, Any], None, None]:
        accumulated_tool_calls = {}
        
        for event in response:
            text = None
            tool_calls = None
            
            if event.type == 'content_block_start':
                if hasattr(event.content_block, 'type'):
                    if event.content_block.type == 'text':
                        text = ""  # Start text block
                    elif event.content_block.type == 'tool_use':
                        # Initialize tool call
                        tool_call = {
                            "id": event.content_block.id,
                            "type": "function",
                            "function": {
                                "name": event.content_block.name,
                                "arguments": {}
                            }
                        }
                        accumulated_tool_calls[event.index] = tool_call
            
            elif event.type == 'content_block_delta':
                if hasattr(event.delta, 'type'):
                    if event.delta.type == 'text_delta':
                        text = event.delta.text
                    elif event.delta.type == 'input_json_delta':
                        # Accumulate tool arguments
                        if event.index in accumulated_tool_calls:
                            tool_call = accumulated_tool_calls[event.index]
                            if 'partial_json' not in tool_call['function']:
                                tool_call['function']['partial_json'] = ""
                            tool_call['function']['partial_json'] += event.delta.partial_json
            
            elif event.type == 'content_block_stop':
                # If this was a tool use block, finalize the tool call
                if event.index in accumulated_tool_calls:
                    tool_call = accumulated_tool_calls[event.index]
                    if 'partial_json' in tool_call['function']:
                        try:
                            tool_call['function']['arguments'] = json.loads(tool_call['function']['partial_json'])
                            del tool_call['function']['partial_json']
                        except json.JSONDecodeError:
                            # If JSON parsing fails, keep as empty dict
                            tool_call['function']['arguments'] = {}
                    tool_calls = [tool_call]
            
            yield text, tool_calls, event

    async def handle_streaming_response_async(self, response: AsyncGenerator[RawMessageStreamEvent, None]) -> AsyncGenerator[tuple[str | None, List[Dict] | None, Any], None]:
        accumulated_tool_calls = {}
        
        async for event in response:
            text = None
            tool_calls = None
            
            if event.type == 'content_block_start':
                if hasattr(event.content_block, 'type'):
                    if event.content_block.type == 'text':
                        text = ""  # Start text block
                    elif event.content_block.type == 'tool_use':
                        # Initialize tool call
                        tool_call = {
                            "id": event.content_block.id,
                            "type": "function",
                            "function": {
                                "name": event.content_block.name,
                                "arguments": {}
                            }
                        }
                        accumulated_tool_calls[event.index] = tool_call
            
            elif event.type == 'content_block_delta':
                if hasattr(event.delta, 'type'):
                    if event.delta.type == 'text_delta':
                        text = event.delta.text
                    elif event.delta.type == 'input_json_delta':
                        # Accumulate tool arguments
                        if event.index in accumulated_tool_calls:
                            tool_call = accumulated_tool_calls[event.index]
                            if 'partial_json' not in tool_call['function']:
                                tool_call['function']['partial_json'] = ""
                            tool_call['function']['partial_json'] += event.delta.partial_json
            
            elif event.type == 'content_block_stop':
                # If this was a tool use block, finalize the tool call
                if event.index in accumulated_tool_calls:
                    tool_call = accumulated_tool_calls[event.index]
                    if 'partial_json' in tool_call['function']:
                        try:
                            tool_call['function']['arguments'] = json.loads(tool_call['function']['partial_json'])
                            del tool_call['function']['partial_json']
                        except json.JSONDecodeError:
                            # If JSON parsing fails, keep as empty dict
                            tool_call['function']['arguments'] = {}
                    tool_calls = [tool_call]
            
            yield text, tool_calls, event

    def create_assistant_message(self, text: str | None, tool_calls: List[Dict]) -> Dict:
        """Create an assistant message with tool calls for Anthropic format."""
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        for tool_call in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": tool_call["function"]["arguments"]
            })
        
        return {
            "role": "assistant",
            "content": content
        }

    def create_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Create tool result messages for Anthropic format."""
        content = []
        for result in tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
                "content": str(result["output"])
            })
        
        return [{
            "role": "user",
            "content": content
        }]
    
    #@Override
    def prepare_tool_schemas(self, tools: List[Any]) -> tuple[List[Dict], Dict]:
        """Prepare tool schemas in Anthropic format."""
        
        # Get OpenAI-format schemas from parent
        openai_tool_schemas, tool_map = super().prepare_tool_schemas(tools)
        
        # Convert OpenAI format to Anthropic format
        anthropic_tool_schemas = []
        for openai_schema in openai_tool_schemas:
            anthropic_schema = {
                "name": openai_schema['function']['name'],
                "description": openai_schema['function']['description'],
                "input_schema": openai_schema['function']['parameters']
            }
            anthropic_tool_schemas.append(anthropic_schema)
        
        return anthropic_tool_schemas, tool_map

    def parse_structured_output(self, tool_call: Dict, response_format: Any) -> Any:
        """Handle the structured output from the tool call."""
        tool_arguments = tool_call["function"]["arguments"]
        response_data = tool_arguments.get('response', tool_arguments)
        return response_format.model_validate(response_data)

    def _format_tool_call(self, tool_use_block) -> Dict:
        """Format Anthropic tool_use block to standard format."""
        return {
            "id": tool_use_block.id,
            "type": "function",
            "function": {
                "name": tool_use_block.name,
                "arguments": tool_use_block.input,
            },
        }
