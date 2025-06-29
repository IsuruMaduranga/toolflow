# src/toolflow/providers/openai/handlers.py
import json
from typing import Any, List, Dict, Generator, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from ...core.handlers import AbstractProviderHandler

class OpenAIHandler(AbstractProviderHandler):
    def __init__(self, client: OpenAI | AsyncOpenAI, original_create):
        self.client = client
        self.original_create = original_create

    def call_api(self, **kwargs) -> Any:
        return self.original_create(**kwargs)

    async def call_api_async(self, **kwargs) -> Any:
        return await self.original_create(**kwargs)

    def handle_response(self, response: ChatCompletion) -> tuple[str | None, List[Dict], Any]:
        message = response.choices[0].message
        text = message.content
        tool_calls = []
        if message.tool_calls:
            tool_calls = [self._format_tool_call(tc) for tc in message.tool_calls]
        return text, tool_calls, response

    def handle_streaming_response(self, response: Generator[ChatCompletionChunk, None, None]) -> Generator[tuple[str | None, List[Dict] | None, Any], None, None]:
        tool_calls = []
        for chunk in response:
            delta = chunk.choices[0].delta
            text = delta.content
            
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    if len(tool_calls) <= tool_call_chunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    
                    tc = tool_calls[tool_call_chunk.index]
                    if tool_call_chunk.id:
                        tc["id"] += tool_call_chunk.id
                    if tool_call_chunk.function:
                        if tool_call_chunk.function.name:
                            tc["function"]["name"] += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            tc["function"]["arguments"] += tool_call_chunk.function.arguments
            
            yield text, None, chunk # Yielding partial tool calls is complex, handling at the end for now

    async def handle_streaming_response_async(self, response: AsyncGenerator[ChatCompletionChunk, None]) -> AsyncGenerator[tuple[str | None, List[Dict] | None, Any], None]:
        tool_calls = []
        async for chunk in response:
            delta = chunk.choices[0].delta
            text = delta.content
            
            if delta.tool_calls:
                # Simplified streaming logic for this refactor
                # A more robust implementation would properly accumulate and yield tool calls
                pass
            
            yield text, None, chunk

    def create_assistant_message(self, text: str | None, tool_calls: List[Dict]) -> Dict:
        """Create an assistant message with tool calls for OpenAI format."""
        message = {
            "role": "assistant",
            "content": text,
        }
        if tool_calls:
            # Convert tool calls back to OpenAI format
            openai_tool_calls = []
            for tc in tool_calls:
                openai_tool_calls.append({
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"])
                    }
                })
            message["tool_calls"] = openai_tool_calls
        return message

    def create_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Create individual tool result messages for OpenAI format."""
        messages = []
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(result["output"])
            })
        return messages

    def _format_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> Dict:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            },
        }
