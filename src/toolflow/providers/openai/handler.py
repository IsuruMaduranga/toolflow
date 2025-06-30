# src/toolflow/providers/openai/handlers.py
import json
from typing import Any, List, Dict, Generator, AsyncGenerator, Union, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from toolflow.core import Handler

class OpenAIHandler(Handler):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI], original_create):
        self.client = client
        self.original_create = original_create

    def call_api(self, **kwargs) -> Any:
        return self.original_create(**kwargs)

    async def call_api_async(self, **kwargs) -> Any:
        return await self.original_create(**kwargs)

    def stream_response(self, response: Generator[ChatCompletionChunk, None, None]) -> Generator[ChatCompletionChunk, None, None]:
        """Handle a streaming response and yield raw chunks."""
        for chunk in response:
            yield chunk

    async def stream_response_async(self, response: AsyncGenerator[ChatCompletionChunk, None]) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle an async streaming response and yield raw chunks."""
        async for chunk in response:
            yield chunk

    def parse_response(self, response: ChatCompletion) -> Tuple[Optional[str], List[Dict], Any]:
        """Parse a complete response into (text, tool_calls, raw_response)."""
        message = response.choices[0].message
        text = message.content
        tool_calls = []
        if message.tool_calls:
            tool_calls = [self._format_tool_call(tc) for tc in message.tool_calls]
        return text, tool_calls, response

    def check_max_tokens_reached(self, response: ChatCompletion) -> None:
        """Check if max tokens was reached and raise exception if so."""
        if response.choices[0].finish_reason == "length":
            raise Exception("Max tokens reached without finding a solution")

    def parse_stream_chunk(self, chunk: ChatCompletionChunk) -> Tuple[Optional[str], Optional[List[Dict]], Any]:
        """Parse a streaming chunk into (text, tool_calls, raw_chunk)."""
        # Note: For OpenAI, tool calls are accumulated across chunks, so individual chunks
        # typically only contain text. This method parses individual chunks.
        # Tool call accumulation is handled by the handle_streaming_response method.
        delta = chunk.choices[0].delta
        text = delta.content
        tool_calls = None
        
        # Individual chunks rarely contain complete tool calls for OpenAI
        # Tool call completion is handled by the streaming execution logic
        if delta.tool_calls:
            # This is a partial tool call, mark as None since it's incomplete
            tool_calls = None
            
        return text, tool_calls, chunk

    def accumulate_streaming_response(self, response: Generator[ChatCompletionChunk, None, None]) -> Generator[Tuple[Optional[str], Optional[List[Dict]], Any], None, None]:
        """Handle streaming response with proper tool call accumulation."""
        tool_calls = []
        for chunk in self.stream_response(response):
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
            
            yield text, None, chunk
        
        # After stream completes, yield tool calls if any were accumulated
        if tool_calls:
            # Parse JSON arguments for each tool call
            formatted_tool_calls = []
            for tc in tool_calls:
                if tc["function"]["arguments"]:
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        # If JSON parsing fails, keep as string
                        pass
                formatted_tool_calls.append(tc)
            yield None, formatted_tool_calls, None

    async def accumulate_streaming_response_async(self, response: AsyncGenerator[ChatCompletionChunk, None]) -> AsyncGenerator[Tuple[Optional[str], Optional[List[Dict]], Any], None]:
        """Handle async streaming response with proper tool call accumulation."""
        tool_calls = []
        async for chunk in self.stream_response_async(response):
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
            
            yield text, None, chunk
        
        # After stream completes, yield tool calls if any were accumulated
        if tool_calls:
            # Parse JSON arguments for each tool call
            formatted_tool_calls = []
            for tc in tool_calls:
                if tc["function"]["arguments"]:
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        # If JSON parsing fails, keep as string
                        pass
                formatted_tool_calls.append(tc)
            yield None, formatted_tool_calls, None

    def build_assistant_message(self, text: Optional[str], tool_calls: List[Dict], original_response: Any = None) -> Dict:
        """Build an assistant message with tool calls for OpenAI format."""
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

    def build_tool_result_messages(self, tool_results: List[Dict]) -> List[Dict]:
        """Build individual tool result messages for OpenAI format."""
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
