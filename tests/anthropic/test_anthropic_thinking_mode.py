"""
Test Anthropic thinking mode functionality with the toolflow library.

This module tests:
- Basic thinking mode responses
- Streaming with thinking content
- Tool execution combined with thinking
- Structured output with thinking
- Error handling for thinking mode
- Both sync and async operations
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Iterator, AsyncIterator
from pydantic import BaseModel

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

import toolflow
from ..conftest import (
    simple_math_tool,
    calculator_tool,
    weather_tool,
    create_mock_anthropic_tool_call,
    create_mock_anthropic_response,
    create_mock_anthropic_streaming_chunk
)


# Test models for structured output
class MathResult(BaseModel):
    operation: str
    result: float
    explanation: str


class ThinkingAnalysis(BaseModel):
    problem_type: str
    approach: str
    confidence: float


def create_thinking_streaming_chunk(chunk_type: str, **kwargs):
    """Helper to create thinking-specific streaming chunks."""
    chunk = Mock()
    chunk.type = chunk_type
    
    if chunk_type == "content_block_start":
        chunk.index = kwargs.get("index", 0)
        if kwargs.get("thinking", False):
            chunk.content_block = Mock(type="thinking")
        else:
            chunk.content_block = kwargs.get("content_block", Mock(type="text"))
    elif chunk_type == "content_block_delta":
        chunk.index = kwargs.get("index", 0)
        if kwargs.get("thinking", False):
            chunk.delta = Mock(type="thinking_delta", thinking=kwargs.get("text", ""))
        elif kwargs.get("delta"):
            chunk.delta = kwargs.get("delta")
        else:
            chunk.delta = Mock(type="text_delta", text=kwargs.get("text", ""))
    elif chunk_type == "content_block_stop":
        chunk.index = kwargs.get("index", 0)
    elif chunk_type == "message_start":
        chunk.message = kwargs.get("message", Mock())
    elif chunk_type == "message_stop":
        pass
    
    return chunk


class TestAnthropicThinkingModeBasics:
    """Test basic thinking mode functionality."""
    
    def test_thinking_mode_basic_response(self, sync_anthropic_client, mock_anthropic_client):
        """Test basic thinking mode response without streaming."""
        # Mock a response with thinking content
        response_content = """I need to think about this math problem step by step.
        
First, let me convert the speed from km/h to m/s:
- Speed = 120 km / 2 h = 60 km/h
- To convert: 60 km/h × (1000 m/km) × (1 h/3600 s) = 16.67 m/s

The train's speed is 16.67 m/s."""

        mock_response = create_mock_anthropic_response(content=response_content)
        mock_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{
                "role": "user",
                "content": "If a train travels 120 km in 2 hours, what's its speed in m/s?"
            }]
        )
        
        assert response == response_content
        
        # Verify thinking parameters were passed
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["thinking"]["type"] == "enabled"
        assert call_args[1]["thinking"]["budget_tokens"] == 1024

    def test_thinking_mode_budget_validation(self, sync_anthropic_client, mock_anthropic_client):
        """Test that thinking budget is validated properly."""
        # Test budget must be >= 1024
        mock_anthropic_client.messages.create.side_effect = anthropic.BadRequestError(
            "budget_tokens must be at least 1024",
            response=Mock(status_code=400),
            body={"error": {"message": "`thinking.budget_tokens` must be at least 1024"}}
        )
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        
        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                thinking={"type": "enabled", "budget_tokens": 500},  # Too low
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "1024" in str(exc_info.value)

    def test_thinking_mode_budget_vs_max_tokens(self, sync_anthropic_client, mock_anthropic_client):
        """Test that thinking budget must be less than max_tokens."""
        mock_anthropic_client.messages.create.side_effect = anthropic.BadRequestError(
            "budget_tokens must be less than max_tokens",
            response=Mock(status_code=400),
            body={"error": {"message": "`max_tokens` must be greater than `thinking.budget_tokens`"}}
        )
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        
        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                thinking={"type": "enabled", "budget_tokens": 1500},  # Too high
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "max_tokens" in str(exc_info.value) or "budget_tokens" in str(exc_info.value)


class TestAnthropicThinkingModeStreaming:
    """Test thinking mode with streaming functionality."""
    
    def test_thinking_mode_streaming_basic(self, sync_anthropic_client, mock_anthropic_client):
        """Test streaming with thinking content."""
        chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, thinking=True),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text="I need to think about this..."),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text=" Let me work through it step by step."),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("content_block_start", index=1, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=1, text="The answer is 16.67 m/s."),
            create_thinking_streaming_chunk("content_block_stop", index=1),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.return_value = iter(chunks)
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            stream=True,
            messages=[{"role": "user", "content": "Calculate train speed"}]
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(str(chunk))
        
        full_content = "".join(content_parts)
        
        # Should contain thinking tags and content
        assert "<THINKING>" in full_content
        assert "I need to think about this..." in full_content
        assert "Let me work through it step by step." in full_content
        assert "The answer is 16.67 m/s." in full_content

    def test_thinking_mode_streaming_full_response_true(self, mock_anthropic_client):
        """Test streaming with full_response=True returns raw events."""
        chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, thinking=True),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text="Thinking..."),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.return_value = iter(chunks)
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=True)
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            stream=True,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        chunk_list = list(stream)
        
        # Should return raw chunk objects
        assert len(chunk_list) == 5
        assert chunk_list[0].type == "message_start"
        assert chunk_list[1].type == "content_block_start"
        assert chunk_list[1].content_block.type == "thinking"
        assert chunk_list[2].type == "content_block_delta"
        assert chunk_list[2].delta.type == "thinking_delta"
        assert chunk_list[2].delta.thinking == "Thinking..."

    def test_thinking_mode_streaming_with_tools(self, sync_anthropic_client, mock_anthropic_client):
        """Test streaming with both thinking and tool execution."""
        # Initial stream with thinking and tool call
        tool_call = Mock(type="tool_use", id="call_123", name="simple_math_tool")
        
        initial_chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, thinking=True),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text="I need to calculate 5 + 3..."),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text=" I'll use the math tool."),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("content_block_start", index=1, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=1, text="I'll calculate that for you."),
            create_thinking_streaming_chunk("content_block_stop", index=1),
            create_thinking_streaming_chunk("content_block_start", index=2, content_block=tool_call),
            create_thinking_streaming_chunk("content_block_delta", index=2, 
                                          delta=Mock(type="input_json_delta", partial_json='{"a": 5.0, "b": 3.0}')),
            create_thinking_streaming_chunk("content_block_stop", index=2),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        # Follow-up stream after tool execution
        follow_up_chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=0, text="The result is 8.0"),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.side_effect = [
            iter(initial_chunks),
            iter(follow_up_chunks)
        ]
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=3000,
            thinking={"type": "enabled", "budget_tokens": 1500},
            tools=[simple_math_tool],
            stream=True,
            messages=[{"role": "user", "content": "What is 5 + 3?"}]
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(str(chunk))
        
        full_content = "".join(content_parts)
        
        # Should contain thinking content, regular content, and tool results
        assert "<THINKING>" in full_content
        assert "I need to calculate 5 + 3..." in full_content
        assert "I'll use the math tool." in full_content
        assert "I'll calculate that for you." in full_content
        assert "The result is 8.0" in full_content


class TestAnthropicThinkingModeWithTools:
    """Test thinking mode combined with tool execution."""
    
    def test_thinking_mode_single_tool_call(self, sync_anthropic_client, mock_anthropic_client):
        """Test thinking mode with a single tool call."""
        tool_call = create_mock_anthropic_tool_call("call_1", "simple_math_tool", {"a": 10.0, "b": 5.0})
        
        # Initial response with tool call
        initial_response = create_mock_anthropic_response(tool_calls=[tool_call])
        
        # Follow-up response after tool execution
        follow_up_response = create_mock_anthropic_response(content="The calculation result is 15.0")
        
        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            follow_up_response
        ]
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=3000,
            thinking={"type": "enabled", "budget_tokens": 1500},
            tools=[simple_math_tool],
            messages=[{"role": "user", "content": "Add 10 and 5"}]
        )
        
        assert response == "The calculation result is 15.0"
        
        # Verify two calls were made (initial + follow-up)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify thinking was passed to both calls
        first_call = mock_anthropic_client.messages.create.call_args_list[0]
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        
        assert first_call[1]["thinking"]["type"] == "enabled"
        assert second_call[1]["thinking"]["type"] == "enabled"

    def test_thinking_mode_multiple_tool_calls(self, sync_anthropic_client, mock_anthropic_client):
        """Test thinking mode with multiple tool calls."""
        tool_call_1 = create_mock_anthropic_tool_call("call_1", "simple_math_tool", {"a": 10.0, "b": 5.0})
        tool_call_2 = create_mock_anthropic_tool_call("call_2", "calculator_tool", 
                                                     {"operation": "multiply", "a": 3.0, "b": 4.0})
        
        # Initial response with multiple tool calls
        initial_response = create_mock_anthropic_response(tool_calls=[tool_call_1, tool_call_2])
        
        # Follow-up response
        follow_up_response = create_mock_anthropic_response(
            content="Addition: 15.0, Multiplication: 12.0"
        )
        
        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            follow_up_response
        ]
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=3000,
            thinking={"type": "enabled", "budget_tokens": 1500},
            tools=[simple_math_tool, calculator_tool],
            parallel_tool_execution=True,
            messages=[{"role": "user", "content": "Add 10+5 and multiply 3*4"}]
        )
        
        assert response == "Addition: 15.0, Multiplication: 12.0"
        assert mock_anthropic_client.messages.create.call_count == 2


class TestAnthropicThinkingModeStructuredOutput:
    """Test thinking mode with structured output."""
    
    def test_thinking_mode_with_pydantic_model(self, sync_anthropic_client, mock_anthropic_client):
        """Test thinking mode with Pydantic structured output."""
        # Create a proper MathResult instance
        structured_response = MathResult(
            operation="25 * 4 + 10",
            result=110.0,
            explanation="Following order of operations: 25 * 4 = 100, then 100 + 10 = 110"
        )
        
        # Mock the response with tool call for structured output
        tool_call = create_mock_anthropic_tool_call(
            "call_1", 
            "final_response_tool_internal", 
            structured_response.model_dump()
        )
        mock_response = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response.stop_reason = "tool_use"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        result = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            response_format=MathResult,
            messages=[{"role": "user", "content": "Calculate 25 * 4 + 10"}]
        )
        
        assert isinstance(result, MathResult)
        assert result.operation == "25 * 4 + 10"
        assert result.result == 110.0
        assert "order of operations" in result.explanation.lower()
        
        # Verify thinking and response_format were passed
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["thinking"]["type"] == "enabled"
        # Note: response_format handling may be internal to toolflow

    def test_thinking_mode_with_complex_structured_output(self, sync_anthropic_client, mock_anthropic_client):
        """Test thinking mode with more complex structured output."""
        # Create a proper ThinkingAnalysis instance
        structured_response = ThinkingAnalysis(
            problem_type="mathematical_reasoning",
            approach="step_by_step_calculation",
            confidence=0.95
        )
        
        # Mock the response with tool call for structured output
        tool_call = create_mock_anthropic_tool_call(
            "call_1", 
            "final_response_tool_internal", 
            structured_response.model_dump()
        )
        mock_response = create_mock_anthropic_response(tool_calls=[tool_call])
        mock_response.stop_reason = "tool_use"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        result = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            response_format=ThinkingAnalysis,
            messages=[{"role": "user", "content": "Analyze this problem"}]
        )
        
        assert isinstance(result, ThinkingAnalysis)
        assert result.problem_type == "mathematical_reasoning"
        assert result.approach == "step_by_step_calculation"
        assert result.confidence == 0.95


class TestAnthropicThinkingModeAsync:
    """Test async thinking mode functionality."""
    
    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_thinking_mode_basic(self, mock_async_anthropic_client):
        """Test basic async thinking mode."""
        response_content = "I need to think about this calculation... The result is 78.54 square units."
        
        mock_response = create_mock_anthropic_response(content=response_content)
        mock_async_anthropic_client.messages.create.return_value = mock_response
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client)
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "Calculate circle area with radius 5"}]
        )
        
        assert response == response_content
        
        # Verify thinking parameters were passed
        call_args = mock_async_anthropic_client.messages.create.call_args
        assert call_args[1]["thinking"]["type"] == "enabled"
        assert call_args[1]["thinking"]["budget_tokens"] == 1024

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_thinking_mode_with_tools(self, mock_async_anthropic_client):
        """Test async thinking mode with tool execution."""
        tool_call = create_mock_anthropic_tool_call("call_1", "simple_math_tool", {"a": 5.0, "b": 3.0})
        
        initial_response = create_mock_anthropic_response(tool_calls=[tool_call])
        follow_up_response = create_mock_anthropic_response(content="The result is 8.0")
        
        mock_async_anthropic_client.messages.create.side_effect = [
            initial_response,
            follow_up_response
        ]
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client)
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            tools=[simple_math_tool],
            messages=[{"role": "user", "content": "Add 5 and 3"}]
        )
        
        assert response == "The result is 8.0"
        assert mock_async_anthropic_client.messages.create.call_count == 2

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_async_thinking_mode_streaming(self, mock_async_anthropic_client):
        """Test async streaming with thinking mode."""
        chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, thinking=True),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text="Thinking async..."),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("content_block_start", index=1, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=1, text="Async result: 42"),
            create_thinking_streaming_chunk("content_block_stop", index=1),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        async def async_iter_chunks():
            for chunk in chunks:
                yield chunk
        
        mock_async_anthropic_client.messages.create.return_value = async_iter_chunks()
        
        client = toolflow.from_anthropic_async(mock_async_anthropic_client, full_response=False)
        stream = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            stream=True,
            messages=[{"role": "user", "content": "Calculate something"}]
        )
        
        content_parts = []
        async for chunk in stream:
            if chunk:
                content_parts.append(str(chunk))
        
        full_content = "".join(content_parts)
        
        assert "<THINKING>" in full_content
        assert "Thinking async..." in full_content
        assert "Async result: 42" in full_content


class TestAnthropicThinkingModeErrorHandling:
    """Test error handling for thinking mode."""
    
    def test_thinking_mode_invalid_model(self, sync_anthropic_client, mock_anthropic_client):
        """Test thinking mode with model that doesn't support thinking."""
        mock_anthropic_client.messages.create.side_effect = anthropic.BadRequestError(
            "Model does not support thinking",
            response=Mock(status_code=400),
            body={"error": {"message": "The model claude-3-haiku-20240307 does not support thinking"}}
        )
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        
        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model="claude-3-haiku-20240307",  # Doesn't support thinking
                max_tokens=2000,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "thinking" in str(exc_info.value).lower()

    def test_thinking_mode_with_tool_choice_required(self, sync_anthropic_client, mock_anthropic_client):
        """Test that thinking mode doesn't work with tool_choice required."""
        mock_anthropic_client.messages.create.side_effect = anthropic.BadRequestError(
            "Thinking mode only supports tool_choice auto or none",
            response=Mock(status_code=400),
            body={"error": {"message": "When using thinking, `tool_choice` must be 'auto' or 'none'"}}
        )
        
        client = toolflow.from_anthropic(mock_anthropic_client)
        
        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                thinking={"type": "enabled", "budget_tokens": 1024},
                tools=[simple_math_tool],
                tool_choice={"type": "tool", "name": "simple_math_tool"},  # Not supported with thinking
                messages=[{"role": "user", "content": "Test"}]
            )
        
        assert "tool_choice" in str(exc_info.value).lower()

    def test_thinking_mode_streaming_interruption(self, sync_anthropic_client, mock_anthropic_client):
        """Test handling of interrupted thinking mode streaming."""
        def failing_stream():
            yield create_thinking_streaming_chunk("message_start", message=Mock())
            yield create_thinking_streaming_chunk("content_block_start", index=0, thinking=True)
            yield create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, text="Starting to think...")
            # Simulate interruption
            mock_request = Mock()
            raise anthropic.APIConnectionError(message="Connection lost", request=mock_request)
        
        mock_anthropic_client.messages.create.return_value = failing_stream()
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            stream=True,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        content_parts = []
        with pytest.raises(anthropic.APIConnectionError):
            for chunk in stream:
                if chunk:
                    content_parts.append(str(chunk))
        
        # Should have collected partial content before failure
        partial_content = "".join(content_parts)
        assert "<THINKING>" in partial_content
        assert "Starting to think..." in partial_content


class TestAnthropicThinkingModeIntegration:
    """Integration tests combining multiple thinking mode features."""
    
    def test_thinking_mode_complex_workflow(self, sync_anthropic_client, mock_anthropic_client):
        """Test complex workflow with thinking, tools, and streaming."""
        # Initial stream with thinking and multiple tool calls
        tool_call_1 = Mock(type="tool_use", id="call_1", name="calculator_tool")
        tool_call_2 = Mock(type="tool_use", id="call_2", name="weather_tool")
        
        initial_chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, thinking=True),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True, 
                                          text="I need to calculate sales and check weather..."),
            create_thinking_streaming_chunk("content_block_delta", index=0, thinking=True,
                                          text=" Let me use the calculator and weather tools."),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("content_block_start", index=1, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=1, text="I'll help you with both calculations."),
            create_thinking_streaming_chunk("content_block_stop", index=1),
            create_thinking_streaming_chunk("content_block_start", index=2, content_block=tool_call_1),
            create_thinking_streaming_chunk("content_block_delta", index=2,
                                          delta=Mock(type="input_json_delta", 
                                                   partial_json='{"operation": "add", "a": 100.0, "b": 200.0}')),
            create_thinking_streaming_chunk("content_block_stop", index=2),
            create_thinking_streaming_chunk("content_block_start", index=3, content_block=tool_call_2),
            create_thinking_streaming_chunk("content_block_delta", index=3,
                                          delta=Mock(type="input_json_delta", 
                                                   partial_json='{"city": "San Francisco"}')),
            create_thinking_streaming_chunk("content_block_stop", index=3),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        # Follow-up stream after tool execution
        follow_up_chunks = [
            create_thinking_streaming_chunk("message_start", message=Mock()),
            create_thinking_streaming_chunk("content_block_start", index=0, content_block=Mock(type="text")),
            create_thinking_streaming_chunk("content_block_delta", index=0, 
                                          text="Total sales: 300.0. Weather in San Francisco: Sunny, 72°F"),
            create_thinking_streaming_chunk("content_block_stop", index=0),
            create_thinking_streaming_chunk("message_stop")
        ]
        
        mock_anthropic_client.messages.create.side_effect = [
            iter(initial_chunks),
            iter(follow_up_chunks)
        ]
        
        client = toolflow.from_anthropic(mock_anthropic_client, full_response=False)
        stream = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            tools=[calculator_tool, weather_tool],
            stream=True,
            parallel_tool_execution=True,
            messages=[{"role": "user", "content": "Add 100+200 and get weather for San Francisco"}]
        )
        
        content_parts = []
        for chunk in stream:
            if chunk:
                content_parts.append(str(chunk))
        
        full_content = "".join(content_parts)
        
        # Verify all components are present
        assert "<THINKING>" in full_content
        assert "I need to calculate sales and check weather..." in full_content
        assert "Let me use the calculator and weather tools." in full_content
        assert "I'll help you with both calculations." in full_content
        assert "Total sales: 300.0" in full_content
        assert "Weather in San Francisco: Sunny, 72°F" in full_content
        
        # Verify proper call sequence
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify thinking was preserved across calls
        for call in mock_anthropic_client.messages.create.call_args_list:
            assert call[1]["thinking"]["type"] == "enabled"
            assert call[1]["thinking"]["budget_tokens"] == 2000 