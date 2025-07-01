"""
End-to-end integration tests for toolflow.

These tests verify that the entire toolflow system works together correctly,
testing the complete flow from client creation through tool execution to final response.
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from typing import List
from pydantic import BaseModel

import toolflow
from toolflow import from_openai, from_anthropic, tool
from tests.conftest import (
    create_openai_response,
    create_openai_tool_call,
    create_openai_structured_response,
    create_anthropic_response,
    create_anthropic_tool_call,
    create_anthropic_structured_response,
    simple_math_tool,
    weather_tool,
    multiply_tool,
    Person,
    TeamAnalysis
)


class CalculationResult(BaseModel):
    """Result of a calculation."""
    operation: str
    operands: List[float]
    result: float
    timestamp: str


class TestCompleteWorkflows:
    """Test complete workflows from start to finish."""
    
    def test_openai_complete_workflow(self, mock_openai_client):
        """Test a complete workflow with OpenAI: tools + structured output + parallel execution."""
        client = from_openai(mock_openai_client)
        
        # Define a calculation tool
        @tool
        def calculate_and_format(a: float, b: float, operation: str) -> str:
            """Calculate and format the result."""
            import datetime
            if operation == "add":
                result = a + b
            elif operation == "multiply":
                result = a * b
            else:
                result = 0
            
            return f'{{"operation": "{operation}", "operands": [{a}, {b}], "result": {result}, "timestamp": "{datetime.datetime.now().isoformat()}"}}'
        
        # First response: model decides to use tools
        tool_calls = [
            create_openai_tool_call("call_1", "calculate_and_format", {"a": 10, "b": 5, "operation": "add"}),
            create_openai_tool_call("call_2", "calculate_and_format", {"a": 3, "b": 4, "operation": "multiply"})
        ]
        mock_response_1 = create_openai_response(tool_calls=tool_calls)
        
        # Second response: model provides structured output
        structured_data = {
            "operation": "combined",
            "operands": [10, 5, 3, 4],
            "result": 27,
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_response_2 = create_openai_structured_response(structured_data)
        
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        # Execute complete workflow
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 10+5 and 3*4, then combine results"}],
            tools=[calculate_and_format],
            response_format=CalculationResult,
            parallel_tool_execution=True
        )
        
        # Verify structured output
        assert isinstance(response, CalculationResult)
        assert response.operation == "combined"
        assert 10 in response.operands
        assert response.result == 27
        
        # Verify both API calls were made
        assert mock_openai_client.chat.completions.create.call_count == 2
    
    def test_anthropic_complete_workflow(self, mock_anthropic_client):
        """Test a complete workflow with Anthropic."""
        client = from_anthropic(mock_anthropic_client)
        
        # Define team analysis tools
        @tool
        def analyze_person(name: str, age: int, skills: str) -> str:
            """Analyze a person's profile."""
            skills_list = [s.strip() for s in skills.split(",")]
            return f'{{"name": "{name}", "age": {age}, "skills": {skills_list}}}'
        
        # First response: model uses tools
        tool_calls = [
            create_anthropic_tool_call("toolu_1", "analyze_person", {"name": "Alice", "age": 25, "skills": "Python, React"}),
            create_anthropic_tool_call("toolu_2", "analyze_person", {"name": "Bob", "age": 30, "skills": "Go, Docker"})
        ]
        mock_response_1 = create_anthropic_response(tool_calls=tool_calls)
        
        # Second response: structured team analysis
        team_analysis_data = {
            "people": [
                {"name": "Alice", "age": 25, "skills": ["Python", "React"]},
                {"name": "Bob", "age": 30, "skills": ["Go", "Docker"]}
            ],
            "average_age": 27.5,
            "top_skills": ["Python", "React", "Go", "Docker"]
        }
        mock_response_2 = create_anthropic_structured_response(team_analysis_data)
        
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=200,
            messages=[{"role": "user", "content": "Analyze Alice (25, Python/React) and Bob (30, Go/Docker)"}],
            tools=[analyze_person],
            response_format=TeamAnalysis,
            parallel_tool_execution=True
        )
        
        assert isinstance(response, TeamAnalysis)
        assert len(response.people) == 2
        assert response.average_age == 27.5
        assert "Python" in response.top_skills
    
    def test_multi_round_tool_execution(self, mock_openai_client):
        """Test workflow with multiple rounds of tool execution."""
        client = from_openai(mock_openai_client)
        
        @tool
        def get_base_number() -> str:
            """Get a base number."""
            return "10"
        
        @tool
        def multiply_by_factor(number: str, factor: int) -> str:
            """Multiply a number by a factor."""
            return str(int(number) * factor)
        
        # Round 1: Get base number
        tool_call_1 = create_openai_tool_call("call_1", "get_base_number", {})
        mock_response_1 = create_openai_response(tool_calls=[tool_call_1])
        
        # Round 2: Multiply by factor
        tool_call_2 = create_openai_tool_call("call_2", "multiply_by_factor", {"number": "10", "factor": 3})
        mock_response_2 = create_openai_response(tool_calls=[tool_call_2])
        
        # Round 3: Final response
        mock_response_3 = create_openai_response(content="The final result is 30")
        
        mock_openai_client.chat.completions.create.side_effect = [
            mock_response_1, mock_response_2, mock_response_3
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Get a base number and multiply it by 3"}],
            tools=[get_base_number, multiply_by_factor],
            max_tool_calls=5  # Allow multiple rounds
        )
        
        assert response == "The final result is 30"
        assert mock_openai_client.chat.completions.create.call_count == 3


class TestErrorRecoveryWorkflows:
    """Test workflows with error recovery."""
    
    def test_tool_error_recovery(self, mock_openai_client):
        """Test workflow with tool error and recovery."""
        client = from_openai(mock_openai_client)
        
        @tool
        def unreliable_tool(should_fail: bool) -> str:
            """An unreliable tool."""
            if should_fail:
                raise ValueError("Tool failed as expected")
            return "Tool succeeded"
        
        @tool
        def backup_tool() -> str:
            """A backup tool."""
            return "Backup result"
        
        # Round 1: Tool fails
        tool_call_1 = create_openai_tool_call("call_1", "unreliable_tool", {"should_fail": True})
        mock_response_1 = create_openai_response(tool_calls=[tool_call_1])
        
        # Round 2: Use backup tool
        tool_call_2 = create_openai_tool_call("call_2", "backup_tool", {})
        mock_response_2 = create_openai_response(tool_calls=[tool_call_2])
        
        # Round 3: Final response
        mock_response_3 = create_openai_response(content="Used backup tool successfully")
        
        mock_openai_client.chat.completions.create.side_effect = [
            mock_response_1, mock_response_2, mock_response_3
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Try the unreliable tool, use backup if needed"}],
            tools=[unreliable_tool, backup_tool]
        )
        
        assert response == "Used backup tool successfully"
        
        # Check that error was properly communicated
        call_args = mock_openai_client.chat.completions.create.call_args_list
        second_call_messages = call_args[1][1]['messages']
        
        # Find the tool error message
        tool_messages = [msg for msg in second_call_messages if msg['role'] == 'tool']
        assert any("Tool failed as expected" in msg['content'] for msg in tool_messages)
    
    def test_graceful_degradation(self, mock_openai_client):
        """Test graceful degradation when tools are unavailable."""
        client = from_openai(mock_openai_client)
        
        # Response where model doesn't call any tools
        mock_response = create_openai_response(
            content="I don't have access to specific tools, but I can provide general information."
        )
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Get current weather"}],
            tools=[weather_tool]  # Tool is available but model chooses not to use it
        )
        
        assert "general information" in response
        assert mock_openai_client.chat.completions.create.call_count == 1


class TestConfigurationWorkflows:
    """Test workflows with different configurations."""
    
    def test_high_concurrency_workflow(self, mock_openai_client):
        """Test workflow with high concurrency settings."""
        # Configure for high concurrency
        original_workers = toolflow.get_max_workers()
        toolflow.set_max_workers(8)
        
        try:
            client = from_openai(mock_openai_client)
            
            # Create multiple tools
            tools = []
            tool_calls = []
            for i in range(6):
                @tool
                def concurrent_tool(task_id: int = i) -> str:
                    """Concurrent task."""
                    time.sleep(0.01)
                    return f"Task {task_id} completed"
                
                # Give each tool a unique name
                concurrent_tool.__name__ = f"concurrent_tool_{i}"
                tools.append(concurrent_tool)
                tool_calls.append(
                    create_openai_tool_call(f"call_{i}", f"concurrent_tool_{i}", {"task_id": i})
                )
            
            mock_response_1 = create_openai_response(tool_calls=tool_calls)
            mock_response_2 = create_openai_response(content="All concurrent tasks completed")
            mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Run all concurrent tasks"}],
                tools=tools,
                parallel_tool_execution=True
            )
            execution_time = time.time() - start_time
            
            assert response == "All concurrent tasks completed"
            # With 8 workers and 6 tasks of 0.01s each, should complete quickly
            assert execution_time < 0.05
            
        finally:
            toolflow.set_max_workers(original_workers)
    
    def test_custom_response_modes(self, mock_openai_client):
        """Test different response mode configurations."""
        # Test simplified response mode (default)
        client_simple = from_openai(mock_openai_client, full_response=False)
        
        mock_response = create_openai_response(content="Simple response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        response = client_simple.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response == "Simple response"
        
        # Reset mock
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test full response mode
        client_full = from_openai(mock_openai_client, full_response=True)
        
        response = client_full.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response == mock_response
        assert hasattr(response, 'choices')


class TestProviderCompatibility:
    """Test compatibility across different providers."""
    
    def test_same_tools_different_providers(self, mock_openai_client, mock_anthropic_client):
        """Test using the same tools with different providers."""
        # Define shared tools
        @tool
        def shared_calculation(x: int, y: int) -> str:
            """Shared calculation tool."""
            return str(x + y)
        
        # Test with OpenAI
        openai_client = from_openai(mock_openai_client)
        
        tool_call = create_openai_tool_call("call_1", "shared_calculation", {"x": 5, "y": 3})
        mock_response_1 = create_openai_response(tool_calls=[tool_call])
        mock_response_2 = create_openai_response(content="OpenAI result: 8")
        mock_openai_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 5 + 3"}],
            tools=[shared_calculation]
        )
        
        # Test with Anthropic
        anthropic_client = from_anthropic(mock_anthropic_client)
        
        tool_call = create_anthropic_tool_call("toolu_1", "shared_calculation", {"x": 5, "y": 3})
        mock_response_1 = create_anthropic_response(tool_calls=[tool_call])
        mock_response_2 = create_anthropic_response(content="Anthropic result: 8")
        mock_anthropic_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        
        anthropic_response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Calculate 5 + 3"}],
            tools=[shared_calculation]
        )
        
        # Both should work and produce results
        assert "8" in openai_response
        assert "8" in anthropic_response
    
    def test_provider_specific_features(self, mock_openai_client, mock_anthropic_client):
        """Test that provider-specific features work correctly."""
        # OpenAI with function calling
        openai_client = from_openai(mock_openai_client)
        
        mock_response = create_openai_response(content="OpenAI specific response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            top_p=0.9
        )
        
        # Anthropic with system message
        anthropic_client = from_anthropic(mock_anthropic_client)
        
        mock_response = create_anthropic_response(content="Anthropic specific response")
        mock_anthropic_client.messages.create.return_value = mock_response
        
        anthropic_response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant",
            temperature=0.7
        )
        
        assert openai_response == "OpenAI specific response"
        assert anthropic_response == "Anthropic specific response"
        
        # Verify provider-specific parameters were passed through
        openai_call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert openai_call_args['temperature'] == 0.7
        assert openai_call_args['top_p'] == 0.9
        
        anthropic_call_args = mock_anthropic_client.messages.create.call_args[1]
        assert anthropic_call_args['system'] == "You are a helpful assistant"
        assert anthropic_call_args['temperature'] == 0.7


class TestRealWorldScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_data_analysis_workflow(self, mock_openai_client):
        """Test a data analysis workflow."""
        client = from_openai(mock_openai_client)
        
        @tool
        def load_data(source: str) -> str:
            """Load data from source."""
            return f"Loaded 100 records from {source}"
        
        @tool
        def analyze_data(data_description: str) -> str:
            """Analyze loaded data."""
            return "Analysis complete: Mean=50, Std=15, Outliers=3"
        
        @tool
        def generate_report(analysis: str) -> str:
            """Generate analysis report."""
            return "Report generated: Key insights identified"
        
        # Multi-step workflow
        responses = [
            # Step 1: Load data
            create_openai_response(tool_calls=[
                create_openai_tool_call("call_1", "load_data", {"source": "database"})
            ]),
            # Step 2: Analyze data
            create_openai_response(tool_calls=[
                create_openai_tool_call("call_2", "analyze_data", {"data_description": "Loaded 100 records from database"})
            ]),
            # Step 3: Generate report
            create_openai_response(tool_calls=[
                create_openai_tool_call("call_3", "generate_report", {"analysis": "Analysis complete: Mean=50, Std=15, Outliers=3"})
            ]),
            # Final response
            create_openai_response(content="Data analysis workflow completed successfully")
        ]
        
        mock_openai_client.chat.completions.create.side_effect = responses
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Perform complete data analysis on database"}],
            tools=[load_data, analyze_data, generate_report],
            max_tool_calls=10
        )
        
        assert "completed successfully" in response
        assert mock_openai_client.chat.completions.create.call_count == 4 