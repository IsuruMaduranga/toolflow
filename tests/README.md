# Toolflow Test Suite

A comprehensive test suite for the toolflow library covering all major functionality.

## Test Structure

```
tests/
├── __init__.py              # Test package
├── conftest.py              # Shared fixtures and utilities
├── pytest.ini              # Pytest configuration
├── README.md               # This file
│
├── core/                   # Core functionality tests
│   ├── __init__.py
│   ├── test_decorators.py  # @tool decorator tests
│   ├── test_tool_execution.py  # Tool execution configuration
│   └── test_utils.py       # Core utility functions
│
├── openai/                 # OpenAI provider tests
│   ├── __init__.py
│   ├── test_basic_functionality.py  # Basic OpenAI integration
│   └── test_integration_live.py     # Live OpenAI API tests
│
├── anthropic/              # Anthropic provider tests
│   ├── __init__.py
│   ├── test_basic_functionality.py  # Basic Anthropic integration
│   └── test_integration_live.py     # Live Anthropic API tests
│
├── test_structured_outputs.py  # Structured output tests
├── test_parallel_execution.py  # Parallel execution tests
└── test_end_to_end.py      # End-to-end integration tests
```

## Test Categories

### Unit Tests
- **Core functionality**: Decorator, configuration, utilities
- **Provider wrappers**: Basic wrapping and parameter passing
- **Individual components**: Isolated testing of specific modules

### Integration Tests
- **Tool execution**: Complete tool calling workflows
- **Provider integration**: Full provider-specific functionality
- **Error handling**: Error scenarios and recovery

### End-to-End Tests
- **Complete workflows**: Multi-step tool execution
- **Cross-provider compatibility**: Same tools across providers
- **Real-world scenarios**: Complex use cases

### Live Integration Tests
- **Basic tool calling**: Single and multiple tool executions with real APIs
- **Parallel execution**: Performance testing of concurrent tool calls
- **Async functionality**: Async client operations and async tools
- **Streaming**: Real-time response streaming with and without tools
- **Structured output**: Pydantic model parsing with actual API responses
- **Error handling**: Real error scenarios and graceful recovery
- **Comprehensive workflows**: Multi-tool, multi-step real-world scenarios
- **Performance benchmarks**: Optional performance comparison tests

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Run unit tests only
pytest -m unit

# Run integration tests
pytest -m integration

# Run end-to-end tests
pytest -m e2e

# Run tests for specific provider
pytest -m openai
pytest -m anthropic

# Run parallel execution tests
pytest -m parallel

# Run structured output tests
pytest -m structured

# Skip slow tests
pytest -m "not slow"

# Run live tests (requires API keys)
pytest -m live
```

## Running Tests

### Quick Test Run
```bash
# Run all tests except live tests
pytest -m "not live"

# Run with coverage
pytest --cov=toolflow --cov-report=html

# Run specific test file
pytest tests/core/test_decorators.py

# Run specific test class
pytest tests/openai/test_basic_functionality.py::TestOpenAIWrapper

# Run specific test method
pytest tests/core/test_decorators.py::TestToolDecorator::test_decorator_without_parentheses
```

### Development Workflow
```bash
# Run fast tests during development
pytest -m "unit and not slow" -v

# Run integration tests
pytest -m "integration and not live" -v

# Run full test suite (except live tests)
pytest -m "not live" --tb=short

# Run with parallel execution (faster for large test suites)
pytest -n auto -m "not live"
```

### Live API Testing

Live integration tests make actual API calls to verify end-to-end functionality.

```bash
# Set API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run all live tests (requires API keys)
pytest -m live -v

# Run live tests for specific provider
pytest tests/openai/test_integration_live.py -v
pytest tests/anthropic/test_integration_live.py -v

# Run specific live test categories
pytest -k "test_basic" tests/*/test_integration_live.py -v
pytest -k "test_structured_output" tests/*/test_integration_live.py -v
pytest -k "test_streaming" tests/*/test_integration_live.py -v
pytest -k "test_async" tests/*/test_integration_live.py -v

# Run performance benchmarks (optional, disabled by default)
pytest -k "test_performance" --runxfail tests/*/test_integration_live.py -v
```

**Note**: Live tests consume API credits and may take longer to complete. They are automatically skipped if API keys are not set.

## Test Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for live OpenAI tests
- `ANTHROPIC_API_KEY`: Required for live Anthropic tests
- `TOOLFLOW_TEST_TIMEOUT`: Test timeout in seconds (default: 30)
- `TOOLFLOW_TEST_PARALLEL_WORKERS`: Workers for parallel test execution

### Pytest Configuration
The `pytest.ini` file contains:
- Test discovery patterns
- Marker definitions
- Warning filters
- Default options

## Writing New Tests

### Test Structure
```python
"""
Test module description.
"""
import pytest
from unittest.mock import Mock

from toolflow import tool, from_openai
from tests.conftest import (
    create_openai_response,
    simple_math_tool,
    BASIC_TOOLS
)


class TestFeature:
    """Test a specific feature."""
    
    def test_basic_functionality(self, toolflow_openai_client, mock_openai_client):
        """Test basic functionality."""
        # Setup
        mock_response = create_openai_response(content="Test response")
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Execute
        response = toolflow_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        # Assert
        assert response == "Test response"
        mock_openai_client.chat.completions.create.assert_called_once()
```

### Test Fixtures
Use shared fixtures from `conftest.py`:
- `mock_openai_client`: Mock OpenAI client
- `mock_anthropic_client`: Mock Anthropic client
- `toolflow_openai_client`: Toolflow-wrapped OpenAI client
- `toolflow_anthropic_client`: Toolflow-wrapped Anthropic client
- Tool definitions: `simple_math_tool`, `weather_tool`, etc.
- Helper functions: `create_openai_response`, `create_anthropic_response`

### Mock Helpers
```python
# Create mock responses
response = create_openai_response(content="Hello")
response = create_anthropic_response(content="Hello")

# Create mock tool calls
tool_call = create_openai_tool_call("call_123", "function_name", {"arg": "value"})
tool_call = create_anthropic_tool_call("toolu_123", "function_name", {"arg": "value"})
```

### Test Markers
Add appropriate markers to your tests:
```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test."""
    pass

@pytest.mark.integration
@pytest.mark.openai
def test_openai_integration():
    """OpenAI integration test."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test that takes time."""
    pass

@pytest.mark.live
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="API key required")
def test_live_api():
    """Live API test."""
    pass
```

## Test Coverage

Run with coverage to ensure comprehensive testing:
```bash
# Generate coverage report
pytest --cov=toolflow --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

Target coverage areas:
- Core functionality: >95%
- Provider integrations: >90%
- Error handling: >85%
- Edge cases: >80%

## Continuous Integration

The test suite is designed to run in CI environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest -m "not live" --cov=toolflow --cov-report=xml

- name: Run live tests
  if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    pytest -m live -v
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure toolflow is installed in development mode
   ```bash
   pip install -e .
   ```

2. **Mock issues**: Check that mocks are properly reset between tests
   ```python
   @pytest.fixture(autouse=True)
   def reset_mocks():
       # Reset any global state
       pass
   ```

3. **Timeout issues**: Increase timeout for slow tests
   ```python
   @pytest.mark.timeout(60)
   def test_slow_operation():
       pass
   ```

4. **Parallel execution issues**: Some tests may not be thread-safe
   ```bash
   # Run without parallelization
   pytest -m "not parallel" --tb=short
   ```

### Debug Mode
```bash
# Run with verbose output and stop on first failure
pytest -vvs -x

# Drop into debugger on failure
pytest --pdb

# Show local variables in tracebacks
pytest --tb=long -vv
``` 