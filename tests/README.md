# Test Suite Organization

The test suite is organized by provider and functionality. Each provider has its own directory with comprehensive test coverage.

## Directory Structure

```
tests/
├── anthropic/           # Anthropic provider tests
├── openai/             # OpenAI provider tests
├── conftest.py         # Shared test fixtures and utilities
├── smoke_test.py       # Basic smoke tests
└── README.md           # This file
```

## Provider Test Coverage

### Anthropic Provider (`tests/anthropic/`)
- Core functionality and tool execution
- Async functionality and mixed sync/async execution
- Error handling and edge cases
- Streaming response handling
- Structured output parsing
- Parallel execution
- Schema generation
- Integration tests with live API

### OpenAI Provider (`tests/openai/`)
- Core functionality and tool execution
- Async functionality
- Error handling
- Streaming response handling
- Structured output parsing
- Parallel execution
- Schema generation (basic and enhanced)
- Integration tests (comprehensive and live)

## Running Tests

### All Tests
```bash
python -m pytest tests/
```

### Provider-Specific Tests
```bash
# OpenAI provider tests
python -m pytest tests/openai/ -v

# Anthropic provider tests
python -m pytest tests/anthropic/ -v
```

### Smoke Tests
```bash
python -m pytest tests/smoke_test.py -v
```

## Test Development Guidelines

1. **Provider Organization**: Place provider-specific tests in their respective directories
2. **Shared Fixtures**: Use `conftest.py` for common test utilities and fixtures
3. **Test Categories**: Group related tests into logical categories (core, async, streaming, etc.)
4. **Documentation**: Include clear docstrings explaining test purpose and scope
5. **Error Handling**: Test both success and failure scenarios
6. **Integration**: Include both unit tests and integration tests with live APIs

## Test Statistics

**Total Tests: 140 passed, 3 skipped, 3 warnings**

### Breakdown by Provider
- **OpenAI**: ~70 tests
- **Anthropic**: ~70 tests
