# Test Suite Organization

The test suite has been restructured to be more organized and maintainable. Here's the new structure:

## Test Files

### Core Test Files
- **`test.py`** - Basic smoke tests to verify fundamental functionality
- **`test_core_functionality.py`** - Core decorator functionality, basic tool execution, imports, and basic OpenAI integration
- **`test_async_functionality.py`** - Async-specific functionality including async tools, async clients, and mixed sync/async execution
- **`test_schema_generation.py`** - Schema generation, type annotation handling, parameter validation, and metadata attachment
- **`test_error_handling.py`** - Error handling scenarios, invalid arguments, unknown tools, and edge cases (note: some tests are based on current implementation behavior)
- **`conftest.py`** - Shared fixtures, common test tools, and utility functions

## Test Organization Philosophy

1. **Logical Grouping**: Tests are organized by functionality rather than being scattered
2. **Clear Naming**: Test file names clearly indicate their purpose
3. **Shared Utilities**: Common fixtures and tools are centralized in `conftest.py`
4. **Maintainability**: Smaller, focused test files are easier to maintain and understand
5. **Documentation**: Each test module has clear docstrings explaining its scope

### Test Categories

#### Smoke Tests (`test.py`)
Quick verification that basic functionality works:
- Import functionality
- Basic decorator usage
- Core tool creation

#### Core Functionality (`test_core_functionality.py`)
Essential functionality tests:
- `@tool` decorator behavior
- Function preservation and metadata attachment  
- Basic tool execution
- Import verification
- OpenAI integration basics

#### Async Functionality (`test_async_functionality.py`)
Async-specific behavior:
- Async tool decoration and execution
- Async client behavior
- Mixed sync/async tool handling
- Async error handling
- Async limits and constraints

#### Schema Generation (`test_schema_generation.py`)
Schema and metadata handling:
- Basic schema structure
- Complex type handling (Union, Any, nested types)
- Parameter handling (defaults, keyword-only, etc.)
- Schema properties and validation
- Utility function testing

#### Error Handling (`test_error_handling.py`)
Error scenarios and edge cases:
- Tool execution errors
- Invalid arguments and JSON parsing
- Unknown tool calls
- Import error simulation
- Edge case validation

## Usage

### Running All Tests
```bash
python -m pytest tests/
```

### Running Specific Test Categories
```bash
# Basic smoke tests
python -m pytest tests/test.py -v

# Core functionality only
python -m pytest tests/test_core_functionality.py -v

# Async functionality only
python -m pytest tests/test_async_functionality.py -v

# Schema generation only  
python -m pytest tests/test_schema_generation.py -v

# Error handling only
python -m pytest tests/test_error_handling.py -v

# Legacy tests only
python -m pytest tests/test_edge_cases.py tests/test_async.py tests/test_parallel.py -v
```

### Running Specific Test Classes
```bash
# Test just the tool decorator
python -m pytest tests/test_core_functionality.py::TestToolDecorator -v

# Test just async tool execution
python -m pytest tests/test_async_functionality.py::TestAsyncToolExecution -v

# Test just basic schema generation
python -m pytest tests/test_schema_generation.py::TestBasicSchemaGeneration -v
```

## Test Development Guidelines

### When Adding New Tests
1. **Choose the Right File**: Place tests in the most appropriate module based on functionality
2. **Use Shared Fixtures**: Leverage `conftest.py` fixtures for common setup
3. **Follow Naming Conventions**: Use descriptive test names that explain what is being tested
4. **Group Related Tests**: Use test classes to group related functionality
5. **Add Documentation**: Include docstrings explaining test purpose and scope

### Test Class Organization
Each test file uses classes to group related tests:
- `TestBasicFunctionality` - Core/basic behavior tests
- `TestAdvancedFeatures` - Complex/advanced behavior tests  
- `TestErrorHandling` - Error scenarios and edge cases
- `TestConfiguration` - Configuration and parameter tests

### Fixture Usage
Common fixtures are available from `conftest.py`:
- `mock_openai_client` - Mock sync OpenAI client
- `mock_async_openai_client` - Mock async OpenAI client
- `sync_toolflow_client` - Pre-configured sync toolflow client
- `async_toolflow_client` - Pre-configured async toolflow client
- Common test tools: `simple_math_tool`, `divide_tool`, `async_math_tool`, etc.
- Helper functions: `create_mock_tool_call`, `create_mock_response`
