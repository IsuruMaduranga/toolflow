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

### Enhanced Test Files (New Functionality)
- **`test_structured_output.py`** - Comprehensive tests for structured output functionality using the `parse` method
- **`test_schema_generation_enhanced.py`** - Enhanced schema generation tests for strict vs non-strict modes, internal tools, and complex type handling
- **`test_integration_comprehensive.py`** - Integration tests showing how different features work together

## Test Organization Philosophy

1. **Logical Grouping**: Tests are organized by functionality rather than being scattered
2. **Clear Naming**: Test file names clearly indicate their purpose
3. **Shared Utilities**: Common fixtures and tools are centralized in `conftest.py`
4. **Maintainability**: Smaller, focused test files are easier to maintain and understand
5. **Documentation**: Each test module has clear docstrings explaining its scope
6. **Comprehensive Coverage**: New functionality is thoroughly tested with integration scenarios

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

#### Structured Output (`test_structured_output.py`)
Comprehensive testing of structured output functionality:
- Parse method for sync and async OpenAI wrappers
- Beta API wrapper functionality with strict schemas
- Structured output with tool execution
- Error handling for missing Pydantic, invalid formats, malformed JSON
- Integration with Pydantic BaseModel classes
- Response format tool integration

#### Enhanced Schema Generation (`test_schema_generation_enhanced.py`)
Advanced schema generation features:
- Strict vs non-strict schema generation differences
- Internal tool functionality and name restrictions
- Enhanced schema properties (required fields, title removal, description fallbacks)
- Complex type handling in strict mode (Union, List, Optional)
- Edge cases (async functions, decorator preservation, parameter filtering)
- Pydantic Field integration with Annotated types

#### Integration Tests (`test_integration_comprehensive.py`)
Integration testing of how features work together:
- Structured output with multiple tools
- Beta API with strict schema validation
- Enhanced parameter handling and filtering
- Error handling across feature combinations
- Schema metadata consistency
- Feature interoperability testing

## Current Test Statistics

**Total Tests: 140 passed, 3 skipped, 3 warnings**

### Test Breakdown by Functionality:
- **Core functionality**: ~40 tests
- **Schema generation**: ~25 tests  
- **Structured output**: ~18 tests
- **Enhanced schema features**: ~21 tests
- **Integration tests**: ~13 tests
- **Async functionality**: ~15 tests
- **Error handling**: ~8 tests

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

# Structured output functionality
python -m pytest tests/test_structured_output.py -v

# Enhanced schema generation
python -m pytest tests/test_schema_generation_enhanced.py -v

# Integration tests
python -m pytest tests/test_integration_comprehensive.py -v

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

# Test structured output functionality
python -m pytest tests/test_structured_output.py::TestStructuredOutputUtilities -v

# Test schema generation enhancements
python -m pytest tests/test_schema_generation_enhanced.py::TestStrictSchemaGeneration -v

# Test feature integration
python -m pytest tests/test_integration_comprehensive.py::TestStructuredOutputIntegration -v
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

## New Functionality Test Coverage

The enhanced test suite provides comprehensive coverage for all new toolflow features:

### ✅ Structured Output (Parse Method)
- Parse method functionality for sync/async wrappers
- Beta API wrapper with strict schemas
- Response format tool integration
- Error handling for invalid formats
- Pydantic model integration

### ✅ Enhanced Schema Generation
- Strict vs non-strict schema differences
- Internal tool name protection
- Required fields always present
- Description fallback mechanisms
- Complex type handling improvements

### ✅ Feature Integration
- Multiple features working together
- Parameter filtering and validation
- Schema metadata consistency
- Error handling across features
- Beta API integration

### ✅ Enhanced Error Handling
- Streaming limitations with response formats
- Beta API error scenarios
- Complex validation failures
- Feature interoperability errors

This comprehensive test suite ensures robust coverage of both existing and new functionality, providing confidence in the stability and correctness of the toolflow library.
