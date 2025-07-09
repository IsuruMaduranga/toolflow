# Gemini Examples

This directory contains examples and utilities for using Google Gemini with toolflow.

## Quick Start Scripts

### 🔍 Model Discovery

- **`models_summary.py`** - Quick reference of recommended Gemini models
- **`gemini_models_check.py`** - Comprehensive script to list all available models and test API connectivity
- **`list_models.py`** - Detailed model listing with recommendations by use case

### 📚 Usage Examples

- **`sync_basic.py`** - Basic synchronous usage with tool calling
- **`sync_structured_outputs.py`** - Structured outputs with Pydantic models
- **`sync_streaming.py`** - Streaming responses
- **`sync_parallel.py`** - Parallel tool execution
- **`async.py`** - Asynchronous operations
- **`async_parallel.py`** - Async with parallel tool execution

## Recommended Models

Based on our testing, here are the best models for different use cases:

| Model | Best For | Context | Output | Priority |
|-------|----------|---------|--------|----------|
| `gemini-2.0-flash` | Most applications | 1M tokens | 8K tokens | ⭐⭐⭐⭐⭐ |
| `gemini-1.5-flash` | Stable production | 1M tokens | 8K tokens | ⭐⭐⭐⭐ |
| `gemini-1.5-pro` | Complex reasoning | 2M tokens | 8K tokens | ⭐⭐⭐⭐ |
| `gemini-2.5-flash` | Latest features | 1M tokens | 64K tokens | ⭐⭐⭐⭐⭐ |
| `gemini-1.5-flash-8b` | Cost-effective | 1M tokens | 8K tokens | ⭐⭐⭐ |

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install google-generativeai toolflow
   ```

2. **Set API key:**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

3. **Basic usage:**
   ```python
   import os
   import toolflow
   import google.generativeai as genai

   # Configure
   genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
   
   # Create client
   model = genai.GenerativeModel('gemini-2.0-flash')
   client = toolflow.from_gemini(model)
   
   # Generate content
   response = client.generate_content("Hello, world!")
   print(response)
   ```

## Running Examples

### Check Available Models
```bash
python gemini_models_check.py
```

### Quick Model Summary
```bash
python models_summary.py
```

### Basic Tool Calling
```bash
python sync_basic.py
```

### Structured Outputs
```bash
python sync_structured_outputs.py
```

### Streaming
```bash
python sync_streaming.py
```

## Features Supported

- ✅ **Tool Calling** - Call Python functions from Gemini
- ✅ **Structured Outputs** - Get Pydantic models back
- ✅ **Streaming** - Real-time response streaming
- ✅ **Async Support** - Full async/await support
- ✅ **Parallel Execution** - Run multiple tools simultaneously
- ✅ **Error Handling** - Graceful error management
- ✅ **Context Management** - Large context window support
- ✅ **Multimodal** - Text, images, and other media

## Recent Fixes

The Gemini implementation has been enhanced with:

- **Fixed MaxToolCallsError** - Resolved infinite tool calling loops
- **Enhanced Structured Outputs** - Better support for Pydantic models
- **Improved Tool Schema Conversion** - Proper Gemini format handling
- **Better Error Messages** - More helpful debugging information

## Need Help?

- Check the examples in this directory
- Run `python gemini_models_check.py` to test your setup
- See the main toolflow documentation
- All examples include detailed comments and error handling

## API Key Setup

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it as an environment variable or in a `.env` file.
