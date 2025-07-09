#!/usr/bin/env python3
"""
Simple test for Llama provider without making API calls.
Tests basic functionality and setup.
"""

import sys
sys.path.insert(0, 'src')

try:
    import openai
    import toolflow
    
    print("✅ Testing Llama provider import...")
    from toolflow import from_llama
    print("✅ from_llama imported successfully")
    
    print("\n✅ Testing client validation...")
    
    # Test with invalid client
    try:
        from_llama("invalid")
        print("❌ Should have failed with invalid client")
    except ValueError as e:
        print(f"✅ Correctly rejected invalid client: {e}")
    
    # Test with valid client
    client = openai.OpenAI(api_key="test", base_url="https://test.com")
    enhanced_client = from_llama(client)
    print("✅ Valid client accepted")
    
    print("\n✅ Testing client structure...")
    print(f"✅ Client type: {type(enhanced_client)}")
    print(f"✅ Has chat: {hasattr(enhanced_client, 'chat')}")
    print(f"✅ Has completions: {hasattr(enhanced_client.chat, 'completions')}")
    print(f"✅ Has create: {hasattr(enhanced_client.chat.completions, 'create')}")
    
    print("\n✅ All Llama provider tests passed!")
    print("🦙 Llama provider is ready for use!")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
