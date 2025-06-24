#!/usr/bin/env python3
"""
Test script to verify response cleaning works with qwen3:4b model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, clean_response

def test_response_cleaning():
    """Test the response cleaning function"""
    print("Testing response cleaning...")
    
    # Test case 1: Response with <think> tags
    test_response1 = """<think>
Okay, the user asked "Test availability". I need to figure out what they mean by that. It could be about testing a product, a test environment, or maybe something else. Let me think.
</think>

The system is available and ready to assist you."""
    
    cleaned1 = clean_response(test_response1)
    print(f"Original: {test_response1}")
    print(f"Cleaned: {cleaned1}")
    print()
    
    # Test case 2: Response with only <think> tags
    test_response2 = """<think>
This is just thinking content with no actual response.
</think>"""
    
    cleaned2 = clean_response(test_response2)
    print(f"Original: {test_response2}")
    print(f"Cleaned: {cleaned2}")
    print()
    
    # Test case 3: Normal response without tags
    test_response3 = "This is a normal response without any special tags."
    
    cleaned3 = clean_response(test_response3)
    print(f"Original: {test_response3}")
    print(f"Cleaned: {cleaned3}")
    print()

def test_model_connection():
    """Test actual model connection and response cleaning"""
    print("Testing model connection...")
    try:
        model = get_model()
        response = model.invoke("Say hello in one sentence.")
        print(f"Raw response: {response}")
        
        cleaned = clean_response(response)
        print(f"Cleaned response: {cleaned}")
        
        return True
    except Exception as e:
        print(f"Model connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing qwen3:4b Model Response Cleaning")
    print("=" * 50)
    
    test_response_cleaning()
    
    print("Testing actual model...")
    success = test_model_connection()
    
    if success:
        print("✅ Model connection and response cleaning working!")
    else:
        print("❌ Model connection failed. Check your Ollama setup.")
