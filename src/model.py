from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
import os
import re



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_model():
    """
    Initialize and return a Google Generative AI model instance.

    Returns:
        ChatGoogleGenerativeAI: Configured model instance
    """
    llm = OllamaLLM(
        model="qwen3:4b",
        temperature=0.7,
    )
    return llm

def clean_response(response):
    """
    Clean the response from reasoning models that wrap output in <think> tags
    
    Args:
        response (str): Raw response from the model
        
    Returns:
        str: Cleaned response with thinking tags removed
    """
    if isinstance(response, str):
        # Remove <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # If the entire response was just thinking, try to extract useful content
        if not cleaned.strip():
            # Look for any content after </think> tag
            think_match = re.search(r'</think>\s*(.+)', response, flags=re.DOTALL | re.IGNORECASE)
            if think_match:
                cleaned = think_match.group(1)
            else:
                # If no content after </think>, extract the thinking content as fallback
                think_content = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL | re.IGNORECASE)
                if think_content:
                    cleaned = think_content.group(1)
        
        return cleaned.strip()
    return response

def is_query_harmful(query, model):
    """
    Check if a query is harmful using the specified model.

    Args:
        query (str): The query to check
        model (ChatGoogleGenerativeAI): The model instance to use for checking

    Returns:
        bool: True if the query is harmful, False otherwise
    """
    query_to_check = f"""
    You are a safety classifier. Your task is to determine if a given query is harmful or
    unsafe. A query is considered harmful if it promotes violence, hate speech, self-harm,
    illegal activities, or any other form of unsafe content. Please respond with "harmful"
    if the query is harmful, and "safe" if it is not.

    Here is the query you need to evaluate:
    {query}
    """
    llm = OllamaLLM(
        model="qwen3:4b",
        temperature=0.0,
    )

    response = model.invoke(query_to_check)
    response = clean_response(response)  # Clean the response
    print(f"Safety check response: {response}")
    if "harmful" in response.lower():
        return True
    if "safe" in response.lower():
        return False
    
    return False


def check_model_availability():
    """
    Check if the Google Generative AI model is available.

    Returns:
        bool: True if the model is available, False otherwise
    """
    try:
        model = get_model()
        # Perform a simple invocation to check availability
        response = model.invoke("Test availability")
        response = clean_response(response)  # Clean the response
        print(f"Model is available: {response}")
        return True if response else False
    except Exception as e:
        print(f"Model availability check failed: {e}")
        return False
    

if __name__ == "__main__":
    check_model_availability()