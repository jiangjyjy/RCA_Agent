import time
import re
import random
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

def clean_json_text(text):
    if not text: return "{}"
    text = str(text)
    
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
    
    text = text.replace("```json", "").replace("```", "")
    
    return text.strip()

def api_retry(max_retries=5, initial_delay=2.0, backoff_factor=2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, APIError) as e:
                    last_exception = e
                    
                    if isinstance(e, InternalServerError) or (hasattr(e, 'code') and e.code == 500):
                        current_delay = max(delay, 10.0)
                    else:
                        current_delay = delay

                    if attempt == max_retries:
                        print(f"\n[Give Up] {func.__name__} failed after {max_retries} retries. Error: {e}")
                        raise e
                    
                    sleep_time = current_delay * (1 + random.random() * 0.1)
                    
                    error_type = type(e).__name__
                    print(f"\n[Retry {attempt+1}/{max_retries}] {error_type}: Waiting {sleep_time:.2f}s...", end="", flush=True)
                    
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                
                except Exception as e:
                    print(f"\n[Fatal Error] {e}")
                    raise e
            
            raise last_exception
        return wrapper
    return decorator