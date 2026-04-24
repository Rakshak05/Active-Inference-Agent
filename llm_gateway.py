import json
from typing import Optional
from config import config

class LLMGateway:
    """Interface to communicate with Language Models."""
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.model_name = config.MODEL_NAME
        self.temperature = config.TEMPERATURE
        
    def generate_completion(self, system_prompt: str, user_prompt: str, json_mode: bool = False, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        Implementation of an LLM call supporting OpenRouter and Ollama.
        """
        model = model or self.model_name
        temp = temperature if temperature is not None else self.temperature
        if config.DEBUG_MODE:
            print(f"--- LLM REQUEST ({model}) ---")
            print(f"System: {system_prompt[:100]}...")
            print(f"User: {user_prompt[:100]}...")
            print("-------------------------")
        import urllib.request
        import urllib.error
        
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                from environment_probe import RateLimitTracker
                # Approximate prompt tokens tracking
                RateLimitTracker.log_call(estimated_prompt_tokens=len(system_prompt)//4 + len(user_prompt)//4)
                                
                if self.api_key.startswith("sk-"):
                    # OpenRouter / OpenAI API Integration
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temp
                    }
                    if json_mode:
                        data["response_format"] = {"type": "json_object"}
                else:
                    # Local Ollama Integration
                    url = "http://localhost:11434/api/chat"
                    headers = {
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": temp
                        }
                    }
                    if json_mode:
                        data["format"] = "json"
                
                req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
                
                with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as response:
                    res_body = response.read()
                    res_json = json.loads(res_body)
                    
                    if self.api_key.startswith("sk-"):
                        content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        content = res_json.get("message", {}).get("content", "")
                    
                    if json_mode:
                        content = content.replace("```json", "").replace("```", "").strip()
                        
                    return content
                    
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError) as e:
                # If it's an HTTP 429 (Rate Limit) or Connection Error, retry.
                is_retryable = False
                if isinstance(e, urllib.error.HTTPError):
                    if e.code in (429, 502, 503, 504): is_retryable = True
                else:
                    is_retryable = True # URLError/ConnectionError usually retryable
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"API Error ({e}). Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                err_msg = str(e)
                if isinstance(e, urllib.error.HTTPError):
                    err_msg = e.read().decode('utf-8', errors='ignore')
                print(f"API Error ({err_msg}). Final attempt failed.")
                raise e
            except Exception as e:
                print(f"Unexpected API Error ({e}).")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                raise e
