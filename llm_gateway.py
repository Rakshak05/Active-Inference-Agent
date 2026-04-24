import json
from config import config

class LLMGateway:
    """Interface to communicate with Language Models."""
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.model_name = config.MODEL_NAME
        self.temperature = config.TEMPERATURE
        
    def generate_completion(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """
        Implementation of an LLM call supporting OpenRouter and Ollama.
        """
        if config.DEBUG_MODE:
            print(f"--- LLM REQUEST ({self.model_name}) ---")
            print(f"System: {system_prompt[:100]}...")
            print(f"User: {user_prompt[:100]}...")
            print("-------------------------")
        import urllib.request
        import urllib.error
        
        try:
            from environment_probe import RateLimitTracker
            # Approximate prompt tokens tracking
            RateLimitTracker.log_call(estimated_prompt_tokens=len(system_prompt)//4 + len(user_prompt)//4)
            
            model = self.model_name
            
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
                    "temperature": self.temperature
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
                        "temperature": self.temperature
                    }
                }
                if json_mode:
                    data["format"] = "json"
            
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            
            # with urllib.request.urlopen(req) as response:
            with urllib.request.urlopen(req, timeout=120) as response:
                res_body = response.read()
                res_json = json.loads(res_body)
                
                if self.api_key.startswith("sk-"):
                    content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    content = res_json.get("message", {}).get("content", "")
                
                if json_mode:
                    # Clean markdown json blocks
                    content = content.replace("```json", "").replace("```", "").strip()
                    
                return content
                
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8', errors='ignore')
            api_name = "OpenRouter" if self.api_key.startswith("sk-") else "Ollama"
            print(f"{api_name} API Error ({e.code}): {err_msg}")
            raise e
        except Exception as e:
            print(f"API Error ({e}). LLM Request failed.")
            raise e

