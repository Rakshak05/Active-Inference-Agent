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
        Mock implementation of an LLM call.
        In a real application, replace this with a call to openai.ChatCompletion.create or Anthropic's client.
        """
        if config.DEBUG_MODE:
            print(f"--- LLM REQUEST ({self.model_name}) ---")
            print(f"System: {system_prompt[:100]}...")
            print(f"User: {user_prompt[:100]}...")
            print("-------------------------")
        import urllib.request
        import urllib.error
        
        try:
            # Local Ollama Integration
            url = "http://localhost:11434/api/chat"
            headers = {
                "Content-Type": "application/json"
            }
            
            model = self.model_name
            
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
            
            with urllib.request.urlopen(req) as response:
                res_body = response.read()
                res_json = json.loads(res_body)
                
                content = res_json.get("message", {}).get("content", "")
                
                if json_mode:
                    # Clean markdown json blocks
                    content = content.replace("```json", "").replace("```", "").strip()
                    
                return content
                
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8', errors='ignore')
            print(f"⚠️ Ollama API Error ({e.code}): {err_msg}")
            raise e
        except Exception as e:
            print(f"API Error ({e}). LLM Request failed.")
            raise e

