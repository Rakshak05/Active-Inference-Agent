import os
import sys

# Native .env loader (zero dependencies required)
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

class Config:
    """Configuration settings for the Active Inference Agent."""
    
    # LLM Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME     = os.getenv("MODEL_NAME", "mistral")
    TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.2"))

    # Active Inference Settings
    EFE_THRESHOLD = float(os.getenv("EFE_THRESHOLD", "0.65"))
    MAX_REPLANS   = int(os.getenv("MAX_REPLANS", "3"))
    
    # Judge Settings
    JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", MODEL_NAME)
    JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.5"))
    
    # EFE Component Weights
    RISK_WEIGHT      = 1.0
    AMBIGUITY_WEIGHT = 0.7

    # Application Settings
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

    # Email / SMTP  (set these in .env to send real emails)
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASS = os.getenv("SMTP_PASS", "")
    SMTP_FROM = os.getenv("SMTP_FROM", "")

    # HTTP adapter
    HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
    # LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "600"))
    LLM_TIMEOUT  = int(os.getenv("LLM_TIMEOUT", "90"))

config = Config()
