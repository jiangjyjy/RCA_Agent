import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from rca_framework.utils import api_retry
from rca_framework.config import API_SEMAPHORE

load_dotenv()

MODEL_CONFIG = {
    "deepseek-v3.2": {
        "api_key": os.getenv("XXX"),
        "base_url": "XXX",
        "model": "XXX",
    },
    "kimi": {
        "api_key": os.getenv("XXX"),
        "base_url": "XXX",
        "model": "XXX", 
    },
    "glm-4.7": {
        "api_key": os.getenv("XXX"),
        "base_url": "XXX", 
        "model": "XXX", 
    },
    "gpt-5": {
        "api_key": os.getenv("XXX"), 
        "base_url": "XXX", 
        "model": "XXX",
    },
    "gpt-4o": {
        "api_key": os.getenv("XXX"), 
        "base_url": "XXX", 
        "model": "XXX",
    },
    "gemini-3": {
        "api_key": os.getenv("XXX"),
        "base_url": "XXX",
        "model": "XXX",
    }
}

class UnifiedLLM:
    def __init__(self, config_key):
        if config_key not in MODEL_CONFIG:
            raise ValueError(f"[UnifiedLLM] Config key '{config_key}' not found.")
            
        cfg = MODEL_CONFIG[config_key]

        is_thinking = any(x in config_key.lower() for x in ["thinking", "r1"])
        timeout_setting = 240.0 if is_thinking else 120.0
        
        self.client = OpenAI(
            api_key=cfg["api_key"], 
            base_url=cfg["base_url"],
            timeout=timeout_setting 
        )
        self.model_name = cfg["model"]

    @api_retry(max_retries=5, initial_delay=5.0) 
    def chat(self, messages, temperature=0.7, json_mode=False):
        model_id_lower = self.model_name.lower()
        is_reasoning_model = any(kw in model_id_lower for kw in ["thinking", "reasoner", "r1"])
        
        params = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": temperature
        }

        if is_reasoning_model:
            params["temperature"] = 0.6 
            json_mode = False 
            if "response_format" in params: del params["response_format"]
            params["max_tokens"] = 8192
            
            if params["messages"][-1]["role"] == "user":
                params["messages"][-1]["content"] += "\n\n【System Note】Please output the result strictly in JSON format. Do not use Markdown blocks."

        if json_mode and not is_reasoning_model:
            params["response_format"] = {"type": "json_object"}
            
        with API_SEMAPHORE: 
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content