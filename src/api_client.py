import requests, json, time
from typing import Dict, Optional, Any, List

class OpenAICompatClient:
    def __init__(self, api_base: str, api_key: str, use_chat: bool = True, model: str = ""):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.use_chat = use_chat
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.api_base}/models", headers=self.headers, timeout=20)
        return {"status_code": r.status_code, "text": r.text[:400]}

    def chat_complete(self, messages: List[Dict[str, str]], **gen):
        url = f"{self.api_base}/chat/completions"
        payload = {"model": self.model, "messages": messages, **gen}
        r = requests.post(url, headers=self.headers, json=payload, timeout=gen.get("timeout", 180))
        r.raise_for_status()
        return r.json()

    def text_complete(self, prompt: str, **gen):
        url = f"{self.api_base}/completions"
        payload = {"model": self.model, "prompt": prompt, **gen}
        r = requests.post(url, headers=self.headers, json=payload, timeout=gen.get("timeout", 180))
        r.raise_for_status()
        return r.json()

    def complete(self, user_content: str, system: str = None, **gen):
        if self.use_chat:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user_content})
            return self.chat_complete(msgs, **gen)
        else:
            return self.text_complete(user_content, **gen)
