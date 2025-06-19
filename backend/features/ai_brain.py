import requests
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.memory import MemoryManager

openai = None

class AIBrain:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.memory = MemoryManager()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            import openai as openai_lib
            openai_lib.api_key = self.api_key
            global openai
            openai = openai_lib
        self.client = None

    def ask(self, prompt: str) -> str:
        self.memory.memory["last_prompt"] = prompt
        try:
            if openai:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message['content'].strip() if hasattr(response.choices[0], 'message') else response.choices[0].text.strip()
            else:
                response = requests.post("http://localhost:11434/api/generate", json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                })
                answer = response.json().get("response", "[No response from local model]")
        except Exception as e:
            answer = f"[Error generating response: {e}]"

        self.memory.memory["last_answer"] = answer
        self.memory.save()
        return answer

