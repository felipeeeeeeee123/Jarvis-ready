import requests
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.memory import MemoryManager
from .qa_memory import QAMemory
from .evaluator import Evaluator
from .web_search import web_search

openai = None

class AIBrain:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.memory = MemoryManager()
        self.qa_memory = QAMemory()
        self.evaluator = Evaluator()
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
        score = self.evaluator.score(prompt, answer, "Ollama")
        # auto-correct low confidence answers using web search context
        if score < 0.5:
            try:
                context = web_search(prompt)
                improved_prompt = f"{prompt}\n\nContext:\n{context}"
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "mistral", "prompt": improved_prompt, "stream": False},
                )
                new_answer = resp.json().get("response", answer)
                new_score = self.evaluator.score(prompt, new_answer, "Ollama")
                if new_score > score:
                    answer = new_answer
                    score = new_score
            except Exception:
                pass

        self.qa_memory.add(prompt, answer, "Ollama", score)
        self.evaluator.update_leaderboard(prompt, score)
        self.qa_memory.prune()
        return answer

