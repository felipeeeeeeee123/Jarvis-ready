import requests
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.memory_db import DatabaseMemoryManager
from database.services import qa_service
from .evaluator import Evaluator
from .web_search import web_search
from .engineering_expert import EngineeringExpert
from config.settings import settings

openai = None


class AIBrain:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.memory = DatabaseMemoryManager()
        self.qa_service = qa_service
        self.evaluator = Evaluator()
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key:
            try:
                import openai as openai_lib
                openai_lib.api_key = self.api_key
                global openai
                openai = openai_lib
            except ImportError:
                openai = None
        self.client = None

    def solve_pdf(self, path: str) -> dict:
        expert = EngineeringExpert()
        return expert.solve_pdf_worksheet(path)

    def ask(self, prompt: str) -> str:
        self.memory.memory["last_prompt"] = prompt
        if prompt.lower().startswith("solve worksheet"):
            path = prompt.split(" ", 2)[-1]
            try:
                results = self.solve_pdf(path)
                answer = "\n\n".join(results.values())
                self.memory.memory["last_answer"] = answer
                self.memory.save()
                return answer
            except Exception as exc:
                return f"[Error solving worksheet: {exc}]"
        if EngineeringExpert.is_engineering_question(prompt):
            expert = EngineeringExpert()
            answer = expert.answer(prompt)
            self.memory.set("last_answer", answer)
            score = self.evaluator.score(prompt, answer, "Ollama")
            self.qa_service.add_entry(prompt, answer, "Ollama", score)
            self.evaluator.update_leaderboard(prompt, score)
            return answer
        try:
            if openai:
                response = openai.ChatCompletion.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}]
                )
                answer = (
                    response.choices[0].message["content"].strip()
                    if hasattr(response.choices[0], "message")
                    else response.choices[0].text.strip()
                )
            else:
                ollama_url = f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}/api/generate"
                response = requests.post(
                    ollama_url,
                    json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                    timeout=30
                )
                answer = response.json().get(
                    "response", "[No response from local model]"
                )
        except Exception as e:
            answer = f"[Error generating response: {e}]"

        self.memory.set("last_answer", answer)
        score = self.evaluator.score(prompt, answer, "Ollama")
        
        # auto-correct low confidence answers using web search context
        if score < 0.5 and settings.WEB_SEARCH_ENABLED:
            try:
                context = web_search(prompt)
                improved_prompt = f"{prompt}\n\nContext:\n{context}"
                ollama_url = f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}/api/generate"
                resp = requests.post(
                    ollama_url,
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": improved_prompt,
                        "stream": False,
                    },
                    timeout=30
                )
                new_answer = resp.json().get("response", answer)
                new_score = self.evaluator.score(prompt, new_answer, "Ollama")
                if new_score > score:
                    answer = new_answer
                    score = new_score
            except Exception:
                pass

        self.qa_service.add_entry(prompt, answer, "Ollama", score)
        self.evaluator.update_leaderboard(prompt, score)
        return answer
