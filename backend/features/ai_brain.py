import os
from utils.memory import MemoryManager

try:
    import openai
except Exception:
    openai = None

class AIBrain:
    """Simple AI Brain using OpenAI's chat API if available."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.memory = MemoryManager()
        self.memory.load()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if openai and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def ask(self, prompt: str) -> str:
        """Return a response for the given prompt."""
        if self.client:
            try:
                chat = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = chat.choices[0].message.content.strip()
            except Exception as exc:
                answer = f"Error contacting OpenAI: {exc}"
        else:
            answer = f"(OpenAI unavailable) You said: {prompt}"

        # Persist last interaction in memory
        self.memory.memory["last_prompt"] = prompt
        self.memory.memory["last_answer"] = answer
        self.memory.save()
        return answer
