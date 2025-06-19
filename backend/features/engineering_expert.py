import requests
from typing import Callable
from .web_search import web_search
from .qa_memory import QAMemory
from .evaluator import Evaluator
from utils.memory import MemoryManager


class EngineeringExpert:
    """Domain expert for answering engineering questions."""

    FIELD_KEYWORDS = {
        "mechanical": [
            "mechanical",
            "thermodynamics",
            "kinematics",
            "materials",
            "machine",
            "gear",
            "fluid",
        ],
        "civil": [
            "civil",
            "structure",
            "bridge",
            "beam",
            "soil",
            "transport",
            "foundation",
        ],
        "electrical": [
            "electrical",
            "circuit",
            "control",
            "signal",
            "power",
            "electronics",
        ],
        "computer": [
            "computer",
            "algorithm",
            "architecture",
            "embedded",
            "software",
            "hardware",
        ],
        "chemical": [
            "chemical",
            "process",
            "reaction",
            "kinetics",
            "thermodynamics",
            "distillation",
        ],
        "aerospace": [
            "aerospace",
            "rocket",
            "aircraft",
            "flight",
            "aerodynamics",
            "satellite",
        ],
        "industrial": [
            "industrial",
            "manufacturing",
            "operations",
            "logistics",
            "optimization",
        ],
        "biomedical": [
            "biomedical",
            "medical",
            "prosthetic",
            "biomaterial",
            "tissue",
        ],
    }

    def __init__(self):
        self.evaluator = Evaluator()
        self.engineering_memory = QAMemory(path="data/engineering_memory.json")
        self.memory = MemoryManager(path="data/engineering_insights.json")

    @classmethod
    def is_engineering_question(cls, query: str) -> bool:
        q = query.lower()
        if "engineer" in q:
            return True
        for words in cls.FIELD_KEYWORDS.values():
            for kw in words:
                if kw in q:
                    return True
        return False

    def answer(self, query: str) -> str:
        field = self._detect_field(query)
        method: Callable[[str], str] = getattr(
            self, f"_answer_{field}", self._generic_answer
        )
        answer = method(query)
        score = self.evaluator.score(query, answer, "Ollama")
        self.engineering_memory.add(query, answer, "Ollama", score)
        self.memory.memory[query] = {"answer": answer, "score": score}
        self.memory.save()
        return answer

    def _detect_field(self, query: str) -> str:
        q = query.lower()
        best_field = ""
        best_matches = 0
        for field, kws in self.FIELD_KEYWORDS.items():
            matches = sum(1 for kw in kws if kw in q)
            if matches > best_matches:
                best_matches = matches
                best_field = field
        return best_field

    def _generic_answer(self, query: str, field: str = "engineering") -> str:
        context = web_search(f"{query} {field}")
        if context and "Web search error" not in context and "No results" not in context:
            prompt = f"{query}\n\nContext:\n{context}"
        else:
            prompt = query
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=10,
            )
            return resp.json().get("response", "")
        except Exception as exc:
            return f"[Error generating response: {exc}]"

    def _answer_mechanical(self, query: str) -> str:
        return self._generic_answer(query, "mechanical engineering")

    def _answer_civil(self, query: str) -> str:
        return self._generic_answer(query, "civil engineering")

    def _answer_electrical(self, query: str) -> str:
        return self._generic_answer(query, "electrical engineering")

    def _answer_computer(self, query: str) -> str:
        return self._generic_answer(query, "computer engineering")

    def _answer_chemical(self, query: str) -> str:
        return self._generic_answer(query, "chemical engineering")

    def _answer_aerospace(self, query: str) -> str:
        return self._generic_answer(query, "aerospace engineering")

    def _answer_industrial(self, query: str) -> str:
        return self._generic_answer(query, "industrial engineering")

    def _answer_biomedical(self, query: str) -> str:
        return self._generic_answer(query, "biomedical engineering")

