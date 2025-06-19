# pyright: reportMissingImports=false
import requests
from typing import Callable
import fitz
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
        self.textbook_memory = MemoryManager(path="data/engineering_textbooks.json")

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

    def ingest_pdf(self, path: str, discipline: str) -> None:
        """Parse a PDF textbook and store content per discipline."""
        doc = fitz.open(path)
        chapters = []
        current = {"title": "Introduction", "formulas": [], "content": ""}
        for page in doc:
            text = page.get_text()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("chapter"):
                    if current["content"] or current["formulas"]:
                        chapters.append(current)
                    current = {"title": line, "formulas": [], "content": ""}
                elif "=" in line and any(c.isalpha() for c in line):
                    current["formulas"].append(line)
                else:
                    if current["content"]:
                        current["content"] += " " + line
                    else:
                        current["content"] = line
        chapters.append(current)
        self.textbook_memory.memory.setdefault(discipline, []).extend(chapters)
        self.textbook_memory.save()

    def _textbook_context(self, discipline: str, query: str) -> str:
        """Retrieve relevant textbook snippets for the query."""
        data = self.textbook_memory.memory.get(discipline, [])
        q = query.lower()
        snippets = []
        for chap in data:
            if q in chap.get("title", "").lower():
                snippets.append(chap.get("content", ""))
                snippets.extend(chap.get("formulas", []))
                continue
            if q in chap.get("content", "").lower():
                snippets.append(chap.get("content", ""))
            for formula in chap.get("formulas", []):
                if q in formula.lower():
                    snippets.append(formula)
        return "\n".join(snippets[:5])

    def _generic_answer(self, query: str, field: str = "engineering") -> str:
        discipline = field.split()[0]
        context = self._textbook_context(discipline, query)
        if not context:
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

