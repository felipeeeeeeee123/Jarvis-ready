# pyright: reportMissingImports=false
import os
import time
import requests
from typing import Callable, List, Tuple
import re
from uuid import uuid4
from difflib import SequenceMatcher
import fitz
import sympy as sp
import matplotlib

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bpy = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    go = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .web_search import web_search
from .qa_memory import QAMemory
from .evaluator import Evaluator
from utils.memory import MemoryManager

BLUEPRINT_DIR = "blueprints"
os.makedirs(BLUEPRINT_DIR, exist_ok=True)
SIMULATION_DIR = "simulations"
os.makedirs(SIMULATION_DIR, exist_ok=True)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


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
        "nuclear": [
            "nuclear",
            "reactor",
            "radiation",
            "neutron",
            "fission",
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
        self.formula_index = MemoryManager(path="data/formula_index.json")
        self.sim_index = MemoryManager(path="data/simulation_index.json")

    def _index_formula(self, formula: str, tags: List[str], steps: str) -> None:
        entry = {
            "formula": formula,
            "tags": tags,
            "steps": steps,
            "timestamp": time.time(),
        }
        formulas = self.formula_index.memory.setdefault("formulas", [])
        formulas.append(entry)
        self.formula_index.save()

    def _find_similar_formula(self, formula: str) -> dict | None:
        """Return the most similar indexed formula entry if similarity > 0.8."""
        best = None
        best_score = 0.0
        for entry in self.formula_index.memory.get("formulas", []):
            existing = entry.get("formula", "")
            score = SequenceMatcher(None, existing, formula).ratio()
            if score > best_score:
                best = entry
                best_score = score
        if best_score > 0.8:
            return best
        return None

    @staticmethod
    def _is_math_problem(query: str) -> bool:
        q = query.lower()
        triggers = ["integrate", "derivative", "solve", "=", "d/d"]
        return any(t in q for t in triggers)

    def _solve_symbolically(self, query: str) -> str:
        try:
            q = query.lower()
            if "integrate" in q:
                expr_str = q.split("integrate", 1)[1]
                formula = f"∫ {expr_str}"
                cached = self._find_similar_formula(formula)
                if cached:
                    return cached.get("steps", cached["formula"])
                expr = sp.sympify(expr_str)
                result = sp.integrate(expr)
                simplified = sp.simplify(result)
                steps = f"Integrate {expr_str}\n\\boxed{{{simplified}}}"
                self._index_formula(formula, expr_str.split(), steps)
                return steps
            if "derivative" in q or "d/d" in q:
                expr_str = (
                    q.split("derivative of", 1)[1]
                    if "derivative of" in q
                    else q.split("d/dx", 1)[1]
                )
                x = sp.symbols("x")
                formula = f"d({expr_str})/dx"
                cached = self._find_similar_formula(formula)
                if cached:
                    return cached.get("steps", cached["formula"])
                expr = sp.sympify(expr_str)
                result = sp.diff(expr, x)
                simplified = sp.simplify(result)
                steps = f"Differentiate {expr_str} w.r.t x\n\\boxed{{{simplified}}}"
                self._index_formula(formula, expr_str.split(), steps)
                return steps
            if "solve" in q:
                m = re.search(r"solve (.+?)=([^ ]+) for ([a-zA-Z])", q)
                if m:
                    left, right, var = m.groups()
                    symbol = sp.symbols(var)
                    formula = f"{left}={right}"
                    cached = self._find_similar_formula(formula)
                    if cached:
                        return cached.get("steps", cached["formula"])
                    equation = sp.Eq(sp.sympify(left), sp.sympify(right))
                    solution = sp.solve(equation, symbol)
                    steps = f"Solve {left} = {right} for {var}\n\\boxed{{{solution}}}"
                    self._index_formula(formula, [var], steps)
                    return steps
            if "=" in q:
                left, right = q.split("=", 1)
                vars_ = list(
                    sp.sympify(left).free_symbols | sp.sympify(right).free_symbols
                )
                if vars_:
                    formula = f"{left}={right}"
                    cached = self._find_similar_formula(formula)
                    if cached:
                        return cached.get("steps", cached["formula"])
                    solution = sp.solve(
                        sp.Eq(sp.sympify(left), sp.sympify(right)), vars_
                    )
                    steps = f"Solve {left} = {right} for {', '.join(map(str, vars_))}\n\\boxed{{{solution}}}"
                    self._index_formula(formula, [str(v) for v in vars_], steps)
                    return steps
        except Exception as exc:
            return f"[Error solving symbolically: {exc}]"
        return "[Unable to solve symbolically]"

    @staticmethod
    def _is_blueprint_request(query: str) -> bool:
        q = query.lower()
        return any(w in q for w in ["draw", "blueprint", "diagram"])

    @staticmethod
    def _is_simulation_request(query: str) -> bool:
        q = query.lower()
        keywords = [
            "simulate",
            "simulation",
            "stress",
            "current",
            "voltage",
            "thermal",
            "heat",
        ]
        return any(k in q for k in keywords)

    def _generate_blueprint(self, query: str) -> str:
        q = query.lower()
        if "truss" in q:
            joints = 3
            width = joints
            height = 1
            jm = re.search(r"(\d+)[- ]*beam", q)
            if jm:
                joints = int(jm.group(1)) + 1
            m = re.search(r"(\d+)\s*joints?", q)
            if m:
                joints = int(m.group(1))
            dm = re.search(r"(\d+(?:\.\d+)?)\s*[xby]\s*(\d+(?:\.\d+)?)", q)
            if dm:
                width = float(dm.group(1))
                height = float(dm.group(2))
            return self._draw_truss(joints, width, height)
        if "beam" in q:
            spans = 1
            length = spans
            m = re.search(r"(\d+)\s*spans?", q)
            if m:
                spans = int(m.group(1))
            dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters)?", q)
            if dm:
                length = float(dm.group(1))
            return self._draw_beam(spans, length)
        if "circuit" in q:
            comps = 2
            m = re.search(r"(\d+)\s*(?:resistors|components)", q)
            if m:
                comps = int(m.group(1))
            return self._draw_circuit(comps)
        if "pcb" in q:
            comps = 2
            m = re.search(r"(\d+)\s*components", q)
            if m:
                comps = int(m.group(1))
            return self._draw_pcb(comps)
        return "[Blueprint request not understood]"

    def _draw_truss(self, joints: int, width: float, height: float) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, width, joints)
        ax.plot(x, [0] * joints, "ko-")
        for i in range(joints - 2):
            ax.plot([x[i], x[i + 1]], [0, height], "k-")
            ax.plot([x[i + 1], x[i + 2]], [0, height], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = f"truss with {joints} joints"
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_beam(self, spans: int, length: float) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, length, spans + 1)
        ax.plot([0, length], [0, 0], "k-", lw=2)
        for pos in x:
            ax.plot([pos, pos], [0, -0.2], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = f"beam with {spans} spans"
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_circuit(self, components: int) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, components, components + 1)
        for i in range(components):
            ax.plot([x[i], x[i + 1]], [0, 0], "k-")
            ax.text((x[i] + x[i + 1]) / 2, 0.1, f"R{i + 1}", ha="center")
        ax.plot([0, 0], [0, 1], "k-")
        ax.plot([components, components], [0, 1], "k-")
        ax.plot([0, components], [1, 1], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = (
            f"circuit with {components} components"
        )
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_pcb(self, components: int) -> str:
        fig, ax = plt.subplots()
        for i in range(components):
            rect = plt.Rectangle(
                (i, i % 2), 0.8, 0.4, edgecolor="black", facecolor="lightgray"
            )
            ax.add_patch(rect)
        ax.set_xlim(0, components)
        ax.set_ylim(0, 2)
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = (
            f"pcb with {components} components"
        )
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

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
        if self._is_simulation_request(query):
            answer = self.simulate(query)
        elif self._is_blueprint_request(query):
            answer = self._generate_blueprint(query)
        elif self._is_math_problem(query):
            answer = self._solve_symbolically(query)
        else:
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

    def _format_worksheet_answer(self, question: str, solution: str) -> str:
        """Return tutor-style formatted solution string."""
        final_line = solution.splitlines()[-1]
        boxed = f"**{final_line}**"
        return f"**Question:** {question}\n{solution}\n\n{boxed}"

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

    def solve_pdf_worksheet(self, path: str) -> dict:
        """Solve each numbered problem in a PDF worksheet."""
        doc = fitz.open(path)
        results = {}
        pattern = re.compile(r"^\d+\.\s+")
        for page in doc:
            lines = page.get_text().splitlines()
            current = ""
            for line in lines:
                if pattern.match(line.strip()):
                    if current:
                        sol = self.answer(current)
                        results[current] = self._format_worksheet_answer(current, sol)
                    current = pattern.sub("", line.strip())
                else:
                    current += " " + line.strip()
            if current:
                sol = self.answer(current)
                results[current] = self._format_worksheet_answer(current, sol)
                current = ""
        return results

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
        if (
            context
            and "Web search error" not in context
            and "No results" not in context
        ):
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

    def _answer_nuclear(self, query: str) -> str:
        return self._generic_answer(query, "nuclear engineering")

    def _answer_industrial(self, query: str) -> str:
        return self._generic_answer(query, "industrial engineering")

    def _answer_biomedical(self, query: str) -> str:
        return self._generic_answer(query, "biomedical engineering")

    # --- Simulation Capabilities ---
    def simulate(
        self, prompt: str, params: dict | None = None, tags: list[str] | None = None
    ) -> str:
        """Run a simple physics simulation based on the prompt."""
        if params:
            extras = " " + " ".join(f"{k}={v}" for k, v in params.items())
            prompt += extras
        q = prompt.lower()
        field = self._detect_field(q)
        if "flow" in q or "fluid" in q:
            method = self._simulate_fluid_flow
        else:
            method = getattr(self, f"_simulate_{field}", self._simulate_mechanical)
        result_tuple = method(q)
        model = None
        margin = 0.0
        if len(result_tuple) == 5:
            result, path, model, fail, margin = result_tuple
        elif len(result_tuple) == 4:
            result, path, fail, margin = result_tuple
        else:
            result, path, fail = result_tuple
        self.engineering_memory.add(prompt, result, "Simulation", 1.0, tags or ["simulation"])
        self.memory.memory["last_simulation"] = path
        if model:
            self.memory.memory["last_model"] = model
        self.memory.save()
        entry = {
            "uuid": str(uuid4()),
            "prompt": prompt,
            "field": field,
            "result": result,
            "path": path,
            "model": model,
            "timestamp": time.time(),
            "params": params or {},
            "performance": margin,
            "tags": tags or [],
        }
        sims = self.sim_index.memory.setdefault("simulations", [])
        sims.append(entry)
        self.sim_index.save()
        msg = f"{result}\nSimulation saved to {path}\n![simulation]({path})"
        if model:
            msg += f"\n3D model saved to {model}"
        if fail:
            msg += f"\n**Failure detected. Suggestion:** {self._suggest_fix(field)}"
        msg += f"\nPerformance score: {margin:.2f}"
        return msg

    def optimize(
        self,
        sim_type: str,
        objective: str,
        ranges: dict,
        iterations: int = 20,
    ) -> dict:
        """Simple random search optimization for a simulation."""
        import random

        best: dict | None = None
        best_score = float("inf")
        for _ in range(iterations):
            params = {
                k: random.uniform(v[0], v[1]) if isinstance(v, (list, tuple)) else v
                for k, v in ranges.items()
            }
            result_msg = self.simulate(sim_type, params)
            entry = self.sim_index.memory.get("simulations", [])[-1]
            perf = entry.get("performance", 0)
            score = abs(perf)
            if score < best_score:
                best_score = score
                best = {"params": params, "result": result_msg, "score": perf}
        return best or {}

    def design_assistant(self, description: str) -> str:
        """Heuristic-based design suggestions."""
        desc = description.lower()
        material = "steel" if "steel" in desc else "aluminum"
        span = re.search(r"(\d+(?:\.\d+)?)\s*m", desc)
        load = re.search(r"(\d+(?:\.\d+)?)\s*n", desc)
        length = float(span.group(1)) if span else 1.0
        force = float(load.group(1)) if load else 1000.0
        suggestion = (
            f"Material: {material}\nCross-section height: 0.1 m\n"
            f"Estimated span: {length} m under {force} N"
        )
        return suggestion

    def analysis_chain(self, description: str) -> str:
        """Run blueprint, simulation and optimization in sequence."""
        blueprint = self._generate_blueprint(description)
        sim_res = self.simulate(description)
        opt = self.optimize("mechanical", "stress", {"length": [1, 5], "force": [500, 2000]})
        return f"{blueprint}\n\n{sim_res}\n\nBest config: {opt}"

    def _suggest_fix(self, field: str) -> str:
        fixes = {
            "mechanical": "Use stronger material or add support",
            "civil": "Use I-beam instead of flat bar",
            "electrical": "Add a heat sink",
            "chemical": "Lower temperature or pressure",
            "aerospace": "Increase wing area",
            "nuclear": "Insert control rods",
            "computer": "Improve cooling",
        }
        return fixes.get(field, "Review design parameters")

    def _create_beam_model(self, length: float, height: float) -> str:
        """Create a simple 3D beam model using Blender and export as GLB."""
        if bpy is None:  # pragma: no cover - optional dependency
            return ""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("Beam")
        verts = [
            (0, 0, 0),
            (length, 0, 0),
            (length, height, 0),
            (0, height, 0),
            (0, 0, height / 10),
            (length, 0, height / 10),
            (length, height, height / 10),
            (0, height, height / 10),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        mesh.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new("Beam", mesh)
        bpy.context.scene.collection.objects.link(obj)
        glb_path = os.path.join(MODEL_DIR, f"beam_{int(time.time()*1000)}.glb")
        bpy.ops.export_scene.gltf(filepath=glb_path, export_format="GLB")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return glb_path

    def _simulate_mechanical(self, q: str) -> Tuple[str, str, str, bool, float]:
        length = 1.0
        force = 1000.0
        h = 0.1
        m = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if m:
            length = float(m.group(1))
        fm = re.search(r"(\d+(?:\.\d+)?)\s*n", q)
        if fm:
            force = float(fm.group(1))
        hm = re.search(r"h=(\d+(?:\.\d+)?)", q)
        if hm:
            h = float(hm.group(1))
        x = np.linspace(0, length, 100)
        inertia = max(h ** 4 / 12, 1e-6)
        stress = force * (length - x) * (h / 2) / inertia
        fig, ax = plt.subplots()
        ax.plot(x, stress)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Stress")
        img_path = os.path.join(
            SIMULATION_DIR, f"mechanical_{int(time.time()*1000)}.png"
        )
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        model_path = ""
        try:
            model_path = self._create_beam_model(length, h)
        except Exception:
            model_path = ""
        max_stress = float(stress.max())
        fail = max_stress > 2.5e8
        result = f"Max stress: {max_stress:.2f} Pa"
        margin = 2.5e8 - max_stress
        return result, img_path, model_path, fail, margin

    def _simulate_electrical(self, q: str) -> Tuple[str, str, bool, float]:
        voltage = 5.0
        resistors = re.findall(r"(\d+(?:\.\d+)?)\s*ohm", q)
        values = [float(r) for r in resistors] if resistors else [1.0, 1.0]
        vm = re.search(r"(\d+(?:\.\d+)?)\s*v", q)
        if vm:
            voltage = float(vm.group(1))
        total_r = sum(values)
        current = voltage / total_r
        drops = [current * r for r in values]
        fig, ax = plt.subplots()
        ax.bar(range(1, len(drops) + 1), drops)
        ax.set_xlabel("Resistor")
        ax.set_ylabel("Voltage Drop (V)")
        path = os.path.join(SIMULATION_DIR, f"electrical_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = current > 10
        result = f"Circuit current: {current:.2f} A"
        margin = 10 - current
        return result, path, fail, margin

    def _simulate_thermal(self, q: str) -> Tuple[str, str, bool, float]:
        length = 1.0
        t1, t2 = 100.0, 0.0
        lm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if lm:
            length = float(lm.group(1))
        tmatch = re.findall(r"(\d+(?:\.\d+)?)\s*c", q)
        if len(tmatch) >= 2:
            t1, t2 = float(tmatch[0]), float(tmatch[1])
        x = np.linspace(0, length, 50)
        temp = t1 + (t2 - t1) * x / length
        fig, ax = plt.subplots()
        ax.plot(x, temp)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Temperature (C)")
        path = os.path.join(SIMULATION_DIR, f"thermal_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        max_temp = max(t1, t2)
        fail = max_temp > 500
        result = f"Temperature range: {t1}C to {t2}C"
        margin = 500 - max_temp
        return result, path, fail, margin

    def _simulate_civil(self, q: str) -> Tuple[str, str, bool, float]:
        load = 1e4
        span = 5.0
        lm = re.search(r"(\d+(?:\.\d+)?)\s*ton", q)
        if lm:
            load = float(lm.group(1)) * 9.81e3
        sm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if sm:
            span = float(sm.group(1))
        x = np.linspace(0, span, 50)
        E = 2e11
        moment_of_inertia = 1e-4
        y = load * x**2 * (3 * span - x) / (6 * E * moment_of_inertia)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Deflection (m)")
        path = os.path.join(SIMULATION_DIR, f"civil_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        max_def = float(y.max())
        allow = span / 100
        fail = max_def > allow
        result = f"Max deflection: {max_def:.4f} m"
        margin = allow - max_def
        return result, path, fail, margin

    def _simulate_aerospace(self, q: str) -> Tuple[str, str, bool, float]:
        speed = 50.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*m/s", q)
        if m:
            speed = float(m.group(1))
        rho = 1.225
        area = 1.0
        lift = 0.5 * rho * speed**2 * area
        v = np.linspace(0, speed, 50)
        fig, ax = plt.subplots()
        ax.plot(v, 0.5 * rho * v**2 * area)
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Lift (N)")
        path = os.path.join(SIMULATION_DIR, f"aero_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = lift < 1e3
        result = f"Lift at {speed} m/s: {lift:.1f} N"
        margin = lift - 1e3
        return result, path, fail, margin

    def _simulate_chemical(self, q: str) -> Tuple[str, str, bool, float]:
        temp = 300.0
        pressure = 1.0
        tm = re.search(r"(\d+(?:\.\d+)?)\s*c", q)
        if tm:
            temp = float(tm.group(1))
        pm = re.search(r"(\d+(?:\.\d+)?)\s*atm", q)
        if pm:
            pressure = float(pm.group(1))
        k = np.exp(-(5000) / (8.314 * (temp + 273.15)))
        fig, ax = plt.subplots()
        t = np.linspace(0, 10, 100)
        conc = np.exp(-k * t)
        ax.plot(t, conc)
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        path = os.path.join(SIMULATION_DIR, f"chemical_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = temp > 400 or pressure > 5
        result = f"Rate constant: {k:.4f}"
        margin = 400 - temp
        return result, path, fail, margin

    def _simulate_nuclear(self, q: str) -> Tuple[str, str, bool, float]:
        flux = 1e12
        m = re.search(r"(\d+(?:\.\d+)?)\s*n/s", q)
        if m:
            flux = float(m.group(1))
        t = np.linspace(0, 10, 100)
        decay = np.exp(-0.1 * t)
        fig, ax = plt.subplots()
        ax.plot(t, decay)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Activity")
        path = os.path.join(SIMULATION_DIR, f"nuclear_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = flux > 1e14
        result = f"Neutron flux: {flux:.2e} n/s"
        margin = 1e14 - flux
        return result, path, fail, margin

    def _simulate_computer(self, q: str) -> Tuple[str, str, bool, float]:
        freq = 1.0
        fm = re.search(r"(\d+(?:\.\d+)?)\s*ghz", q)
        if fm:
            freq = float(fm.group(1))
        ops = freq * 1e9
        t = np.linspace(0, 1, 50)
        throughput = ops * t
        fig, ax = plt.subplots()
        ax.plot(t, throughput)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Operations")
        path = os.path.join(SIMULATION_DIR, f"computer_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = freq > 5
        result = f"Throughput: {ops:.2e} ops/s"
        margin = 5 - freq
        return result, path, fail, margin

    def _simulate_fluid_flow(self, q: str) -> Tuple[str, str, bool, float]:
        """Generate a simple laminar flow visualization using Plotly."""
        width = 1.0
        height = 1.0
        wm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if wm:
            width = float(wm.group(1))
        hm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if hm:
            height = float(hm.group(1))
        x = np.linspace(0, width, 30)
        y = np.linspace(0, height, 30)
        X, Y = np.meshgrid(x, y)
        vmax = 1.0
        velocity = 4 * vmax * (Y / height) * (1 - Y / height)
        if go is None:  # pragma: no cover - optional dependency
            fig, ax = plt.subplots()
            c = ax.pcolormesh(X, Y, velocity, shading="auto")
            fig.colorbar(c, ax=ax)
            img_path = os.path.join(
                SIMULATION_DIR, f"fluid_{int(time.time()*1000)}.png"
            )
            fig.savefig(img_path, bbox_inches="tight")
            plt.close(fig)
        else:
            fig = go.Figure(data=go.Heatmap(z=velocity, x=x, y=y, colorscale="Viridis"))
            img_path = os.path.join(
                SIMULATION_DIR, f"fluid_{int(time.time()*1000)}.png"
            )
            try:
                fig.write_image(img_path)
            except Exception:
                fig.write_html(img_path.replace(".png", ".html"))
        max_vel = float(velocity.max())
        fail = False
        result = f"Max velocity: {max_vel:.2f} m/s"
        margin = 1.0 - max_vel
        return result, img_path, fail, margin
