import ast
import os

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git"}
INTERNAL_PREFIXES = {"backend", "gui", "features", "utils"}
SKIP_MODULES = {"bpy"}
SKIP_FILES = {os.path.join("backend", "features", "autotrade.py")}
PROJECT_ROOT = os.getcwd()
COMBINED_FILE = "combined_jarvis.py"

future_imports: set[str] = set()
regular_imports: set[str] = set()
file_contents: list[str] = []


def is_internal(module: str | None) -> bool:
    """Return True if the given module resolves to a file or package inside the
    project directory or is a relative import."""
    if not module:
        return True
    root = module.split(".")[0]
    if root in INTERNAL_PREFIXES:
        return True
    return (
        os.path.isdir(os.path.join(PROJECT_ROOT, root))
        or os.path.isfile(os.path.join(PROJECT_ROOT, f"{root}.py"))
    )


for root, dirs, files in os.walk(PROJECT_ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    for file in files:
        if not file.endswith(".py") or file in {COMBINED_FILE, "combine_files.py"} or file.startswith("."):
            continue
        path = os.path.join(root, file)
        rel_path = os.path.relpath(path, PROJECT_ROOT)
        if rel_path in SKIP_FILES:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            print(f"⚠️ Skipped {rel_path} due to read error: {e}")
            continue
        try:
            tree = ast.parse(source, filename=rel_path)
        except Exception as e:
            print(f"⚠️ Skipped {rel_path} due to parse error: {e}")
            continue

        lines = source.splitlines()
        remove: set[int] = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node) or ""
                segment = segment.strip()
                is_future = segment.startswith("from __future__")
                internal = False
                skip = False
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    internal = node.level > 0 or is_internal(module)
                    skip = module and module.split(".")[0] in SKIP_MODULES
                else:  # ast.Import
                    internal = all(is_internal(alias.name) for alias in node.names)
                    skip = any(alias.name.split(".")[0] in SKIP_MODULES for alias in node.names)

                if is_future:
                    future_imports.add(segment)
                elif not internal and not skip:
                    regular_imports.add(segment)

                if is_future or internal:
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno) - 1
                    for i in range(start, end + 1):
                        remove.add(i)
            elif (
                isinstance(node, ast.If)
                and isinstance(getattr(node, "test", None), ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
            ):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno) - 1
                for i in range(start, end + 1):
                    remove.add(i)

        kept_lines = [line for idx, line in enumerate(lines) if idx not in remove]
        code = "\n".join(kept_lines).rstrip()
        file_contents.append(f"\n# === FILE: {rel_path} ===\n{code}\n")

with open(COMBINED_FILE, "w", encoding="utf-8") as out:
    for imp in sorted(future_imports):
        out.write(f"{imp}\n")
    out.write("\n# Combined JARVIS Python File\n")
    for imp in sorted(regular_imports):
        out.write(f"{imp}\n")
    out.write("\n")
    for content in file_contents:
        out.write(content)

print(f"✅ Successfully regenerated {COMBINED_FILE}")
