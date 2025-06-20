import os
import re

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git"}
PROJECT_ROOT = os.getcwd()

combined_file = "combined_jarvis.py"
future_imports = set()
regular_imports = set()
file_contents = []

for root, dirs, files in os.walk(PROJECT_ROOT):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for file in files:
        if (
            file.endswith(".py")
            and file != combined_file
            and file != "combine_files.py"
            and not file.startswith(".")
        ):
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, PROJECT_ROOT)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    code = []
                    for line in lines:
                        if "fetch --progress" in line or re.match(r"^[0-9a-f]{40}", line):
                            continue
                        if line.strip().startswith("from __future__ import"):
                            future_imports.add(line.strip())
                        elif line.strip().startswith("import ") or line.strip().startswith("from "):
                            regular_imports.add(line.strip())
                        else:
                            code.append(line)
                    file_contents.append(f"\n# === FILE: {rel_path} ===\n{''.join(code)}")
            except Exception as e:
                print(f"⚠️ Skipped {rel_path} due to read error: {e}")

with open(combined_file, "w", encoding="utf-8") as out:
    # future imports must be at the top
    for imp in sorted(future_imports):
        out.write(f"{imp}\n")

    out.write("\n# Combined JARVIS Python File\n")

    for imp in sorted(regular_imports):
        out.write(f"{imp}\n")
    out.write("\n")
    for content in file_contents:
        out.write(content)

print(f"✅ Successfully regenerated {combined_file}")
