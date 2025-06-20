import os

IGNORED_DIRS = {'venv', '.venv', '__pycache__'}
SEARCH_DIRS = ['.']


def iter_py_files():
    py_files = []
    for base in SEARCH_DIRS:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for name in files:
                if name.endswith('.py'):
                    path = os.path.relpath(os.path.join(root, name), '.')
                    py_files.append(path)
    root_files = sorted([f for f in py_files if '/' not in f])
    backend_files = sorted([f for f in py_files if f.startswith('backend/')])
    gui_files = sorted([f for f in py_files if f.startswith('gui/')])
    return root_files + backend_files + gui_files


def collect_imports_and_code(files):
    imports = []
    seen = set()
    sections = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        body_start = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped not in seen:
                    imports.append(line.rstrip())
                    seen.add(stripped)
                body_start = idx + 1
            elif stripped == '' or stripped.startswith('#'):
                body_start = idx + 1
            else:
                break
        body = ''.join(lines[body_start:])
        sections.append((path, body))
    return imports, sections


def main():
    files = iter_py_files()
    imports, sections = collect_imports_and_code(files)
    with open('combined_jarvis.py', 'w', encoding='utf-8') as out:
        out.write('# Auto-generated combined file\n')
        for imp in imports:
            out.write(f'{imp}\n')
        out.write('\n')
        for path, body in sections:
            out.write(f'# === FILE: {path} ===\n')
            out.write(body)
            if not body.endswith('\n'):
                out.write('\n')


if __name__ == '__main__':
    main()
