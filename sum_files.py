import os
import sys

EXTS = {'.py', '.md', '.json', '.txt', '.sh', '.csv'}


def iter_files(base: str):
    for root, _, files in os.walk(base):
        for name in files:
            if any(name.endswith(ext) for ext in EXTS):
                yield os.path.join(root, name)


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else '.'
    total_lines = 0
    total_bytes = 0
    for path in iter_files(base):
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except Exception:
            continue
        total_bytes += len(data)
        total_lines += data.count(b'\n') + 1
    print(f'Total lines: {total_lines}')
    print(f'Total bytes: {total_bytes}')


if __name__ == '__main__':
    main()
