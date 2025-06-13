
import json
import os

class MemoryManager:
    def __init__(self, path='data/memory.json'):
        self.path = path
        self.memory = {}

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.memory = json.load(f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=4)
