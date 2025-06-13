from features.ai_brain import AIBrain
from features.web_search import web_search

def main():
    brain = AIBrain()
    print("🤖 JARVIS is online. Type 'exit' to quit.")
    online_mode = True  # Turn this False to go fully offline

    while True:
        prompt = input("🧠 You: ").strip()
        if prompt.lower() == "exit":
            print("👋 JARVIS shutting down.")
            break

        if online_mode and prompt.lower().startswith("search:"):
            query = prompt.split("search:", 1)[-1].strip()
            response = web_search(query)
        else:
            response = brain.ask(prompt)

        print(f"🤖 JARVIS: {response}")
