from features.ai_brain import AIBrain

def main():
    brain = AIBrain()
    print("🤖 JARVIS is online. Type 'exit' to quit.")

    while True:
        prompt = input("🧠 You: ").strip()
        if prompt.lower() == "exit":
            print("👋 JARVIS shutting down.")
            break

        response = brain.ask(prompt)
        print(f"🤖 JARVIS: {response}")

if __name__ == "__main__":
    main()
