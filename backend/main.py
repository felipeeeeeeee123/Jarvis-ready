from features.ai_brain import AIBrain


def main() -> None:
    """Run the JARVIS command-line interface."""
    brain = AIBrain()
    print("\U0001F916 JARVIS is online. Type 'exit' to quit.")

    while True:
        prompt = input("\n\U0001F9E0 You: ").strip()
        if prompt.lower() == "exit":
            print("\U0001F44B JARVIS shutting down.")
            break

        response = brain.ask(prompt)
        print(f"\n\U0001F916 JARVIS: {response}")


if __name__ == "__main__":
    main()

