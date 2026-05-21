import factorio_rcon
from fle.env import FactorioInstance
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


RCON_HOST = "127.0.0.1"
RCON_PORT = 27000      
RCON_PASSWORD = "factorio"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def main():
    instance = FactorioInstance(
        address="127.0.0.1",
        tcp_port=27000,
        inventory={},
    )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = instance.get_system_prompt()

    print()
    print(system_prompt)
    print("\n")
    # Seed Gemini with the FLE system prompt so it knows the API
    chat = model.start_chat(history=[
        {"role": "user", "parts": [system_prompt]},
        {"role": "model", "parts": ["Understood. I will use the FLE Python API to play Factorio."]},
    ])

    last_output = "Game just started. No previous output."

    max_steps = 20
    for step in range(max_steps):
        # The observation IS the output from the last eval
        message = (
            f"Step {step}.\n"
            f"Output from last action:\n{last_output}\n\n"
            f"Write Python code using the FLE API to progress. "
            f"Respond with raw Python only, no markdown, no explanation."
        )
        response = chat.send_message(message)
        code = response.text.replace("```python", "").replace("```", "").strip()

        print(f"\n--- Step {step} ---")
        print(f"Gemini:\n{code}")

        # stdout/stderr of the executed code becomes the next observation
        result, error = instance.eval(code)
        last_output = f"Result: {result}\nError: {error}" if error else str(result)

        print(f"Output: {last_output}")
    
if __name__ == "__main__":
    main()