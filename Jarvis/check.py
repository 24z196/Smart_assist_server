import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("YOUR_GOOGLE_API_KEY"))

print("Checking for models with Live API (Multimodal) support...\n")

# List all models and check their supported actions
for m in client.models.list():
    # In 2026, the specific capability is called 'BIDI_GENERATE_CONTENT' 
    # or categorized under specific 'live' tags.
    if "bidiGenerateContent" in m.supported_actions:
        print(f"✅ FOUND: {m.name}")
        print(f"   Display Name: {m.display_name}")
        print(f"   Description: {m.description}\n")

print("If the list is empty, ensure your API key is correct and your region supports Gemini Live.")