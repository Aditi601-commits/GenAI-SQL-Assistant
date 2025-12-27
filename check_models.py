import google.generativeai as genai
import toml  # Library to read the secrets file

# 1. Manually read the secrets file
try:
    data = toml.load(".streamlit/secrets.toml")
    api_key = data["GOOGLE_API_KEY"]
except FileNotFoundError:
    print("❌ Could not find .streamlit/secrets.toml")
    exit()

# 2. Configure Google AI
genai.configure(api_key=api_key)

# 3. List Models
print("Checking available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")