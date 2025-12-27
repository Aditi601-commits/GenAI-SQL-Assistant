import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyDfuc9zm-cqqksYD6AOPZXKAqyI4fiRgG0"
genai.configure(api_key=GOOGLE_API_KEY)

print("Checking available models for your key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")