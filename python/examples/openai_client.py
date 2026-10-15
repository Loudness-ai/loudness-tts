"""Start the server first (local-tts serve), then: pip install openai && python examples/openai_client.py"""
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8880/v1", api_key="not-needed")
client.audio.speech.create(model="tts-1", voice="nova", input="Hello from my own speech server!",
                           response_format="wav").write_to_file("openai-client.wav")
print("openai-client.wav written")
