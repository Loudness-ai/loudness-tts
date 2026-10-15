"""python examples/streaming.py: write speech sentence by sentence as it is made (here into a WAV file; send each
chunk to an audio player or a network socket the same way)."""
import time
import wave

import loudness_tts

tts = loudness_tts.TTS()
tts.preload()
started = time.monotonic()
with wave.open("streamed.wav", "wb") as out:
    for chunk in tts.stream("Streaming starts at once. Each sentence arrives as soon as it is ready. Like this one."):
        if chunk.index == 0:
            out.setnchannels(1), out.setsampwidth(2), out.setframerate(chunk.sample_rate)
            print(f"first audio after {time.monotonic() - started:.2f} s")
        out.writeframes(chunk.pcm())
