"""python examples/basic.py: one sentence in English and Spanish, with subtitles."""
import loudness_tts

speech = loudness_tts.speak("Dr. Smith paid $5.50 on 4/5/2026 and left at 9:30 AM.", "english.wav", srt="english.srt")
print(f"english.wav: {speech.duration:.1f} s with {speech.voice.id}; it read: {speech.spoken}")
if speech.voice.credit:
    print("Credit:", speech.voice.credit)

loudness_tts.speak("El Sr. García llegó el 15/10/2026 y pagó 5,50 €.", "spanish.wav", language="es-ES")
