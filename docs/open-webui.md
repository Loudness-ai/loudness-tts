# Free local voices for Open WebUI (and other OpenAI-compatible apps)

1. `pip install local-tts` and start the server: `local-tts serve --port 8880` (Docker: `docker run -p 8880:8880 -v local-tts:/data ghcr.io/loudness-ai/local-tts`).
2. In Open WebUI: **Admin Panel → Settings → Audio → Text-to-Speech**: engine **OpenAI**, API base URL
   `http://127.0.0.1:8880/v1` (from another container: `http://host.docker.internal:8880/v1`), any API key (or the one you
   set with `--api-key`), model `tts-1`, voice `nova` (or any id from `local-tts voices`).
3. Save, and press the speaker icon under any answer.

The same settings work in LibreChat, AnythingLLM, SillyTavern and any app that accepts an OpenAI-compatible
text-to-speech URL. OpenAI voice names map to strong English (US) voices: nova, shimmer, coral, sage and marin are female;
echo, onyx, ash, ballad, verse and cedar are male; alloy and fable are the top voices overall. For other languages pass a
voice id such as `de_DE-mls-medium-s162`, or start the server with `--language de-DE`.
