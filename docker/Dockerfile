# loudness-tts speech server: docker run -p 8880:8880 -v loudness-tts:/data ghcr.io/loudness-ai/loudness-tts
# Voices download into /data on first use (or set LOUDNESS_TTS_PRELOAD="en-US es-ES" to fetch them at start).
FROM python:3.12-slim
ARG VERSION=0.1.1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 tts && mkdir /data && chown tts /data
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/loudness_tts-*.whl && rm /tmp/*.whl
USER tts
ENV LOUDNESS_TTS_HOME=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8880
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8880/health',timeout=4)"
ENTRYPOINT ["sh", "-c", "for l in ${LOUDNESS_TTS_PRELOAD:-}; do loudness-tts download --language $l -q; done; exec loudness-tts serve --host 0.0.0.0 --port 8880 \"$@\"", "--"]
