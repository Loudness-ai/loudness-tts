# Security

Report vulnerabilities privately to asad@loudness.ai (please do not open a public issue). We aim to reply within a week.

The server listens on 127.0.0.1 by default. To accept other machines, use `--host 0.0.0.0` together with `--api-key`
(or `LOUDNESS_TTS_API_KEY`), and preferably a reverse proxy with TLS. Request bodies are limited to 1 MB and text to
`--max-characters`; text is never logged or stored. Voice files are only used after their size and checksum match the
catalogue.
