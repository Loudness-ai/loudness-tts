// SPDX-License-Identifier: GPL-3.0-or-later
// The same clause handling as Piper's espeakbridge.c, compiled to WebAssembly so the JavaScript edition produces exactly
// the phonemes the Python edition (and Piper itself) produces.
#include <stdlib.h>
#include <string.h>
#include <espeak-ng/speak_lib.h>
#include <emscripten/emscripten.h>

#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000
#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000
#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

EMSCRIPTEN_KEEPALIVE int lt_initialize(const char *data_dir) {
    return espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir, 0);
}

EMSCRIPTEN_KEEPALIVE int lt_set_voice(const char *voice) { return espeak_SetVoiceByName(voice); }

// Returns a malloc'd string: one record per clause, "phonemes\x1Fterminator\x1Fs" (s is 1 at a sentence end), records
// separated by \x1E. The caller frees it.
EMSCRIPTEN_KEEPALIVE char *lt_phonemes(const char *text) {
    size_t size = 256, used = 0;
    char *out = malloc(size);
    if (!out) return NULL;
    out[0] = 0;
    while (text != NULL) {
        int terminator = 0;
        const char *t = "";
        const char *phonemes = espeak_TextToPhonemesWithTerminator((const void **)&text, espeakCHARS_AUTO,
                                                                   espeakPHONEMES_IPA, &terminator);
        terminator &= 0x000FFFFF;
        if (terminator == CLAUSE_PERIOD) t = ".";
        else if (terminator == CLAUSE_QUESTION) t = "?";
        else if (terminator == CLAUSE_EXCLAMATION) t = "!";
        else if (terminator == CLAUSE_COMMA) t = ",";
        else if (terminator == CLAUSE_COLON) t = ":";
        else if (terminator == CLAUSE_SEMICOLON) t = ";";
        if (!phonemes) phonemes = "";
        size_t need = used + strlen(phonemes) + strlen(t) + 5;
        if (need > size) {
            while (need > size) size *= 2;
            char *grown = realloc(out, size);
            if (!grown) { free(out); return NULL; }
            out = grown;
        }
        used += sprintf(out + used, "%s%s\x1F%s\x1F%c", used ? "\x1E" : "", phonemes, t,
                        (terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE ? '1' : '0');
    }
    return out;
}
