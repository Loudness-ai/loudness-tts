#!/bin/sh
# Builds the espeak-ng WebAssembly module (src/wasm/espeak.mjs) from the espeak-ng commit Piper 1.8 uses, with the
# same clause handling as Piper's espeakbridge.c (bridge.c here). Needs Emscripten (emcc, emcmake) and CMake.
#   sh espeak/build.sh [work dir]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-"$HERE/.build"}
COMMIT=724808c5a83f9ef95fdd0db886ba7ba537ff224a
mkdir -p "$WORK"
if [ ! -d "$WORK/espeak-ng" ]; then
  git clone --quiet https://github.com/espeak-ng/espeak-ng.git "$WORK/espeak-ng"
fi
cd "$WORK/espeak-ng"
git checkout --quiet "$COMMIT"
git checkout --quiet -- src/include/compat/wctype.h
git apply "$HERE/wctype.patch"
mkdir -p "$WORK/build" && cd "$WORK/build"
emcmake cmake "$WORK/espeak-ng" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF \
  -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=OFF -DUSE_SPEECHPLAYER=OFF -DENABLE_TESTS=OFF \
  -DCOMPILE_INTONATIONS=OFF -DESPEAK_BUILD_MANPAGES=OFF \
  "-DCMAKE_C_FLAGS=-D_FILE_OFFSET_BITS=64 -I$WORK/espeak-ng/src/ucd-tools/src/include" >/dev/null
cmake --build . --target espeak-ng -j4 >/dev/null
emcc -O3 -flto "$HERE/bridge.c" -I"$WORK/espeak-ng/src/include" src/libespeak-ng/libespeak-ng.a src/ucd-tools/libucd.a \
  -o "$HERE/../src/wasm/espeak.mjs" -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEspeak -sENVIRONMENT=web,worker \
  -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576 -sINITIAL_MEMORY=16777216 -sSINGLE_FILE=1 -sFILESYSTEM=1 -sEXIT_RUNTIME=0 \
  -sEXPORTED_FUNCTIONS=_lt_initialize,_lt_set_voice,_lt_phonemes,_free \
  -sEXPORTED_RUNTIME_METHODS=FS,UTF8ToString,stringToNewUTF8
echo "built src/wasm/espeak.mjs"
