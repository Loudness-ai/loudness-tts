export interface EspeakModule {
  FS: {
    mkdir(path: string): void;
    writeFile(path: string, data: Uint8Array | string): void;
    analyzePath(path: string): { exists: boolean };
  };
  UTF8ToString(ptr: number): string;
  stringToNewUTF8(text: string): number;
  _lt_initialize(path: number): number;
  _lt_set_voice(voice: number): number;
  _lt_phonemes(text: number): number;
  _free(ptr: number): void;
}
export default function createEspeak(options?: Record<string, unknown>): Promise<EspeakModule>;
