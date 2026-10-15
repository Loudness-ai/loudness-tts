// SPDX-License-Identifier: GPL-3.0-or-later
/** Stable error codes, the same as the Python edition's. */
export type ErrorCode = "VOICE_NOT_FOUND" | "DOWNLOAD_FAILED" | "CHECKSUM_MISMATCH" | "OFFLINE" | "TEXT_INVALID"
  | "TEXT_TOO_LONG" | "NOTHING_TO_SPEAK" | "OPTION_INVALID" | "CANCELLED" | "TIMEOUT" | "ENGINE_FAILED"
  | "FORMAT_UNAVAILABLE" | "NOT_AVAILABLE_HERE";

/** Every error local-tts throws: a stable `code` for programs and a plain message for people. */
export class LocalTTSError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "LocalTTSError";
    this.code = code;
  }
  toJSON(): { code: ErrorCode; message: string } {
    return { code: this.code, message: this.message };
  }
}

export class CancelledError extends LocalTTSError {
  constructor(message = "Speech generation was cancelled.") {
    super("CANCELLED", message);
    this.name = "CancelledError";
  }
}
