#!/usr/bin/env node
// Transcribes an mp4's audio with whisper.cpp and writes word-level
// timestamps as JSON: [{ text, startMs, endMs }, ...]
//
// Usage:
//   node scripts/transcribe.mjs <input.mp4> [--out <output.json>] [--model <ggml-model.bin>]

import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const whisperDir = path.join(projectRoot, "bin", "whisper.cpp");
const whisperCli = path.join(whisperDir, "build", "bin", "whisper-cli");
const defaultModel = path.join(whisperDir, "models", "ggml-base.en.bin");

function parseArgs(argv) {
  const args = { input: null, out: null, model: defaultModel };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--out") {
      args.out = argv[++i];
    } else if (arg === "--model") {
      args.model = argv[++i];
    } else if (!args.input) {
      args.input = arg;
    }
  }
  return args;
}

function run(cmd, cmdArgs) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, cmdArgs, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${cmd} exited with code ${code}\n${stderr}`));
      }
    });
  });
}

// whisper.cpp's offsets are already in milliseconds (its internal
// centisecond ticks * 10), one "segment" per word when run with -ml 1 -sow.
function toWords(whisperJson) {
  const segments = whisperJson.transcription ?? [];
  return segments
    .map((segment) => ({
      text: segment.text.trim(),
      startMs: segment.offsets.from,
      endMs: segment.offsets.to,
    }))
    .filter((word) => word.text.length > 0);
}

async function main() {
  const { input, out, model } = parseArgs(process.argv.slice(2));

  if (!input) {
    console.error(
      "Usage: node scripts/transcribe.mjs <input.mp4> [--out <output.json>] [--model <ggml-model.bin>]",
    );
    process.exit(1);
  }

  const inputPath = path.resolve(input);
  const outputPath = out
    ? path.resolve(out)
    : inputPath.replace(/\.[^.]+$/, "") + ".words.json";

  const workDir = await mkdtemp(path.join(tmpdir(), "transcribe-"));
  const wavPath = path.join(workDir, "audio.wav");
  const whisperOutBase = path.join(workDir, "audio");

  try {
    // Extract mono 16kHz PCM WAV, the format whisper.cpp expects.
    await run("ffmpeg", [
      "-y",
      "-i",
      inputPath,
      "-ar",
      "16000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      wavPath,
    ]);

    // -ml 1 -sow splits segments at word boundaries, giving one
    // "segment" per word with its own start/end offset.
    await run(whisperCli, [
      "-m",
      model,
      "-f",
      wavPath,
      "-ml",
      "1",
      "-sow",
      "-oj",
      "-of",
      whisperOutBase,
      "-np",
      "-nt",
    ]);

    const raw = await readFile(`${whisperOutBase}.json`, "utf8");
    const words = toWords(JSON.parse(raw));

    await writeFile(outputPath, JSON.stringify(words, null, 2));
    console.log(`Wrote ${words.length} words to ${outputPath}`);
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
