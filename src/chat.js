#!/usr/bin/env node
// QVAC LoRA Trainer — chat mode. Talk to the base model, or to the model
// plus your locally-trained LoRA adapter (--lora), entirely on-device via
// Tether's QVAC SDK's completion(). Use this to compare before/after
// fine-tuning: ask "What is QVAC?" without --lora (it hallucinates), then
// with --lora (it answers correctly, from data/qvac-facts.jsonl).

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { loadModel, completion, unloadModel, QWEN3_600M_INST_Q4 } from "@qvac/sdk";

const OUTPUT_DIR = path.resolve("output");

function findAdapter() {
  if (!fs.existsSync(OUTPUT_DIR)) return null;
  const ggufs = fs.readdirSync(OUTPUT_DIR).filter((f) => f.endsWith(".gguf"));
  return ggufs.length ? path.join(OUTPUT_DIR, ggufs[0]) : null;
}

async function main() {
  const useLora = process.argv.includes("--lora");
  let loraPath = null;

  if (useLora) {
    loraPath = findAdapter();
    if (!loraPath) {
      console.error("✖ No trained adapter found in ./output — run `npm run train` first.");
      process.exit(1);
    }
    console.log(`▸ Using fine-tuned adapter: ${loraPath}`);
  } else {
    console.log("▸ Using the base model only (no fine-tuning). Pass --lora to use the trained adapter.");
  }

  console.log("▸ Loading Qwen3 0.6B on-device (first run downloads the weights)...");

  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelType: "llm",
    modelConfig: {
      device: "gpu",
      ctx_size: 512,
      ...(loraPath ? { lora: loraPath } : {}),
    },
    onProgress: (p) => {
      const mb = (n) => (n / 1e6).toFixed(1);
      const line = `  ▸ Downloading ${p.percentage.toFixed(0)}% (${mb(p.downloaded)}/${mb(p.total)} MB)`;
      process.stderr.write(process.stderr.isTTY ? `\r${line}` : `${line}\n`);
      if (p.percentage >= 100) process.stderr.write("\n");
    },
  });

  console.log("▸ Model loaded. Chatting entirely on-device — type 'exit' to quit.\n");

  const history = [];
  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      let userInput;
      try {
        userInput = await rl.question("you> ");
      } catch (error) {
        if (error.code === "ERR_USE_AFTER_CLOSE") break;
        throw error;
      }
      if (!userInput.trim() || ["exit", "quit"].includes(userInput.trim().toLowerCase())) {
        break;
      }

      history.push({ role: "user", content: userInput });

      const run = completion({ modelId, history, stream: true });
      process.stdout.write("qvac> ");
      let reply = "";
      for await (const token of run.tokenStream) {
        process.stdout.write(token);
        reply += token;
      }
      console.log("\n");

      history.push({ role: "assistant", content: reply });
    }
  } finally {
    rl.close();
    await unloadModel({ modelId });
  }
}

main().catch((error) => {
  console.error("✖", error);
  process.exit(1);
});
