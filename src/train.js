#!/usr/bin/env node
// QVAC LoRA Trainer — fine-tunes a small local LLM entirely on-device via
// Tether's QVAC SDK. No cloud call, no API key: the base model downloads
// once, then every training step runs on this machine, producing a small
// .gguf LoRA adapter you can load back into completion().
//
// The bundled dataset (data/qvac-facts.jsonl) teaches the model what QVAC
// actually is — the base model has no idea and hallucinates wildly (see
// README), so this is a clear, checkable before/after demo.

import path from "node:path";
import { loadModel, finetune, unloadModel, QWEN3_600M_INST_Q4 } from "@qvac/sdk";

const OUTPUT_DIR = path.resolve("output");

async function main() {
  console.log("▸ Loading Qwen3 0.6B on-device (first run downloads the weights)...");

  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelType: "llm",
    modelConfig: { device: "gpu", ctx_size: 512 },
    onProgress: (p) => {
      const mb = (n) => (n / 1e6).toFixed(1);
      const line = `  ▸ Downloading ${p.percentage.toFixed(0)}% (${mb(p.downloaded)}/${mb(p.total)} MB)`;
      process.stderr.write(process.stderr.isTTY ? `\r${line}` : `${line}\n`);
      if (p.percentage >= 100) process.stderr.write("\n");
    },
  });

  console.log(`▸ Model loaded: ${modelId}`);
  console.log("▸ Fine-tuning on-device with LoRA (data/qvac-facts.jsonl)...\n");

  const handle = finetune({
    modelId,
    options: {
      trainDatasetDir: path.resolve("data/qvac-facts.jsonl"),
      validation: { type: "dataset", path: path.resolve("data/qvac-facts-eval.jsonl") },
      numberOfEpochs: 6,
      learningRate: 1e-4,
      lrMin: 1e-8,
      loraModules: "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down",
      assistantLossOnly: true,
      checkpointSaveSteps: 10,
      checkpointSaveDir: path.join(OUTPUT_DIR, "checkpoints"),
      outputParametersDir: OUTPUT_DIR,
    },
  });

  for await (const tick of handle.progressStream) {
    const phase = tick.is_train ? "train" : "val";
    console.log(
      `  epoch=${tick.current_epoch + 1} step=${tick.global_steps} batch=${tick.current_batch}/${tick.total_batches} ${phase} loss=${tick.loss?.toFixed(4)} acc=${tick.accuracy?.toFixed(4)} eta=${Math.round(tick.eta_ms / 1000)}s`
    );
  }

  const result = await handle.result;
  console.log("\n✔ Fine-tuning finished:", result);
  console.log(`\nAdapter saved under: ${OUTPUT_DIR}`);
  console.log("Run `npm run chat -- --lora` to talk to the fine-tuned model.");

  await unloadModel({ modelId, clearStorage: false });
}

main().catch((error) => {
  console.error("✖", error);
  process.exit(1);
});
