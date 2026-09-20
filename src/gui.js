#!/usr/bin/env node
// QVAC LoRA Trainer — GUI mode. A tiny local web UI (no framework, no
// extra dependencies) on top of the same on-device QVAC pipeline as
// src/train.js and src/chat.js: loadModel() + finetune() to train, and
// loadModel() + completion() to chat with either the base model or the
// model plus the trained LoRA adapter, streamed over Server-Sent Events.
// Everything — training data, the adapter, and every chat message — stays
// on this machine.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { URL, fileURLToPath } from "node:url";
import { loadModel, finetune, completion, unloadModel, QWEN3_600M_INST_Q4 } from "@qvac/sdk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ? Number(process.env.PORT) : 7171;
const PUBLIC_DIR = path.join(__dirname, "..", "public");
const OUTPUT_DIR = path.resolve("output");
const NUM_EPOCHS = 6;

function serveStatic(res) {
  const html = fs.readFileSync(path.join(PUBLIC_DIR, "index.html"));
  res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
  res.end(html);
}

function sseWrite(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function findAdapter() {
  if (!fs.existsSync(OUTPUT_DIR)) return null;
  const ggufs = fs.readdirSync(OUTPUT_DIR).filter((f) => f.endsWith(".gguf"));
  return ggufs.length ? path.join(OUTPUT_DIR, ggufs[0]) : null;
}

// Chat models are loaded lazily and cached per mode so repeated messages
// don't reload the model every time.
const chatModels = {};

async function getChatModel(mode) {
  if (chatModels[mode]) return chatModels[mode];

  const modelConfig = { device: "gpu", ctx_size: 2048 };
  if (mode === "lora") {
    const adapterPath = findAdapter();
    if (!adapterPath) throw new Error("No trained adapter yet — train first");
    modelConfig.lora = adapterPath;
  }

  const modelId = await loadModel({
    modelSrc: QWEN3_600M_INST_Q4,
    modelType: "llm",
    modelConfig,
  });
  chatModels[mode] = modelId;
  return modelId;
}

async function main() {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (req.method === "GET" && url.pathname === "/") {
      return serveStatic(res);
    }

    if (req.method === "GET" && url.pathname === "/api/train") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      let modelId;
      try {
        sseWrite(res, "log", { line: "▸ Loading Qwen3 0.6B on-device..." });
        modelId = await loadModel({
          modelSrc: QWEN3_600M_INST_Q4,
          modelType: "llm",
          modelConfig: { device: "gpu", ctx_size: 512 },
        });
        sseWrite(res, "log", { line: `▸ Model loaded: ${modelId}` });
        sseWrite(res, "log", { line: "▸ Fine-tuning on-device with LoRA..." });

        const handle = finetune({
          modelId,
          options: {
            trainDatasetDir: path.resolve("data/qvac-facts.jsonl"),
            validation: { type: "dataset", path: path.resolve("data/qvac-facts-eval.jsonl") },
            numberOfEpochs: NUM_EPOCHS,
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
          sseWrite(res, "log", {
            line: `epoch=${tick.current_epoch + 1}/${NUM_EPOCHS} step=${tick.global_steps} ${phase} loss=${tick.loss?.toFixed(4)} acc=${tick.accuracy?.toFixed(4)}`,
          });
          const pct = Math.min(100, ((tick.current_epoch + tick.current_batch / tick.total_batches) / NUM_EPOCHS) * 100);
          sseWrite(res, "progress", { pct });
        }

        const result = await handle.result;
        sseWrite(res, "log", { line: `✔ Fine-tuning finished: ${result.status}` });
        sseWrite(res, "done", { stats: result.stats });

        // Invalidate any cached fine-tuned chat model so the next chat
        // request picks up the freshly retrained adapter.
        if (chatModels.lora) {
          await unloadModel({ modelId: chatModels.lora }).catch(() => {});
          delete chatModels.lora;
        }
      } catch (error) {
        sseWrite(res, "error", { error: error.message });
      } finally {
        if (modelId) await unloadModel({ modelId, clearStorage: false }).catch(() => {});
        res.end();
      }
      return;
    }

    if (req.method === "GET" && url.pathname === "/api/chat") {
      const message = url.searchParams.get("message");
      const mode = url.searchParams.get("mode") === "lora" ? "lora" : "base";
      if (!message) {
        res.writeHead(400, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ error: "Missing message" }));
      }

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      try {
        const modelId = await getChatModel(mode);
        const run = completion({
          modelId,
          history: [{ role: "user", content: message }],
          stream: true,
        });
        for await (const token of run.tokenStream) {
          sseWrite(res, "token", { token });
        }
        sseWrite(res, "done", {});
      } catch (error) {
        sseWrite(res, "error", { error: error.message });
      } finally {
        res.end();
      }
      return;
    }

    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not found");
  });

  server.listen(PORT, () => {
    console.log(`▸ QVAC LoRA Trainer GUI ready at http://localhost:${PORT}`);
  });

  const shutdown = async () => {
    console.log("\n▸ Shutting down...");
    server.close();
    await Promise.all(Object.values(chatModels).map((id) => unloadModel({ modelId: id }).catch(() => {})));
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((error) => {
  console.error("✖", error);
  process.exit(1);
});
