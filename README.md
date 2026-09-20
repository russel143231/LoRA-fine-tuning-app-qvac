# QVAC LoRA Trainer

A local, on-device LoRA fine-tuning app built with
[Tether's QVAC SDK](https://github.com/tetherto/qvac) — no cloud call, no
API key, no bill. It trains a small LoRA adapter on top of Qwen3 0.6B
entirely on your own machine, teaching it facts it doesn't otherwise know,
then lets you compare the base model against the fine-tuned one live.

It calls the QVAC SDK's `loadModel()`, `finetune()`, and `completion()`
functions directly.

## What it does

The base Qwen3 0.6B model has no idea what "QVAC" is — ask it and it
hallucinates ("Quantum Vast Algorithm", "Quantum Voltage Act", etc., a new
guess every time). This app fine-tunes a LoRA adapter on a small
hand-written dataset of QVAC facts (`data/qvac-facts.jsonl`), entirely
on-device, and lets you directly compare answers before and after.

```bash
npm run train                # loadModel() + finetune() on data/qvac-facts.jsonl
npm run chat                 # chat with the base model
npm run chat -- --lora       # chat with the base model + your trained adapter
```

Or use the GUI (`npm run gui`) for a side-by-side train/compare interface
with live streaming.

## Verified output

Actually run end-to-end on 2026-09-19/20 (Windows, GPU backend):

**Before fine-tuning** ("What is QVAC?"):
> Maybe QVAC is a specific term in some field... perhaps it's "Quantum
> Voltage Act"... [rambles indefinitely, never converges]

**Training** (6 epochs, 12 training examples + 3 validation examples):
```
epoch=6 step=72 batch=12/12 train loss=0.6493 acc=0.8252
✔ Fine-tuning finished: { status: 'COMPLETED', stats: { train_loss: 0.649, val_accuracy: 0.672, ... } }
```
Loss dropped from ~0.97 (epoch 3) to ~0.65 (epoch 6); training accuracy
rose to 82%+.

**After fine-tuning** ("What is QVAC?", via the GUI):
> QVAC is a software library that runs on-device AI inference on a
> device's own GPU.

A small model with a tiny dataset won't produce a perfect answer, but the
shift from total hallucination to a coherent, largely accurate answer is
clear and directly attributable to the training data.

## SDK version

Built and tested against `@qvac/sdk` **v0.19.1** (see [package.json](package.json)).

## Requirements

- Node.js `>= 22.17`
- A machine that meets [QVAC's system requirements](https://docs.qvac.tether.io/system-requirements)
- ~400 MB free disk space for the base model weights on first run

## Install

```bash
git clone https://github.com/<your-username>/qvac-lora-trainer.git
cd qvac-lora-trainer
npm install
```

## Run

```bash
npm run train              # fine-tune (writes output/trained-lora-adapter.gguf)
npm run chat                # chat with the base model
npm run chat -- --lora      # chat with the fine-tuned model
npm run gui                  # web UI: train + side-by-side compare
```

`data/qvac-facts.jsonl` and `data/qvac-facts-eval.jsonl` are the bundled
training/validation sets (SFT chat format). Swap in your own facts to
fine-tune the model on a different topic.

## GUI mode

`npm run gui` starts a local server (`http://localhost:7171` by default)
with two panels: **Train** (runs `finetune()`, streams live loss/accuracy
per step via Server-Sent Events, shows a progress bar) and **Compare** (a
chat box with a Base/Fine-tuned toggle, so you can ask the same question to
both and see the difference immediately). No data ever leaves your
machine. Override the port with `PORT=8080 npm run gui`.

## How it uses QVAC

```js
import { loadModel, finetune, completion, unloadModel, QWEN3_600M_INST_Q4 } from "@qvac/sdk";

const modelId = await loadModel({
  modelSrc: QWEN3_600M_INST_Q4,
  modelType: "llm",
  modelConfig: { device: "gpu", ctx_size: 512 },
});

const handle = finetune({
  modelId,
  options: {
    trainDatasetDir: "data/qvac-facts.jsonl",
    validation: { type: "dataset", path: "data/qvac-facts-eval.jsonl" },
    numberOfEpochs: 6,
    learningRate: 1e-4,
    loraModules: "attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down",
    assistantLossOnly: true,
    outputParametersDir: "output",
  },
});
for await (const tick of handle.progressStream) {
  console.log(tick.current_epoch, tick.loss, tick.accuracy);
}
const result = await handle.result; // writes output/trained-lora-adapter.gguf

// Later, load the adapter back in for inference:
const tunedModelId = await loadModel({
  modelSrc: QWEN3_600M_INST_Q4,
  modelType: "llm",
  modelConfig: { lora: "output/trained-lora-adapter.gguf" },
});
const run = completion({ modelId: tunedModelId, history: [{ role: "user", content: "What is QVAC?" }] });
```

See [src/train.js](src/train.js) and [src/chat.js](src/chat.js) for the
full implementation.

## License

[MIT](LICENSE)
