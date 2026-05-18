# CLAUDE.md — ai-toolkit ROCm fork

Standing context for Claude Code sessions in this repo. Keep this concise;
the full engineering history lives in the two logs linked at the bottom.

## What this fork is

`deniskokarev/ai-toolkit`, branch **`rocm-integration`** off `main`. Goal: a
working `ostris/ai-toolkit` that trains/serves on this box's two AMD GPUs under
ROCm 7.2.3, upstreamable to `ostris/ai-toolkit` later. Selective port of
upstream PR #563 (iGavroche), not #559.

## Hardware / GPU enumeration (do not get this backwards)

| torch device | card | VRAM | arch |
|---|---|---|---|
| `cuda:0` | Radeon **R9700** | **32 GB** | gfx1201 |
| `cuda:1` | RX **6900 XT** | **16 GB** | gfx1030 |

Smaller index = bigger card. ROCm xformers wheels are broken for both archs
(CK kernels target CDNA only) — a from-source build is the only real fix and
is out of scope.

## Active kludges (each is a documented workaround, not a preference)

- `run.py` — force the **math SDPA backend** on ROCm. AOTriton in the torch
  ROCm wheel only ships flash/mem-efficient kernels for
  gfx90a/gfx942/gfx1100; gfx1030/gfx1201 raise `[AOTriton] Accelerated SDPA
  only supports ...`. (commit `4ce6f12`)
- `jobs/process/BaseSDTrainProcess.py` — silently rewrite `xformers: true` →
  SDPA on HIP, because the ROCm xformers wheels are broken. (commit `972e56d`)
- `toolkit/models/base_model.py` + `flux2_model.py` — `te_device` config key
  wired up (was dead) so the text encoder can sit on a separate GPU.
  (commit `b18ce7c`)
- `ui/.../api/gpu` — AMD GPU detection via `rocm-smi`/`amd-smi`, columns
  looked up by CSV header. (commits `0591d79`, `b3a84c3`)
- `startJob` — child stderr captured to a rotating file + ROCm env passed
  through. (commit `ed82574`)
- `toolkit/util/quantize.py` — import-time, HIP-gated shim of
  `optimum.quanto.tensor.qbits.qbits`: None-safe `version` + force the
  portable `QBitsTensor` (skip AWQ/TinyGemm). quanto 0.2.4 sub-byte
  (`qint4`/`qint2`) else crashes on ROCm — `version.parse(torch.version.cuda
  =None)` and the CPU disjunct's CUDA-only `aten::_convert_weight_to_int4pack`.
  (commit pending)
- `toolkit/optimizer.py` — on HIP, transparently remap bitsandbytes 8-bit
  optimizers to their nearest non-bnb equivalent (`adamw8bit→adamw`,
  `adam8bit→adam`, `lion8bit→lion`, `ademamix8bit→adamw`) with a warning. bnb
  8-bit kernels don't work on gfx1201/gfx1030 → `AdamW8bit.step()` ≈ 20 s for
  a rank-32 LoRA (⅓ of the step). 8-bit state is irrelevant at LoRA sizes.
  (commit pending)
- `toolkit/data_loader.py` — only pass `prefetch_factor` when
  `num_workers > 0` (PyTorch rejects it otherwise; `num_workers: 0` is used
  to avoid worker-fork host-RAM blowups). (commit pending)
- `jobs/process/BaseSDTrainProcess.py` — `flush()` right after
  `accelerator.prepare(self.sd.unet)`: reclaim the ~30 GB pinned CPU copy of
  the transformer before DataLoader workers fork (else host global OOM at
  step 0). Plus `flux2_model.py` `del transformer_state_dict` + the
  caching-defer gate. (commit pending)

Delete any kludge the day its upstream cause is fixed; each commit message
documents the full reasoning.

## How to run

```bash
source ~/ai-toolkit/venv/bin/activate          # Python backend

cd ~/ai-toolkit/ui && npm run dev               # UI dev  → :3000
cd ~/ai-toolkit/ui && npm run build_and_start   # UI prod → :8675
```

Run training from CLI with `AMD_SERIALIZE_KERNEL=0 TORCH_USE_HIP_DSA=0`
(the ROCm debug defaults in `run.py` have negligible perf impact and are
best treated as opt-in).

## Key model-loading facts (FLUX.2)

- TE is **Mistral-Small-3.1-24B** (`extensions_built_in/diffusion_models/flux2/flux2_model.py`).
- Quant types: `uint2..uint8`, `int8`, `float8`/`qfloat8` (`toolkit/util/quantize.py`).
  There is no "Q5"; that would be `uint5`.
- torchao on this ROCm build has **no fused quantized-linear kernel** — every
  quantized Linear falls back to dequantize-whole-weight-then-matmul. uint4
  shrinks *resident* weight size but pays it back (plus an int32 temp) per
  layer during forward. This is why low-bit TE still OOMs in forward.
- `layer_offloading` offloads to **CPU pinned RAM only** — no GPU→GPU
  sharding exists (`toolkit/memory_management/manager*.py`). The loader has
  no `device_map`/tensor-parallel path; the only model-parallel axis is
  component-level `te_device`.

## FLUX.2 LoRA memory pipeline (where it OOMs)

Sizes: transformer `flux2-dev.safetensors` = **60 GB bf16 ≈ ~30 B params**
(qfloat8 ~30 GB · qint4 ~15–16 GB). Mistral-24B TE ≈ 48 GB bf16. Box =
**125 GB RAM + 64 GB swap**, cuda:0 = 32 GB, cuda:1 = 16 GB. Config here:
`low_vram: true`, `qtype_te: qfloat8`, durable transformer-defer fix.

```
STAGE                                   RAM (host)        VRAM cuda:0
─────────────────────────────────────────────────────────────────────────
0. load transformer safetensors→CPU     +60 GB bf16        —
   load_state_dict(assign=True)         (dup aliased)
   ►► del transformer_state_dict+flush   −dup  (LEAK FIX)   —
─────────────────────────────────────────────────────────────────────────
1. Quantize Transformer  (per-block,    60→ ~30 (q8)        ~0
   low_vram⇒stays/returns to CPU         / ~15 (q4) GB,     (low_vram:
   PINNED RAM); freeze; "extras"         **pinned**)         on CPU)
─────────────────────────────────────────────────────────────────────────
2. load_te: Mistral from_pretrained     +~48 GB bf16        —
   (dtype=+low_cpu_mem_usage ►► FIX;     [pre-fix: +96 GB
    pre-fix loaded fp32)                  fp32 → host OOM]
   quantize(qfloat8) one-shot            +~24 GB transient
   freeze → resident                     ~24 GB; →te_device  (TE→cuda:0
                                                              if GPU)
─────────────────────────────────────────────────────────────────────────
3. load VAE                              +0.3 GB            small
─────────────────────────────────────────────────────────────────────────
4. Caching pass (MEMPROBE: xfmr=cpu,     cache HIT: ~0      ~1.6 GB
   te=cpu, durable defer holds)          cache MISS: Mistral fwd ⇒ historic
                                          OOM ~30 GB VRAM (torchao uintX) or
                                          ~95 min CPU (te_device:cpu)
   ►► UNLOAD TEXT ENCODER                 −Mistral (~24–48  −Mistral
      (cache→drop, framework-native)       GB freed)
─────────────────────────────────────────────────────────────────────────
5. prepare_accelerator:                  −pinned xfmr        +q8 ~31 GB
   accelerator.prepare(unet)              (CPU→GPU)           +q4 ~16 GB
   transformer CPU → cuda:0
─────────────────────────────────────────────────────────────────────────
6. TRAIN STEP (per iter)                 dataloader workers  resident
   fwd+bwd; ROCm has NO quanto/torchao   fork (+RAM)         + per-LAYER
   kernel ⇒ each Linear dequant→bf16,                        bf16 transient
   ×2 via gradient_checkpointing                             + activations
   recompute; +LoRA grads +adamw8bit                         (∝ resolution)
─────────────────────────────────────────────────────────────────────────
```

`►►` = a fix/kludge applied this work. Resident weights shrink with bit-width;
the **per-layer bf16 dequant transient + activations are bit-INVARIANT** — on
ROCm that ~16 GB non-weight step cost is the real wall, not the quant size.

### Two encounters (2026-05-17)

- **qint4 + [768,1024] + rank64** — full pipeline green: quant→cache→TE
  unload→**trained to step 5/3000, loss moving**. But **VRAM pinned
  31.99/32 GB**, OOM-retry churn, 54–131 s/it (~45–110 h). Crash locus:
  **stage 6**, GPU — 1024-bucket activations (~1 MP) on top of ~16 GB
  resident + bf16 transient. Lever = drop activations (res→[768]), not lower
  bits (transient is bit-invariant).
- **qfloat8 + [768] + rank32** — got further than the earlier qfloat8 run
  (which OOM'd at **stage 5**, GPU, 30.94 GB): reached `0/3000` then **host
  global OOM at stage 6** — `pt_data_worker` killed, system-wide (not the
  124 G cgroup cap). Cause: low_vram parks the ~30 GB qfloat8 transformer in
  **non-swappable pinned** CPU RAM; dataloader-worker fork at step 0 exceeds
  125 GB and the 64 GB swap can't absorb pinned pages. qfloat8 is dead on
  this box on **two** counts (GPU stage 5 *and* host stage 6).

### Earlier TE (Mistral) struggle — where it OOM'd

The whole prior saga lived at **stages 2 & 4**:
- **Stage 2, host RAM:** `transformers≥5` ignored `torch_dtype` → Mistral
  loaded **fp32 (~96 GB)**; quanto one-shot qfloat8 held orig+quant → **global
  OOM ~124–128 GB RAM** (killed twice). Fixed: `dtype=` +
  `low_cpu_mem_usage=True`. Compounded by the stage-0 `transformer_state_dict`
  leak (≈+30 GB held through stage 2).
- **Stage 4, VRAM:** Mistral forward for embed-caching via torchao uintX
  dequant-fallback peaked **~30 GB on cuda:0** and OOM'd at caching `0/25` —
  **qtype-invariant** (uint4≈uint3≈uint6, the bf16 whole-model dequant
  dominates). uint4-on-cuda:1 (16 GB) also overflowed. Resolved by
  `te_device: cpu` (exact, ~95 min) and then proven moot once **qfloat8 TE
  caching worked end-to-end** with the leak/bf16 fixes — TE is now a solved
  problem; the only open wall is the **transformer training step (stage 6)**.

## Logs

- **Engineering progress log (in-repo, committed):** [`progress.md`](progress.md)
- **Master PARA project log (full history):**
  `~/gdrive/para/01_projects/ai-toolkit for ROCm.md`
