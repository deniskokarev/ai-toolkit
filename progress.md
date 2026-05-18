# progress.md — ai-toolkit ROCm engineering log

Rolling, committed engineering log for the `rocm-integration` branch. Standing
context is in [`CLAUDE.md`](CLAUDE.md). The **master PARA project log** with the
full multi-day history is `~/gdrive/para/01_projects/ai-toolkit for ROCm.md` —
this file is the in-repo working subset, cross-linked from there.

---

## 2026-05-17 — Flux2 LoRA: text-encoder OOM during embedding caching

**Job:** `output/eve_angel_f2d` (FLUX.2-dev LoRA, 25-image dataset).
**Symptom:** crash before training starts; `torch.OutOfMemoryError` on **GPU 1
(15.98 GiB = RX 6900 XT 16 GB)**, "tried to allocate 640 MiB, 984 MiB free".

### Root cause (confirmed from `log.txt`)

It is the **text encoder**, not the training pipeline. Call chain:
`BaseSDTrainProcess.run():2074` → `get_dataloader_from_datasets` →
`AiToolkitDataset.__init__` → `setup_epoch()` → `cache_text_embeddings()`
(`dataloader_mixins.py:2105`) → `encode_prompt` → **Mistral-Small-3.1-24B**
forward → torchao `dequantize()` → OOM.

The FLUX.2 transformer loaded/quantized fine at `uint6` on cuda:0 (32 GB) and
never ran. Two compounding facts:

1. **`te_device: cuda:1` pins Mistral-24B to the 16 GB card.** At `uint4`
   the weights are ~12 GB; the box only has 16 GB there.
2. **No fused quantized-linear kernel on this ROCm/torchao build.** Log shows
   `QuantizedLinearNotImplementedError: No specialized dispatch found for
   quantized linear op` → every Linear falls back to *dequantize-whole-weight-
   to-bf16-then-matmul*, plus an `int32` copy temp in `_dequantize_affine`.
   So uint4 shrinks resident storage but pays it all back per layer during
   forward. 12 GB weights + per-layer dequant temps > 16 GB.

**Verdict:** lowering the *transformer* quant to "Q5"/`uint5` is irrelevant —
wrong model, wrong card. The only fact that matters: Mistral-24B does not fit
in 16 GB in forward on this build.

### Key finding: the staged pipeline the user asked for already exists

`SDTrainer.py:307-340`: when `is_caching_text_embeddings` (forced on by
`cache_text_embeddings: true`), the framework already does exactly
"cache then drop the TE":
- `cache_sample_prompts()` → fixed sample-prompt embeds cached
  (`sample_prompts_cache`), so **sampling no longer needs the live TE**;
- `unload_text_encoder(self.sd)` → Mistral **fully freed** (not CPU-shuttled);
  log message: *"Embeddings cached to disk. We dont need the text encoder
  anymore"*.

So no pipeline refactor is required for the staging itself. The OOM is that
the **dataset** embedding-caching pass (`dataloader_mixins.py`, runs at
`BaseSDTrainProcess:2074`) happens *before* that unload, with Mistral pinned
to the 16 GB card.

### Are text embeddings static across training + sampling? — Yes, here.

`dataloader_mixins.py:408` and `:423`: **caption dropout and token dropout
are explicitly disabled when `cache_text_embeddings` is on**
(`... and not self.dataset_config.cache_text_embeddings`). Caching ⇒ exactly
one fixed embedding per caption, reused every step. Sample prompts are cached
too. So in this config (LoRA, `train_text_encoder: false`, no
embedding/adapter/decorator, fixed captions/sample prompts) embeddings are
fully static — the user's intuition is correct and the live TE is dead weight
after the one caching pass.

Re-computation during training is genuinely needed only when conditioning
changes per step:
1. `train_text_encoder: true` (TE weights update — SDTrainer:311 refuses to
   unload in this case);
2. learnable embeddings / textual inversion;
3. **active caption/token dropout** — caching *disables* it. ⚠️ This config
   has `caption_dropout_rate: 0.05`, silently turned off by
   `cache_text_embeddings: true`. Minor (5% unconditional/CFG regularizer on a
   25-image LoRA) but a real, silent behavior change worth knowing.
4. per-step dynamic prompts (caption shuffle, random multi-caption pick,
   prompt-injection adapters, decorators, diff-output-preservation class).

### Recommended fix (config-only, no refactor, current 32+16 box)

Run the embedding-caching pass on the **32 GB** card with the transformer
*not* co-resident:

- `model.te_device: cuda:0` (or null → defaults to main device), **and**
- `model.low_vram: true` — parks the transformer on CPU during load
  (`flux2_model.py:181/255`), so during caching cuda:0 holds only Mistral
  (~12 GB / 32 GB, comfortable). After caching, `unload_text_encoder` frees
  Mistral; the transformer moves to GPU for training/sampling.

This realizes the user's staging idea using only existing code paths. Without
`low_vram`, transformer (uint6 ≈ 18 GB) + Mistral (uint4 ≈ 12 GB) + dequant
temps risks OOM on the 32 GB card too.

Cheap extra: `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (HIP build also
honors the `PYTORCH_CUDA_ALLOC_CONF` alias) — marginal here; the shortfall is
too-big, not just fragmentation.

### Optional later optimization (real, non-essential refactor)

Make `flux2_model.py:load_model()` skip co-loading the transformer to GPU when
text-embed caching is pending, so `low_vram` isn't needed as a blunt
workaround and the transformer load isn't paid until after the TE is freed.
This is an optimization, not a correctness fix — the config-only path above
already works.

### Update — `low_vram` config fix FAILED (2nd OOM, now on cuda:0)

`te_device: cuda:0` + `low_vram: true` + `qtype_te: uint6` → OOM on **GPU 0
(31.86 GiB, 29.40 GiB allocated)**, again at caching `0/25`.

**Why it failed:** `low_vram` parks the transformer on CPU at load time
(`flux2_model.py`, log "Moving transformer to CPU") **but**
`base_model.set_device_state()` (`:1445`) does an unconditional
`self.unet.to(state['unet']['device'])` — it is NOT low_vram-aware. At
`BaseSDTrainProcess.py:1971`, *before* the caching pass (`:2074`), the
optimizer-params preset is applied with `train_unet=True`, which moves the
transformer **back onto cuda:0**. Caching then runs with **both uint6 models
co-resident**: transformer (~18 GB) + Mistral uint6 (~18 GB) ≈ 29.4 GB → OOM.

Net: any single-card plan fails — both models are forced GPU-resident during
the dataset caching pass, and 2×uint6 ≈ 30 GB overflows the 32 GB card with
the dequant-fallback transient. The framework's cache→unload-TE staging
(`SDTrainer:307-340`) runs *after* this, too late.

### Recommended fix — two-card split + TE layer-offload (config-only)

Transformer alone on cuda:0; Mistral streamed on cuda:1 so it fits 16 GB:

```yaml
model:
  te_device: cuda:1
  qtype_te: uint4            # ~12-14 GB; uint6 (~18 GB) can't fit 16 GB
  low_vram: false            # transformer lives on cuda:0 (32 GB) fine
  layer_offloading: true
  layer_offloading_transformer_percent: 0     # keep transformer resident (fast training)
  layer_offloading_text_encoder_percent: 0.5  # stream ~half of Mistral -> fits 16 GB
```
`offload_percent` = P(layer offloaded to CPU+streamed); 0 = none, 1 = all
(`manager.py:100-105`). uint4-on-cuda:1 alone was only 640 MB over 16 GB, so
0.5 gives a wide margin for the one-time caching pass.

### Durable code fix (proper solution, optional)

Defer the transformer's move to GPU until *after* text-embed caching + TE
unload — i.e., gate the unet device in the params preset
(`BaseSDTrainProcess:225/1971`) on `is_caching_text_embeddings`, or reorder.
Then caching runs with only Mistral resident; transformer (uint6, full
quality) loads after on a free card. Single-card, no quality loss, no
offload slowdown. Behavior-changing core edit — needs testing that LoRA
param enumeration doesn't require the base transformer on GPU.

### Two-card config launched — outcome UNCONFIRMED (session handoff)

Created `output/eve_angel_f2d/config_2card.yaml` (copy of the job config with
the two-card split applied):

```yaml
model:
  te_device: cuda:1
  qtype_te: uint4
  low_vram: false
  layer_offloading: true
  layer_offloading_transformer_percent: 0      # transformer fully resident on cuda:0 (fast)
  layer_offloading_text_encoder_percent: 0.5   # ~50% of Mistral streamed from CPU -> fits 16 GB
```

Launched directly (not via UI), bypassing the db/queue:

```bash
cd /home/dkv/ai-toolkit
AMD_SERIALIZE_KERNEL=0 TORCH_USE_HIP_DSA=0 venv/bin/python run.py output/eve_angel_f2d/config_2card.yaml
```

**RESULT: the background run.py task FAILED with exit code 1** (reason not yet
inspected — could be OOM at the caching step despite offload, a ROCm/HIP
incompatibility in the `MemoryManager` layer-offload streaming path, or a
config/validation error). Last positive observation before failure was the
transformer quantizing (4/56 blocks).

**Output to inspect (EPHEMERAL — may be gone after restart):**
`/tmp/claude-1000/-home-dkv-ai-toolkit/1edd4f30-aaf8-4c26-a5db-f9df742f444e/tasks/b7m1unv18.output`
Grep it (if it still exists) for: `OutOfMemory|Traceback|MemoryManager|attach|
Caching text|UNLOADING TEXT|RuntimeError|HIP`. The db row +
`output/eve_angel_f2d/log.txt` show a *stale* cuda:0 OOM from the prior
(`low_vram`) run — ignore those; the direct run.py output did NOT go there.

**Because that file is session-scoped, relaunch next session with a STABLE
log** so the result survives:

```bash
cd /home/dkv/ai-toolkit
AMD_SERIALIZE_KERNEL=0 TORCH_USE_HIP_DSA=0 venv/bin/python run.py \
  output/eve_angel_f2d/config_2card.yaml --log /tmp/eve_2card.log
```
(`run.py` supports `--log <path>`; tail `/tmp/eve_2card.log`.)

> ⚠️ The run.py process was started via the harness background runner and is
> likely killed by the session restart. **Next session: assume it must be
> relaunched.**

#### Next-session checklist

1. `pgrep -af "run.py .*config_2card"` — is it somehow still alive? Check
   `rocm-smi --showmeminfo vram` for residual GPU use.
2. If not running, relaunch with the command above (background it via the
   harness runner; no `nohup`/`&` needed).
3. Watch for the decisive markers: `Caching text embeddings to disk: 100%` /
   `UNLOADING TEXT ENCODER` / `Embeddings cached to disk` = **fix worked**;
   `OutOfMemoryError` = still broken (then go to the durable code fix).
4. GPU enum reminder: `cuda:0` = R9700 32 GB, `cuda:1` = RX 6900 XT 16 GB
   (cuda:1 has ~1.5 GB baseline display usage → ~15.7 GB usable).

### Permissions changed this session (`.claude/settings.local.json`, gitignored)

- Added run.py launch rules incl. env-prefixed / `nohup` forms; `Bash(python *)`,
  `Bash(sed *)`, `Bash(awk *)`; `Edit`/`Write` under `/home/dkv/ai-toolkit`.
- Set `permissions.defaultMode: bypassPermissions` — **takes effect on next
  session start** (full no-prompt autonomy for this project).

### 2026-05-17 (cont.) — durable fix implemented & PROVEN; real wall is Mistral-24B uint4 *forward*

**Ephemeral log recovered.** The two-card `config_2card.yaml` failure was a clean
OOM on cuda:1 at caching `0/25`, traceback straight through stock `transformers`
(`MistralModel→DecoderLayer→MLP→nn.Linear→torchao.dequantize`) with **no
MemoryManager/offload hook in the stack** and zero offload log lines.
`layer_offloading_text_encoder_percent` wires to
`MemoryManager.attach(self.network, …)` (`BaseSDTrainProcess:1894`) — it wraps
the **LoRA network**, never the standalone FLUX.2 Mistral TE. Config-only
two-card path is **dead**, and Mistral-uint4 (~15.8 GB w/ dequant transient)
overflows the 16 GB cuda:1 even with the transformer absent → unfixable there.

**Durable code fix implemented and verified working:**
- `BaseSDTrainProcess.py` (~:1761) — gate the premature
  `unet.to(self.device_torch)` on `is_caching_text_embeddings`; keep the
  transformer on CPU until `set_device_state(train_device_state_preset)`
  (~:2137, runs *after* TE unload) brings it back. Symmetric with the
  framework's own intent at `SDTrainer.hook_before_train_loop:245`.
- Pairs with `low_vram: true`, which is the *intended* ROCm load-time park
  (`quantize.py:311` returns each quantized block to CPU; with `low_vram:false`
  it deliberately keeps them resident — `:310` "keep on GPU for ROCm" — so they
  collide with Mistral in `load_te` → OOM at load). The documented `low_vram`
  failure was caused **solely** by `:1744` being unconditional pre-fix.
- New single-card config: `output/eve_angel_f2d/config_durable.yaml`
  (`te_device: cuda:0`, `low_vram: true`, `layer_offloading: false`).

**Hard proof (temp MEMPROBE before the caching pass):**
```
[MEMPROBE pre-caching] transformer.device=cpu text_encoder.device=cpu cuda:0 alloc=1.61GB
```
Transformer **is on CPU**, cuda:0 **empty (1.6 GB)** before caching — the
deferral works exactly as designed. The caching pass *alone* then drives cuda:0
1.6 GB → **29.95 GB** and OOMs (tried +640 MiB; 0.93 GB reserved-unalloc).

**Verdict (H2, not H1):** the transformer co-residency was never the sole
problem on 32 GB. **Mistral-Small-3.1-24B at uint4, alone, in one forward via
the torchao no-fused-kernel dequant fallback, peaks ~30 GB** — it does not fit
a 32 GB card even with nothing else resident. No single-card config solves this
at uint4. progress.md's earlier root-cause (lines 28–34) was right; the new
hard fact is the magnitude (≈30 GB for one Mistral forward).

**Lever results — the quant ladder is DEAD (qtype-invariant peak):**
| qtype_te | alloc env | OOM @ allocated |
|---|---|---|
| uint4 | — | 29.95 GB |
| uint4 | expandable_segments | 29.90 GB (frag 933→157 MiB; real-alloc bound) |
| uint3 | expandable_segments | 29.98 GB |

uint4 vs uint3 → **same ~30 GB peak**. The torchao dequant fallback
materializes a full **bf16** copy of every Linear weight, so the forward peak
is **qtype-INVARIANT**; the compact stored form is a rounding error in the
peak. `expandable_segments` only reclaimed ~0.8 GB stranding (real bound, not
fragmentation). **No GPU-resident quant config fits Mistral-24B's forward on
32 GB on this build** — uint2 would OOM at ~30 GB too. FLUX.2 needs the
hidden states of decoder layers **[10,20,30]** stacked
(`flux2/src/pipeline.py:38`), not a pooled embedding.

**Decisive pivot — `te_device: cpu` (config-only, exact fidelity):**
The caching pass is one-time (~25 captions + ~10 sample prompts), then the TE
is unloaded and GPU training runs full speed. Running the *reference* Mistral
on CPU yields byte-faithful layer-[10,20,30] states at **zero VRAM**; box has
125 GB RAM (62 free). Fully wired (`base_model.py:121` → `torch.device('cpu')`;
`flux2_model.py:466-467`). **In flight now.** Strictly dominates a
llama.cpp/GGUF swap, which can't emit arbitrary multi-intermediate-layer
hidden states without C++ patching + numeric-validation risk vs the frozen
coupled FLUX.2.
Remaining heavier fallback if CPU caching is unviable: make `MemoryManager`
wrap the standalone Mistral (GPU layer-streaming), not just the LoRA net.

### Status

- [x] Root-caused (TE, not pipeline; no fused ROCm uintX kernel).
- [x] Confirmed framework stages cache→unload TE (but too late vs caching).
- [x] `low_vram` config fix tried → FAILED (cause: `:1744` unconditional).
- [x] Two-card split + TE layer-offload → **dead** (offload wraps LoRA net,
  not Mistral; Mistral-uint4 forward overflows 16 GB regardless).
- [x] **Durable code fix implemented (`:1761` gate) and PROVEN working**
  (MEMPROBE: transformer on CPU, cuda:0 empty pre-caching).
- [x] Disambiguated H1 vs H2 → **H2**: Mistral-24B uint4 forward alone ≈30 GB
  (no single-card fix at uint4).
- [x] `expandable_segments:True` (uint4) → still OOM (real-alloc bound).
- [x] `qtype_te: uint3` → still OOM ~30 GB ⇒ **peak is qtype-invariant;
  quant ladder dead**.
- [x] `te_device: cpu` (uint4) — **caching SOLVED end-to-end** (25 captions +
  sample prompts + TE unload, ~95 min CPU, first green caching). BUT training
  then **OOM'd at `set_device_state` moving the uint6 transformer CPU→cuda:0**:
  `tried 54 MiB; 4 MiB free; only 5.43 GiB allocated by PyTorch` — i.e. ~26 GB
  held by the process but **invisible to torch's allocator** = the stale
  `expandable_segments` mapping from the transformer-quant phase that ROCm
  never decommitted. **User's "not harmless" call confirmed: the ghost
  reservation is directly fatal at the next phase's model move.** (Proven: GPU0
  dropped 30 GB→0.5 GB the instant the process exited.) `empty_cache()` at the
  pre-caching site does NOT cover this point (`:2137` is after `:2094`).
- [x] **qfloat8 run — reproducibly OOM-killed the HOST at "Quantizing
  Mistral"** (NOT GPU, NOT Chrome, NOT a config error). dmesg/journalctl:
  `Out of memory: Killed process (python) anon-rss ~124-127 GB` on a 125 GB
  box, twice; Chrome was collateral (anon-rss 98 MB). Root cause: `transformers
  >=5` deprecated/ignores `torch_dtype`, so `flux2_model.load_te()` loaded
  Mistral-24B in **fp32 (~96 GB host)** instead of bf16; quanto's **one-shot
  whole-model** qfloat8 quant then held original+quantized copies → >124 GB →
  global OOM (~30 s freeze). uint4/torchao did NOT hit this (lower host peak).
- [x] **FIX APPLIED:** `flux2_model.load_te()` `torch_dtype=`→`dtype=` +
  `low_cpu_mem_usage=True` (stream shards at bf16, ~48 GB peak, no fp32 copy).
  Latent correctness bug regardless; should make qfloat8 viable and de-risk
  every path. (commit pending)
- [ ] Re-run: either qfloat8 (now host-OOM root cause fixed) or the proven
  `te_device: cpu`/uint4 path (no `expandable_segments`) with watch-and-
  abort-fast on the embed-cache phase. NOTE: model load+quantize is
  unconditional & upstream of the embed-cache check, but uint4 Mistral quant
  is proven safe and the May-17 `_t_e_cache` set (15.7 MB files) is on disk →
  cache-hit expected (verify live, kill within ~2 min if recomputing).
- [ ] If still OOM at `set_device_state`: add `empty_cache()` at that
  transition (code), and/or revisit `expandable_segments` alloc-conf tuning.
- [ ] If CPU caching path retained: MemoryManager-wraps-Mistral (GPU stream).
- [ ] Remove temp MEMPROBE/empty_cache diag in `BaseSDTrainProcess.py` once
  resolved (currently load-bearing for measurement).
- [ ] Decide whether to accept losing `caption_dropout_rate: 0.05`.

**Disconnect (rocm-smi vs torch.reserved) — verdict:** the ~26–30 GB the
driver shows while `memory_reserved()`≈1.5 GB is freed-not-returned physical
backing under `expandable_segments` on ROCm (the HIP unmap/decommit path not
firing on idle/`empty_cache`). Partly expected expandable_segments accounting
semantics (true on CUDA too); the *non-return on ROCm* is the bug-like, and now
proven *fatal*, part. Definitive control test = the in-flight no-
expandable_segments run; if the ghost vanishes there, root cause confirmed.
- [ ] (Future) Remote TE offload to Mac Studio — **planned, not started**;
  see Mac section below.

### CORRECTION (2026-05-17 cont.) — "quant ladder dead" was **torchao-only**

A working reference log from a large-GPU run uses `qtype_te: qfloat8` and
caches 72 text embeds in 55 s. `quantize.py:get_qtype` (:56): `uint2..uint8`
**and `float8`** are in `torchao_qtypes` → the torchao dequant-fallback path
(`_dequantize_affine`, the `zero_point.to(torch.int32)` full-weight temp) that
OOM'd every attempt. **`qfloat8` is NOT in `torchao_qtypes`** → routes to
`optimum.quanto` (`qtypes[...]`) — a **different backend, never tested on
ROCm**. quanto float8 is scale-only (no affine int zero-point ⇒ **no int32
temp**, a large slice of the ~30 GB) with its own linear path. So the
qtype-invariance / "ladder dead" verdict holds **only for the torchao uintX
family**; an entire **quanto ladder (`qfloat8`, `qint4`, `qint2`)** is
untested and has a materially different memory profile.

Caveat (honest): qfloat8 ≈ 1 B/param ≈ ~24 GB resident for Mistral-24B; if
quanto also dequantizes per-layer on RDNA (no fp8 ALU on gfx1030/gfx1201) the
bf16 transient could still be tight on 32 GB. Unproven on 32 GB — the
reference machine is a big GPU. But it is the **single highest-value untested
lever** and a free config change. May obsolete BOTH the slow `te_device: cpu`
path and the Mac-offload plan.

Also confirmed from the reference: latents caching is **VAE-only**, fully
independent of the TE; multi-resolution recomputes latents per resolution but
text embeds are resolution-independent (computed once, then cache-hit).

**RDNA4 native fp8.** cuda:0 = Radeon R9700 (gfx1201, **RDNA4**) natively
supports E4M3/E5M2 fp8. So quanto qfloat8 may dispatch a *real* fp8 matmul on
cuda:0 (not a dequant fallback) — materially raising the odds. Caveat: silicon
support ≠ the ROCm torch/quanto stack actually emitting fp8 kernels for
gfx1201; only a run settles it. (cuda:1 = RX 6900 XT gfx1030 RDNA2 = no fp8,
but the single-card plan is all on cuda:0 anyway.)

**STAGED:** `output/eve_angel_f2d/config_qfloat8.yaml` — `qtype` &
`qtype_te` both `qfloat8` (matches the known-good NVIDIA reference),
`te_device: cuda:0`, `low_vram: true`, durable fix applies. Dormant; does not
affect the running job. Fallback noted in-file: if the qfloat8 *transformer*
path misbehaves on ROCm, drop `qtype: uint6` (proven) and keep
`qtype_te: qfloat8`.

**NEXT ACTION (user-directed):** when the in-flight `te_device: cpu` run ends
— **success OR failure, regardless** — launch `config_qfloat8.yaml` (single
job; no concurrency — GPU contention). The cpu run still independently yields
first green training + on-disk cache if it succeeds. If qfloat8 OOMs slightly,
try quanto `qint4`.

### 2026-05-17 (cont.) — qfloat8 host-OOM ROOT-CAUSED: dangling `transformer_state_dict` (NOT a missing gc)

qfloat8 run reproduced the host OOM at `Quantizing Mistral`, **twice** (kernel
log): 16:07:54 cgroup `MemoryMax` kill, anon-rss **120.1 GB**; 16:14:46
unscoped **global** OOM, anon-rss **128.3 GB**. User observed RSS reaching
~85 GB right after transformer quant and **never dropping**, then climbing to
121-122 GB during Mistral quant before the kill.

**Root cause (precise).** `flush()` already calls `gc.collect()` after
transformer quant (`flux2_model.py:172`) and in `load_te()` (`:116/:119`) — gc
is **not** missing. `load_model()` does
`transformer.load_state_dict(transformer_state_dict, assign=True)`
(`:163`): `assign=True` **aliases** the module params onto the dict's tensors,
and `transformer_state_dict` (a never-deleted local) then pins the **entire
unquantized bf16 transformer** alive through `quantize_model()` AND `load_te()`
(Mistral load + one-shot qfloat8 quant). `freeze()` swaps the module's weights
to fp8 but the dict still holds the originals → original bf16 transformer +
fp8 transformer + Mistral bf16 + quanto Mistral-quant transient co-resident →
~120-128 GB → kill. The post-quant gc collected nothing because the reference
was live.

**Durable fix applied** (`flux2_model.py:~165`): `del transformer_state_dict`
+ `flush()` immediately after `load_state_dict(assign=True)`. The existing
`:172` flush now actually reclaims the float weights before Mistral loads.
Commit pending. Unproven until re-run; honest residual risk = the Mistral
quant in `load_te():114` is still **whole-model one-shot** (holds bf16 +
quantized Mistral transiently) — if still tight, real cure is per-block
Mistral quant staging in `load_te()`; interim cushion = 64 GB swap **with
`MemorySwapMax` set** (user's prior scope used `MemorySwapMax=0`, which
disabled cgroup swap entirely — system swap size was irrelevant under that
command).

**Action items added:**
- [ ] Re-run qfloat8 with the `del transformer_state_dict` fix; confirm RSS
  drops after transformer quant (expect baseline → ~fp8-transformer-only
  before `Loading Mistral`).
- [ ] If still OOM at `Quantizing Mistral`: implement per-block Mistral quant
  staging in `load_te()` (mirror the transformer block loop in `quantize.py`).
- [ ] Bump `/swap.img` 8 GB → 64 GB (ext4, ~244 GB free; fstab already has
  the swap line) AND use `MemorySwapMax=64G` (not `0`) in the run scope.

### 2026-05-17 (cont.) — leak fix PROVEN; TE/caching SOLVED; new+final wall = qfloat8 transformer too big

Re-ran `config_qfloat8.yaml` under `systemd-run --user --scope -p
MemoryMax=124G -p MemorySwapMax=infinity` (swap already bumped to 64 GB).

- **`del transformer_state_dict` fix PROVEN.** No host OOM. Sailed through
  `Quantizing Transformer → Moving transformer to CPU → Loading Mistral →
  Quantizing Mistral → Loading VAE → Preparing Model`, swap untouched. The
  dangling-alias diagnosis was correct.
- **TE/caching SOLVED.** MEMPROBE pre-caching: `transformer.device=cpu
  text_encoder.device=cpu cuda:0 alloc=1.61GB` (durable defer still holds).
  Both dataset passes: latents 25/25 + **text embeddings 25/25** at 1790 then
  16747 it/s = disk **cache-HIT** on the May-17 `_t_e_cache` files. The exact
  step that OOM'd on every prior attempt now passes cleanly.
- **NEW wall (different model, post-caching):** `Error running job: CUDA OOM
  ... GPU 0 30.94 GiB / 31.86 GiB, 228 MiB free`. Traceback:
  `BaseSDTrainProcess.py:2129 hook_before_train_loop → DiffusionTrainer:304 →
  SDTrainer:242 → BaseSDTrainProcess:716 prepare_accelerator → :728
  accelerator.prepare(self.sd.unet)` → accelerate `prepare_model →
  model.to(device)` → quanto QTensor `_to_copy`. Mistral on CPU (cache-hit, not
  co-resident — proven by 1.61 GB pre-caching probe). **Verdict: the qfloat8
  FLUX.2-dev transformer ALONE ≈ 31 GB and does not fit a 32 GB card for
  training** — it OOMs just being placed on cuda:0, before any
  activation/optimizer memory. quanto `_to_copy` preserves fp8 `_data` dtype
  (no upcast) → 30.94 GB is the genuine resident fp8 transformer (~30B params).
  This is NOT a TE problem; the TE saga is closed.
- **Fallback applied (the one config_qfloat8.yaml itself prescribes):**
  `qtype: uint6` (~0.75 B/param, ~23 GB resident, torchao — proven to fit
  cuda:0 resident) + `qtype_te: qfloat8` (TE caching now proven). Relaunched
  (scope `eve-u6te8`). Honest risk: torchao uint6 has no fused ROCm kernel →
  per-layer dequant-to-bf16 in the *training* forward/backward (slow + a
  per-layer, NOT whole-model, transient); resident drops 31→~23 GB leaving
  ~9 GB for the largest-layer bf16 transient + activations + LoRA. Plausible
  fit, unproven. If uint6 still OOMs at prepare/training: `qtype: uint4`
  (~15 GB resident) next; deeper option = transformer layer-offload (it IS
  wired for the transformer at `flux2_model.py:177-185`, unlike the TE) but
  `accelerator.prepare`'s `model.to(device)` at `:728` will fight it (same
  bug-class as the `:1761` defer) → would need a `:728` gate too.

**Status delta:** host-RAM OOM = FIXED (leak). qfloat8 TE caching = WORKS
end-to-end. Open question is now purely **transformer training fit on one
32 GB card**, fully decoupled from the (solved) TE/caching problem.

### 2026-05-17 (cont.) — TE/caching CLOSED (TE unloaded, training started); transformer quant ladder

uint6 run: `***** UNLOADING TEXT ENCODER *****` / "Embeddings cached to disk.
We dont need the text encoder anymore" — the framework's cache→unload-TE
staging fired exactly as designed. Training **started** (`eve_angel_f2d:
0/3000`, reached step 2) then `OOM during training step 3 times in a row,
aborting` (`BaseSDTrainProcess.py:2310`; the handler swallows the raw CUDA OOM
text — no MiB figure). **The TE saga is fully closed.** Remaining problem is
solely the quantized transformer's *training step* fitting one 32 GB card.

Commit `d421e9e` landed the two proven `flux2_model.py` TE fixes (bf16
load_te + `del transformer_state_dict`). `BaseSDTrainProcess.py` (durable
defer + temp MEMPROBE diag) intentionally NOT committed yet — strip the diag
after a green run, then commit.

**Transformer quant ladder, one 32 GB card (`get_qtype` routing confirmed
`quantize.py:56`: `uint2..uint8`+`float8`→torchao; `q*`→optimum.quanto):**
| qtype | backend | resident | outcome |
|---|---|---|---|
| qfloat8 | quanto | ~31 GB | OOM at `accelerator.prepare()` *placing* it on cuda:0 — **resident-bound**, no step |
| uint6 | torchao | ~23 GB | fit+trained 2 steps, then OOM on the step — torchao no-fused-kernel dequant→bf16 (+int32 temp) per Linear, **×2 via gradient_checkpointing recompute**. Same mechanism as Mistral. **transient-bound** |
| qint4 | quanto | ~15 GB | blocked by quanto/ROCm bug → shimmed → **IN FLIGHT** (scope `eve-qi4c`) |

Key disambiguation (user-raised): the "lower bits cost more" effect is a
**torchao-family** artifact (stored size ≪ invariant bf16 transient). The
transformer is quantized **per-block** (not Mistral's whole-model one-shot),
so the torchao bf16 transient is one-Linear-sized and lowering bits *does*
cut resident — but uint6 still pays the bf16+int32 transient ×2 (grad-ckpt).
`qint4` is **quanto** (≈0.5 B/param ≈ half qfloat8's resident, NO torchao
int32 temp) → ~17 GB headroom for activations+transient vs uint6's ~9 GB. It
is the single informative untested point. Caveat: gfx1201 has native fp8 but
**no native int4** → qint4 still dequant→bf16 per Linear (×2 grad-ckpt), just
without torchao's int32 temp and with far more resident slack. If qint4 still
OOMs on the step → attack activations: `resolution: [768]` (drop the 1024
bucket) and/or transformer layer-offload (wired at `flux2_model.py:177-185`;
risk: `accelerator.prepare`'s `.to(device)` at `:728` may fight it).

### 2026-05-17 (cont.) — quanto sub-byte path is ROCm-broken; shimmed (new kludge)

First qint4 launch crashed in seconds — **not OOM**:
`optimum/quanto/tensor/qbits/qbits.py:128`
`version.parse(torch.version.cuda).release >= (12, 1)` → on ROCm
`torch.version.cuda is None` → `version.parse(None)` → `TypeError: 'NoneType'
object is not iterable`, *before* the comparison can short-circuit False. This
gates a CUDA-only TinyGemm kernel; every quanto **sub-byte** type
(`qint4`/`qint2`) is unusable on this HIP torch. `qfloat8` dodged it (8-bit =
`qbytes` path, never enters `qbits`).

**Kludge added — `toolkit/util/quantize.py` (import-time, HIP-gated):**
replace the `version` global in `optimum.quanto.tensor.qbits.qbits` with a
None-safe proxy so the guard evaluates False → quanto falls back to its
portable `QBitsTensor` (TinyGemm/AWQ are CUDA-arch kernels unusable on RDNA
anyway). Gotcha hit en route: `from optimum.quanto.tensor.qbits import qbits`
is ambiguous (`qbits/` package also has `tinygemm/qbits.py`) and patched the
wrong object under the full import graph — fixed by
`importlib.import_module("optimum.quanto.tensor.qbits.qbits")` (exact
submodule, import-order-independent). Verified through the real toolkit
`quantize()`: qint4 quantize **and forward** OK on cuda:0, rel-err vs bf16
≈ 0.096 (normal for int4). Secondary non-fatal: quanto's optimized HIP
extension fails to JIT (`hipcc` rejects NVCC-only flags
`--expt-relaxed-constexpr`/`--expt-extended-lambda`/`--use_fast_math`/
`--threads=8`) → quanto auto-falls-back to the portable pure-torch int4 path
(correct, just slower; one noisy build-fail dump per process — harmless).

Net new kludge count +1 (quanto qbits None-safe `version` shim). Belongs in
CLAUDE.md's kludge list + a commit once the qint4 run is green. Relaunched
clean/harness-tracked (scope `eve-qi4c`); resident ~15 GB target, the OOM
thesis (vs uint6 ~23 GB, no torchao int32 temp) is what this run tests.

### 2026-05-17 (cont.) — qint4 TRAINS end-to-end but is activation-bound; VRAM math nailed; → resolution [768]

Shim needed a 2nd fix (`data.device.type=="cpu"` disjunct routes qint4 to
TinyGemm→CUDA-only `aten::_convert_weight_to_int4pack`): replaced the narrow
`version` shim with a full `QBitsTensor.create` override that always returns
the portable `QBitsTensor` (AWQ/TinyGemm HIP builds fail anyway — `hipcc`
rejects NVCC-only flags). Verified CPU- and cuda-quant both OK.

**qint4 ran the WHOLE pipeline green:** transformer qint4 quant (56 blocks) →
Mistral qfloat8 → caching cache-hit → `UNLOADING TEXT ENCODER` → **training,
reached step 5/3000, loss moving** (0.55→0.49→0.59→0.76), never hit "skip
2/3" (every OOM recovered by retry → no abort). **The full pipeline is
proven correct.** BUT pathological: card0 VRAM pinned **31.99/≈32 GB**,
54–131 s/it (→ ~45–110 h), constant OOM-skip churn.

**VRAM math (real on-disk sizes):** `flux2-dev.safetensors` = **60 GB bf16
≈ ~30 B params** (Mistral TE 45 GB). Transformer weights resident: bf16 60 /
qfloat8 ~30 (matches the 30.94 GB `prepare` OOM) / **qint4 ~15–16 GB**
(user's ¼ estimate correct). The other ~16 GB of a *step* is **bit-invariant**:
the ROCm portable path **dequantizes each int4 layer to a full bf16 weight per
matmul, ×2 via gradient_checkpointing recompute** (no working quanto HIP
kernel) + 1024-bucket activations + LoRA grads/adamw8bit + HIP allocator
overhead. So ~16 GB weights + ~16 GB invariant ≈ 32 GB → card pinned.
Cross-check: user's NVIDIA ref (low_vram=false, both qfloat8, 48 GB) = qfloat8
transformer ~30 GB + ~18 GB step overhead **with a working fp8 kernel (no
dequant transient)**; the AMD 32 GB box structurally cannot host that point —
qint4 is the only thing that fits at all, with zero margin + the ROCm dequant
tax.

**Verdict: the wall is bit-invariant (dequant transient + activations), NOT
quant resident.** ⇒ qint2 / lower bits CANNOT help (same bf16 transient,
worse quality — dead). Only effective levers: cut **activations**
(resolution) or offload resident. Stopped the thrash run; dropped the 1024
bucket → `resolution: [768]` (keep qint4), relaunched scope `eve-q4r768`.
Expectation: VRAM off the ceiling, OOM-churn gone, steps several-fold faster
(usable-slow, not thrash-slow). Fundamental speed fix would require a working
ROCm quanto/torchao low-bit kernel — out of scope; even green it stays slow.

- [ ] Confirm [768] gives margin + tolerable s/it; if still tight →
  transformer layer-offload (next, last lever).
- [ ] On green: strip temp MEMPROBE diag, add quanto-qbits shim to CLAUDE.md
  kludge list, commit (`quantize.py` shim + `BaseSDTrainProcess.py` defer).

### 2026-05-17 (cont.) — qfloat8 retry (rank32/res768) fails on HOST pinned RAM, not GPU (prediction corrected)

Predicted: GPU OOM at `accelerator.prepare` (~30.94 GB), as in the first
qfloat8 run. **Wrong.** This run got *further* — past `UNLOADING TEXT
ENCODER` into `eve_angel_f2d: 0/3000` (training start) — then **host global
OOM**: kernel killed `pt_data_worker` pid 14995, `constraint=CONSTRAINT_NONE`
(system-wide, NOT the 124 G cgroup cap), 18:05:31. No GPU OOM / no
`accelerator.prepare` error in output.

**Root cause:** `low_vram: true` parks the ~30 GB qfloat8 transformer in CPU
**pinned** RAM. Pinned pages are **non-swappable** → the 64 GB swap cannot
help. At training start the dataloader workers spawn; physical 125 G is
exceeded by [~30 G pinned xfmr + workers + Python] and swap can't absorb the
pinned portion → global OOM at step 0. qint4 dodged this (CPU-parked weights
~15 G). So qfloat8 is non-viable on **two independent** counts here: GPU
(can't fit 32 GB resident for the step) AND host (low_vram pinned footprint +
workers exceed non-swappable RAM). It reached step 0; qint4 reached step 5 —
qfloat8 strictly worse on this box.

**Implication:** GPU-offload (cold blocks → cuda:1 VRAM, not CPU pinned RAM)
would sidestep BOTH walls. Decision pending: (a) config-only
qint4+res768+rank32 (cheap, proven direction, untested combo, likely usable);
(b) code route A — make transformer `MemoryManager` offload target
configurable cpu→cuda:1 (`flux2_model.py:177-185` hook exists; ROCm-unproven);
(c) route B — true pipeline split blocks across cuda:0+cuda:1 (~48 GB = NV
ref, qfloat8 quality, heavier).

### 2026-05-17 (cont.) — qfloat8 stage-6 host OOM is a SECOND leak, not a wall — fixed

User insight (correct): after the transformer moves CPU→cuda:0 it should not
stay in RAM. Code confirms the gap: `prepare_accelerator()` does
`self.sd.unet = accelerator.prepare(self.sd.unet)` (`BaseSDTrainProcess:728`)
with **no flush/gc/empty_cache before the train loop**, and `num_workers`
defaults to **2** (`config_modules.py:983`) → DataLoader **forks 2
`pt_data_worker`** at train start. The ~30 GB qfloat8 transformer CPU home
copy is *pinned* (non-swappable) and ROCm `hipHostFree` is deferred → it
lingers, the fork snapshots it → host global OOM at step 0. Structurally the
same class as the stage-0 `transformer_state_dict` leak, at stage 5→6.
Crucially run 2 had already cleared the **stage-5 GPU** hurdle → qfloat8 is
plausibly salvageable; this is a fixable leak/fork artifact, NOT a hard wall.

**Fixes applied (user-approved 1+2):**
1. `BaseSDTrainProcess.py:~731` — `flush()` immediately after
   `accelerator.prepare(self.sd.unet)` (reclaim the dead pinned CPU copy
   before workers fork; symmetric to the stage-0 fix).
2. `config_qfloat8.yaml` dataset — `num_workers: 0` (no forked
   `pt_data_worker`; 25 imgs, latents+embeds on disk → ~free).

Relaunched qfloat8 rank32/res768 (scope `eve-q8fix`, harness-tracked). If it
clears step 0 and trains, qfloat8 (best quality) is viable on this box and
the GPU-offload code routes become unnecessary. Fallback unchanged:
qint4+res768+rank32.

### 2026-05-17 (cont.) — host-leak fix VALIDATED; qfloat8 is the FAST path; only GPU-resident wall left

(Intermediate: `num_workers:0` first hit a latent bug — `data_loader.py:695`
passed `prefetch_factor` unconditionally; PyTorch forbids it when
`num_workers==0`. Fixed: guard with `if num_workers > 0`. Real config bug,
not ROCm-specific.)

Post-fix qfloat8 run: **NO host OOM, NO `pt_data_worker` kill** — the
stage-5 `flush()` + `num_workers:0` killed the host wall. User's second-leak
diagnosis confirmed. It trained past step 0 to **step 2/3000 at 2.29 s/it,
ETA ~1:54** — vs qint4's 54–131 s/it (~45–110 h). **qfloat8 ≈ 25–50× faster
per step** (gfx1201 fp8-ish path; quanto qbytes ≪ qint4 portable dequant). A
full run ≈ **~2 h**. Then OOM on the training step (GPU), 3×→abort at step 2:
qfloat8 ~30 GB resident leaves ~2 GB on the 32 GB card for
activations+LoRA+opt — fixed-size gap, rank/res can't close it.

**Strategy reframed.** qfloat8 is not the dead option — it is the *fast,
full-quality* one, now host-clean, needing only a few GB of VRAM relief, with
cuda:1 (16 GB) idle. **Route A (offload cold transformer blocks → cuda:1) is
now the highest-value move** (targets the sole remaining gap; qint4+res768 is
the slow no-code fallback). Decision put to user.

---

### 2026-05-17 (cont.) — bitsandbytes 8-bit optimizer is broken on ROCm (real bug; but NOT this run's bottleneck)

qint4/rank32/res768 ran clean (0 OOM-skip, VRAM ~30/32, host stable) but
~60 s/it. Profiler over 10 steps: `train_loop` 60.25 s = `predict_unet`
20.3 + `backward` 18.1 + **`optimizer_step` 19.8** + misc. `encode_prompt`
0.0055 s ⇒ TE cache/unload conclusively perfect (saga closed). The
`optimizer_step` timer wraps **only** `self.optimizer.step()`
(`SDTrainer.py:2129-2130`; grad-clip is outside it) → `bitsandbytes.optim.
AdamW8bit.step()` is ~20 s for a **rank-32 LoRA** (should be ms). bnb 8-bit
kernels are non-functional on gfx1201/gfx1030; for LoRA-sized params 8-bit
state is irrelevant.

**Kludge added** (`toolkit/optimizer.py`, user-proposed, refined): on HIP,
transparently remap bnb 8-bit optimizers to nearest non-bnb
(`adamw8bit→adamw`, `adam8bit→adam`, `lion8bit→lion`, `ademamix8bit→adamw`)
with a warning. Per-type, NOT all→AdamW (wrong update rule/LR scale would
silently corrupt training — esp. lion/ademamix). Verified remaps correctly.
Mirrors the xformers→SDPA HIP kludge; added to CLAUDE.md kludge list.
Relaunched qint4/rank32/res768, config left `adamw8bit` to exercise the
kludge (scope `eve-q4adamw`).

**CORRECTION (post-relaunch profiler, measured — supersedes the ~40 s/it
projection above).** After the kludge: `optimizer_step` **19.8 → 2.17 s**,
but `train_loop` only **60.25 → 58.48 s** (~1.8 s, ~3% — NOT ⅓). The timed
sub-steps now sum to ~40 s while `train_loop` is ~58 s → a new **~18 s
untimed gap**. Root cause: GPU work is async; `bitsandbytes.AdamW8bit.step()`
was a **hard sync barrier** — its timer was charging the optimizer for the
*deferred dequant backlog from forward/backward*, not optimizer math. Plain
torch `AdamW` returns without forcing that drain, so the same backlog now
syncs elsewhere in the loop (untimed). **`optimizer_step` was a measurement
veil over the real bottleneck, never the bottleneck.** Earlier session
read-the-timer-literally error, now corrected.

**Verdict:** qint4-on-ROCm is **~58 s/it ≈ ~48 h/3000, GPU-bound on the
no-kernel int4 dequant in fwd+bwd (×2 grad-ckpt)** — no tuning knob left
(the missing low-bit ROCm kernel is the wall). The bnb→torch kludge **stays**
(bnb 8-bit genuinely broken on ROCm; free for LoRA; removes a real sync-heavy
optimizer that *would* be pure loss for qfloat8/Route B where no dequant
backlog masks it) — it just can't speed up a dequant-bound run. Reaffirms
qfloat8 + Route B (2× R9700) as the only fast endgame, now firmly
evidence-backed.

### 2026-05-17 — session paused; next session = Route B

Run killed (proof/soak done: qint4/rank32/res768 = steady progress, **0
OOM** — the original "see progress without OOMing" goal met). Committed this
session: `flux2_model.py` TE/leak fixes (`d421e9e`); `toolkit/optimizer.py`
bnb-ROCm kludge (this commit). **Commit-pending (deferred):**
`toolkit/util/quantize.py` (quanto-qbits shim) + `toolkit/data_loader.py`
(prefetch guard) — clean, can commit anytime; `jobs/process/
BaseSDTrainProcess.py` (stage-5 flush + caching-defer) — still carries the
temp MEMPROBE diag, strip before committing. **Next session: Route B**
(pipeline-split across 2 GPUs) per the PLANNED section below — execute when
the 2nd R9700 is installed.

Also added to CLAUDE.md kludge list this session: quanto-qbits shim,
`data_loader.py` prefetch_factor guard, `BaseSDTrainProcess.py:728` stage-5
flush — all commit-pending.

## PLANNED (not yet implemented) — Route B: pipeline-split the FLUX.2 transformer across 2 GPUs

> Decided 2026-05-17. **Execute only when a 2nd R9700 is installed.** This is
> the "proper" fix: real model/pipeline parallelism where the second GPU
> *computes*, not just stores. Until then, the working path is qfloat8 (fast,
> host-clean) + a VRAM-relief stopgap (config CPU layer-offload), or the slow
> qint4+res768 fallback.

### Why 2× R9700, not R9700 + 6900 XT (verified 2026-05-17)

- cuda:0 R9700 = **gfx1201 / RDNA4** → native FP8 (E4M3/E5M2 WMMA).
- cuda:1 6900 XT = **gfx1030 / RDNA2** → **NO native FP8** (first consumer
  RDNA fp8 is RDNA4; RDNA3 has WMMA bf16/fp16/int8 only). torch fp8 dtypes
  exist but are storage-only on gfx1030; matmul emulated via upcast.
- So splitting onto the 6900 XT: that half is slow (no fp8 HW), capped at
  16 GB, and cross-arch P2P (gfx1201↔gfx1030) almost certainly host-staged.
  Worst-of-both. **2× R9700** = symmetric {32 GB, RDNA4 native fp8} ×2 =
  **64 GB aggregate** (> the 48 GB NVIDIA reference), clean 50/50 split,
  same-arch P2P likely, full qfloat8 quality, restore res 1024 / rank 64,
  no offload thrash, ~2 h-class runs.

### PRIOR ART IN THIS REPO — study before designing from scratch

**FLUX.1 already ships a working multi-GPU transformer splitter.** Found while
auditing flux.1 for the flux.2 leaks (2026-05-17):
- config key **`split_model_over_gpus`** (`toolkit/config_modules.py:660`),
  guarded `if split_model_over_gpus and not is_flux: raise` (:661 — flux-only
  today; the gate to relax for flux.2).
- **`add_model_gpu_splitter_to_flux(transformer, other_module_param_count_scale=…)`**
  in **`toolkit/models/flux.py:121`**, invoked at
  `stable_diffusion_model.py:649` (FLUX.1) and `:861`; analogous usage in
  `toolkit/models/wan21/wan21.py:353` and `toolkit/models/cogview4.py:132`.
- It splits the diffusers `FluxTransformer2DModel` across GPUs by param-count
  scale — i.e. the exact "blocks on cuda:0 / cuda:1, activations cross the
  boundary" mechanism Route B needs, **already solved for diffusers Flux +
  Wan + CogView4**. FLUX.2 (`flux2/src/model.py`, custom `Flux2` class) is the
  only one lacking it.
- **Action when Route B is picked up:** read `flux.py:121` first. Likely the
  fastest path is to port/generalize `add_model_gpu_splitter_to_flux` to the
  custom Flux2 module + relax the `not is_flux` config guard, rather than
  build the split from scratch. Re-validate the single-device-collapse points
  below still apply (the splitter may already handle some).

### Design (pipeline/model-parallel, batch_size=1 ⇒ capacity split, no bubble overlap)

Split the 56 transformer blocks: `[0..k)` live **and compute** on cuda:0,
`[k..56)` on cuda:1; one small boundary tensor (the hidden state at block k)
crosses per direction per step (+ its grad on backward) — MBs, not the
whole-weight traffic of Route A.

**Code touch-points (all already mapped this session):**
1. **Config:** `split_devices: [cuda:0, cuda:1]` + `split_block_index: k`
   (or `split_ratio`, default 0.5 for symmetric).
2. **`flux2_model.py` / `quantize.py` block loop:** place each block on its
   target device during the per-block quantize (instead of all →
   `base_model.device_torch`). Quanto QTensor `.to(other_cuda)` works
   (verified for QBitsTensor/QBytesTensor moves this session).
3. **`flux2/src/model.py` Flux2 forward:** insert `hidden.to(cuda:1)` at the
   boundary; carry along everything that conditions later blocks (rotary
   embeds, attn masks, timestep/text/image conditioning). `Tensor.to`
   cross-device is autograd-aware ⇒ backward crosses back automatically.
4. **Gradient checkpointing:** ensure checkpoint segments do NOT straddle the
   boundary (checkpoint each side independently, or make block-k a segment
   boundary) — else recompute device context breaks.
5. **LoRA hijack:** `network.force_to(single_device)` ⇒ make per-module:
   each LoRA adapter on the same device as the block it wraps. Optimizer
   (adamw8bit) then spans both devices — build param groups accordingly.
6. **The single-device collapses — THE central obstacle (bug-class hit
   repeatedly this session):**
   - `BaseSDTrainProcess.py:728` `accelerator.prepare(self.sd.unet)` →
     `model.to(self.device)` collapses the split. **Skip `accelerator.prepare`
     for the split unet**; manage its devices manually (still prepare
     optimizer/dataloader). 
   - `base_model.py:~1445 set_device_state()` unconditional
     `self.unet.to(state['unet']['device'])` — make split-aware or bypass
     (already documented as not low_vram-aware).
   - durable-defer gate `:1762` — under Route B don't CPU-park; place the
     split after TE-caching+unload (the TE staging itself is unchanged/solved).
7. **Checkpoint save:** `patch_dequantization_on_save` already custom — make
   the state_dict gather pull from both devices.

**Validation ladder:** (a) numeric parity vs single-device (cosine / max-abs
on 1–2 steps, fixed seed); (b) short smoke — loss decreasing ~50 steps;
(c) full run. Compare loss curve to the qint4 baseline if one exists.

**Risks/unknowns:** accelerate optimizer/grad-scaler over multi-device param
groups; grad-checkpoint across the boundary; HF/accelerate hooks assuming one
device; same-arch R9700↔R9700 P2P actually enabled by the ROCm build (else
host-staged but still only 1 small tensor/step). None fatal; all isolable.

**Deliverables when picked up:** config keys; `flux2_model.py` per-block
placement + forward boundary; `BaseSDTrainProcess.py` prepare/set_device_state
gates; multi-device optimizer build; `tools/` parity test. Land behind the
config keys so single-GPU configs are unaffected (dormant when unset).

---

## PLANNED (not yet implemented) — Remote TE offload to Mac Studio

> Deferred future work. The `te_device: cpu` path already makes the box
> self-sufficient (one-time ~95 min CPU cache, then written to disk and reused
> on resume). This plan is an **optimization for future jobs / new datasets /
> caption changes**, where the 95 min CPU pass would otherwise recur. Not a
> rescue for any in-flight run. Do NOT start until explicitly picked up.

**Goal.** Move the one-time text-embed step (Mistral-Small-3.1-24B forward →
stack hidden layers `[10,20,30]` → `rearrange`) off the ROCm box onto a Mac
Studio (128 GB unified memory, Apple MPS), reachable over a 10 GbE LAN.

**Why not the obvious tool.** `torch.distributed.rpc` is "supported" but the
wrong fit: heterogeneous MPS↔ROCm backends, torch-version lock, autograd /
coupled-training machinery for what is a stateless one-time ~38-call batch.
Rejected.

### Options (ranked)

- **A — Lightweight HTTP microservice on the Mac (RECOMMENDED).** Mac runs a
  FastAPI/uvicorn daemon that loads *unquantized* Mistral once and exposes
  `POST /embed {prompts:[…]} → tensor`. Box gets a `te_remote_url:` config gate:
  when set, `flux2_model.load_te()` is **short-circuited** (Mistral never
  loaded/quantized/run on the box at all — also frees box RAM/VRAM and
  speeds startup), and `get_prompt_embeds()` calls the Mac, receives the
  tensor, `.to(cuda:0)`. Decoupled, robust, version-independent (safetensors
  transport). ~38 calls, one-time → ~1–3 min wall vs ~95 min.
- **B — Offline pre-seed cache files (zero runtime networking).** Run the exact
  extraction on the Mac offline; write ai-toolkit's on-disk embed-cache files
  directly (NFS-mount the box cache dir over 10 GbE, or rsync results). Most
  operationally robust (no live service) but needs reverse-engineering the
  cache key/format in `dataloader_mixins.py:cache_text_embeddings` **and** the
  separately-keyed sample-prompt cache in `SDTrainer.cache_sample_prompts`;
  must be redone whenever captions/sample prompts change.
- **C — `torch.distributed.rpc`.** Rejected (see above). Documented so it
  isn't re-proposed.
- **D — Do nothing (baseline).** Keep `te_device: cpu`. Valid if the same
  dataset is reused (cache hits on resume); the ~95 min is paid once.

### Recommended (A) deployment outline

1. **Mac env:** `uv` venv, Python 3.12. Deps: torch (default macOS wheel = MPS
   build), transformers, accelerate, safetensors, tokenizers, fastapi,
   uvicorn. **Pin `transformers == 5.5.3`** (box's version as of 2026-05-17 —
   re-check on the day) so `[10,20,30]` layer indexing matches exactly. Torch
   version need NOT match (safetensors is arch/version-stable).
2. **Weights:** `rsync` the box's existing HF snapshot of Mistral-Small-3.1-24B
   over 10 GbE (~48 GB ≈ ~1 min). Guarantees identical weights; do **not**
   re-pull from the hub (revision drift).
3. **dtype/MPS ladder:** (i) MPS + `PYTORCH_ENABLE_MPS_FALLBACK=1`, bf16
   (~48 GB, matches reference dtype) → (ii) fp32 on MPS (~96 GB, fits 128) →
   (iii) CPU-fp32 on the Mac (zero MPS-coverage risk; still ~20× the box CPU).
   Note `Mistral3ForConditionalGeneration` is the multimodal class (text +
   Pixtral vision) — only the text path is used but vision modules instantiate;
   MPS-fallback or CPU mitigates op gaps.
4. **Code:** standalone `mistral_embed_server.py` vendoring the EXACT logic
   from `flux2/src/pipeline.py:_get_mistral_prompt_embeds`
   (`OUTPUT_LAYERS_MISTRAL=[10,20,30]`, same `AutoProcessor`,
   `output_hidden_states=True`, `rearrange("b c l d -> b l (c d)")`) + a
   diff-test vs box reference to catch upstream drift.
5. **Daemon:** macOS `launchd` LaunchAgent, `KeepAlive`, wrapped in
   `caffeinate -s` (no sleep). Bind uvicorn to the 10 GbE IP (not localhost);
   allow through macOS firewall. `/health` endpoint.
6. **Box client:** `te_remote_url` config key; connect timeout + 2–3 retries;
   **fail loudly** if unreachable (no silent 95-min CPU fallback unless
   explicitly configured).
7. **Validation:** diff Mac (unquantized) vs the local CPU cache. Expect a
   *quant-sized* delta (box cache is uint4→dequant; Mac is unquantized → Mac is
   actually MORE faithful). Judge on cosine-sim / per-layer norm / NaN, NOT
   bit-equality. To isolate pure MPS drift: Mac-fp32 vs a box *unquantized*-CPU
   single-prompt run.

### Deliverables (when picked up)

- `tools/mistral_embed_server.py` (Mac) + `requirements`/`uv` lock.
- `launchd` plist + `rsync` weights helper script.
- `flux2_model.py`: `te_remote_url` branch + `load_te()` short-circuit
  (dormant when key unset — zero impact on existing configs).
- `tools/validate_embeds.py` (Mac-vs-local-cache diff).

### Decisions to resolve on the day

- Option A vs B (live service vs offline pre-seed).
- Mac dtype (bf16-MPS vs fp32) — drive by validation result.
- Behaviour when remote unreachable (hard fail vs configured CPU fallback).
- Whether to LM-only-load (skip vision tower) — only after proving identical
  hidden states.
