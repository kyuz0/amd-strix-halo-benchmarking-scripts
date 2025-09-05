# WAN 2.1/2.2 VAE Micro-Benchmark (Standalone, ROCm/Strix Halo)

This repository contains a **standalone benchmark harness** for the 
[WAN 2.1/2.2 Video VAE](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B).  
It is designed to evaluate **encode/decode performance** of the 3D VAE on 
AMD Strix Halo (`gfx1151`) under ROCm.

The script automatically:
- Downloads the pretrained VAE if missing.
- Prints a full system and GPU banner (kernel, ROCm/HIP, distro, gfx, VRAM).
- Runs a micro-benchmark of **encoding** (video → latent) and **decoding** (latent → video).
- Supports **fp32** and **bf16** dtypes, **tiled execution**, and ROCm/MIOpen environment flags.
- Finds the **best tile size** automatically for tiled execution.
- Optionally runs with or without warmup.

---

## Usage

### Install requirements
```bash
# Create and activate a virtual env
python -m venv .venv
source .venv/bin/activate

# Install ROCm nightly (TheRock) for gfx1151
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  rocm[libraries,devel]

# Install PyTorch nightly (TheRock) builds
python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  torch torchaudio torchvision

# Install Python deps from PyPI
python -m pip install einops
````

### Run a benchmark

```bash
python wan_vae_benchmark.py --frames 5 --size "832*480"
```

Arguments:

* `--frames` : number of frames in the input video (default `21`).
* `--size`   : resolution as `WxH` (default `832*480`).
* `--tile_px`: tile size in pixels when using tiling (default `256`).
* `--warmup` : number of warmup iterations per config (default `1`).
* `--fast`   : only run **fp32+tiled** (and skip slower configs).

---

## Fast sweep runner

For convenience, a helper script is provided:

```bash
./run_fast_suite.sh
```

This will:

* Run the benchmark at **WAN default res (832×480)** and **720p horizontal (720×1280)**.
* Test `--frames 5` (short) and `--frames 21` (long).
* Run both with warmup (`--warmup 1`) and without (`--warmup 0`).
* Only test **fp32+tiled** configs (fast path).

You can override the tile size:

```bash
TILE_PX=192 ./run_fast_suite.sh
```

---

## Understanding the tests

The benchmark runs several configurations:

* **fp32-baseline**: full precision, no tiling, default MIOpen.
* **fp32-tiled**: input is split into tiles (default 256 px, sweeps 256/192/128).
* **fp32-fastfind**: sets `MIOPEN_FIND_MODE=2` (forces kernel finder).
* **fp32-disable-naive**: sets `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0` (skip naive fallback).
* **bf16-baseline**: bfloat16 autocast, no tiling.
* **bf16-tiled(best)**: bfloat16 autocast, with best tile size found automatically.

The **auto tile sweep** determines the fastest tile size for the current resolution/frames,
typically **128 px** for WAN’s default (832×480).
This is critical: without tiling, large convolutions can take \~60–100 seconds, but with
tiling performance improves to \~13 seconds.

---

## Warmup

The `--warmup` flag runs dummy iterations before timing.
This helps remove one-off JIT/MIOpen overheads, but results show that warmup does **not
significantly change timings** for stable workloads. Still, it’s safer to use `--warmup 1`
for reproducibility.

---

## TL;DR Summary (832×480, 5 frames)

| Config                  | Encode (s) | Decode (s) | Total (s) | Notes                           |
| ----------------------- | ---------- | ---------- | --------- | ------------------------------- |
| **fp32 baseline**       | \~36–39    | \~58–59    | \~95–98   | Slow, no tiling                 |
| **fp32 tiled (128 px)** | \~5.0      | \~8.0      | \~13.0    | 🚀 \~7× faster, best setting    |
| **bf16 baseline**       | \~36–37    | \~58       | \~95      | Same as fp32 baseline           |
| **bf16 tiled (128 px)** | \~5.0–6.3  | \~8.0–10.5 | \~13–17   | Good, but slower than fp32      |
| **fp32 fastfind/naive** | \~36–37    | \~58–59    | \~95+     | No improvement, sometimes worse |

👉 **Takeaway**: Use **fp32+tiled (128 px)**. Everything else is much slower or no better.

---

## Example Results

Baseline WAN shape (`832×480`, 5 frames):

📄 `results_frames-5_size-832x480-warmup.txt`
📄 `results_frames-5_size-832x480-no-warmup.txt`

👉 Conclusion:
For WAN VAE on Strix Halo, **fp32+tiled with 128 px tiles is the clear winner**, reducing runtime
from \~95s to \~13s without loss of correctness.

---

## Next Steps

* Run with more frames (`--frames 21`) and higher resolutions (`720*1280`).
* Collect results from `./run_fast_suite.sh`.
* Compare across GPUs once other hardware is available.

