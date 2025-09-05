# VAE Micro-Benchmarks (WAN 2.1/2.2 & Qwen-Image)

This repository contains **standalone benchmark harnesses** for two different VAEs, targeting **AMD Strix Halo (`gfx1151`) under ROCm**:

- **WAN 2.1/2.2 Video VAE** (3D VAE for video, [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B))  
- **Qwen-Image VAE** (2D VAE for images, [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image))  

Both scripts measure **encode** (image/video → latent) and **decode** (latent → image/video) performance across precisions, tiling modes, and ROCm/MIOpen runtime options.

---

## Features

Each benchmark script:

- Prints a **full system/GPU banner** (kernel, ROCm/HIP, distro, gfx, VRAM).
- Automatically downloads the pretrained VAE from Hugging Face.
- Runs controlled encode/decode tests with:
  - **fp32** and **bf16** dtypes (autocast).
  - **No tiling vs. native tiling** (WAN uses manual tiling; Qwen uses native tiling via Diffusers).
  - ROCm/MIOpen environment toggles:
    - `MIOPEN_FIND_MODE=2` (force kernel finder).
    - `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0` (disable naive fallback).
- Supports **fast mode** (skip slow configs, run fp32+tiled only).
- Optionally runs with or without **warmup iterations**.

---

## Setup

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

# Install benchmark dependencies
python -m pip install diffusers einops
````

---

## Usage

### WAN VAE (Video)

```bash
python wan_vae_benchmark.py --frames 5 --size "832*480"
```

Arguments:

* `--frames` : number of frames in the input video (default `21`).
* `--size`   : resolution as `WxH` (default `832*480`).
* `--tile_px`: tile size in pixels (default `256`, auto sweep if enabled).
* `--warmup` : warmup iterations per config (default `1`).
* `--fast`   : run fp32+tiled only.

---

### Qwen VAE (Image)

```bash
python qwen_vae_benchmark.py --preset 1664x928
```

Arguments:

* `--preset` : choose from `1664x928`, `1328x1328`, or `both`.
* `--warmup` : warmup iterations per config (default `1`).
* `--fast`   : run fp32+tiled only.

👉 Qwen benchmark uses **native tiling from Diffusers** (`enable_tiling()`), no manual tiling or tile sweep.

---

## TL;DR Results

### WAN 2.1/2.2 (832×480, 5 frames, Strix Halo)

| Config                  | Encode (s) | Decode (s) | Total (s) | Notes                        |
| ----------------------- | ---------- | ---------- | --------- | ---------------------------- |
| **fp32 baseline**       | \~36–39    | \~58–59    | \~95–98   | No tiling, very slow         |
| **fp32 tiled (128 px)** | \~5.0      | \~8.0      | \~13.0    | 🚀 \~7× faster, best setting |
| **bf16 baseline**       | \~36–37    | \~58       | \~95      | Same as fp32 baseline        |
| **bf16 tiled (128 px)** | \~5.0–6.3  | \~8.0–10.5 | \~13–17   | Close to fp32 tiled          |
| **fp32 fastfind/naive** | \~36–37    | \~58–59    | \~95+     | No improvement               |

👉 **Best:** fp32+tiled at 128 px.
👉 Tiling is critical; without it, VAEs are unusably slow.

---

### Qwen-Image (1664×928, Strix Halo)

| Config                       | Encode (s) | Decode (s) | Total (s) | Notes                       |
| ---------------------------- | ---------- | ---------- | --------- | --------------------------- |
| **fp32 baseline**            | \~35       | \~56       | \~91      | No tiling, very slow        |
| **bf16 baseline**            | \~34.5     | \~54.8     | \~89      | Similar to fp32 baseline    |
| **fp32 native tiled**        | \~7.7      | \~12.2     | \~20      | 🚀 \~4.5× faster            |
| **bf16 native tiled**        | \~2.8      | \~4.8      | \~7.6     | 🚀 \~12× faster vs baseline |
| **fp32 tiled fastfind**      | \~7.7      | \~12.2     | \~19.9    | Same as fp32 tiled          |
| **fp32 tiled disable naive** | \~9.3      | \~15.2     | \~24.4    | Slower than normal tiled    |

👉 **Best:** bf16+tiled (native tiling) — fastest, with negligible accuracy loss in inference.

---

## Accuracy Notes

* **bf16 vs fp32**:
  bf16 reduces mantissa precision but keeps the same range as fp32.
  For inference workloads (like encode/decode here), **accuracy loss is negligible**.
  Visual outputs are nearly indistinguishable, while performance is dramatically better.

---

## Example Results

Collected logs are included in the repo:

* 📄 `results_wan-vae_frames-5_size-832x420-warmup.txt`
* 📄 `results_qwen-vae-warmup.txt`
* 📄 `results_qwen-vae-no-warmup.txt`

---

## Takeaways

* **WAN VAE**: fp32+tiled (128 px) is the sweet spot.
* **Qwen VAE**: bf16+tiled (native) is the best, and tiling is *mandatory*.
* **Warmup**: does not significantly affect stable timings, but improves reproducibility.
* Both models: without tiling, performance is **10–15× slower**.
