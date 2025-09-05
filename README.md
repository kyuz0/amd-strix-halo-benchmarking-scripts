# WAN 2.1/2.2 VAE Micro-Benchmark (Standalone)

This script is a **standalone benchmark harness** for the [Wan 2.1 VAE](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/blob/main/Wan2.1_VAE.pth), extracted from the [WAN 2.2 video generation pipeline](https://github.com/modelscope/wan-video).  
It is intended for **AMD ROCm** users (especially Strix Halo / gfx1151) who want to understand performance bottlenecks in the **VAE encode/decode stages**, which are often reported as slow or unstable compared to NVIDIA.

---

## 🎯 Purpose

- **Isolate the VAE**: instead of timing a full diffusion pipeline, this script benchmarks only the encode/decode passes of the VAE.
- **Compare configurations**: test baseline FP32, BF16 autocast, tiled encode/decode, and ROCm environment variable tweaks (`MIOPEN_FIND_MODE=2`, `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0`).
- **Collect system context**: prints PyTorch/HIP/ROCm versions, kernel, distro, GPU name/gfx, tensor shapes, and memory footprints.
- **Reproduce ROCm solver behavior**: shows how MIOpen chooses convolution solvers and whether tiling or env vars improve runtime.

---

## ⚙️ How it works

For each configuration:

1. Run a **warmup pass** (one encode + one decode) to trigger solver selection and caches.
2. Run a **single timed encode** and a **single timed decode**.
3. Print timings in seconds.

By default, the script tests **21 frames at 832×480** (the real WAN resolution).  
For quick smoke tests, you can use smaller sizes.

---

## 🚀 Usage

```bash
# Quick smoke test (tiny workload, finishes in seconds)
python wan_vae_benchmark.py --frames 5 --size 64*64

# Representative WAN resolution and frame count
python wan_vae_benchmark.py --frames 21 --size 832*480

# Force smaller tiles (to trigger tiled code paths)
python wan_vae_benchmark.py --frames 21 --size 832*480 --tile_px 128
````

Options:

* `--frames N` : number of frames in input video (default: 21)
* `--size WxH` : resolution (default: 832\*480)
* `--tile_px N`: tile size in pixels for tiled mode (default: 256)
* `--warmup N` : number of warmup passes per config (default: 1)

The script automatically downloads the VAE checkpoint from Hugging Face if it is not found in `~/.cache/wan`.

---

## 📊 Example Run and Discussion

Below is a sample run with **5 frames at 832×480**, which is small enough to complete quickly while still exercising the same solvers and memory layouts as larger workloads:

```
# (Paste your run output here)
```

➡️ **Why it matters**:
This smaller run is **representative** because:

* The **stride math** (t=4, h=8, w=8) is the same, so the latent shapes match what WAN would use at full resolution.
* The convolution kernel choices in MIOpen are already exposed at this resolution, so you see whether baseline picks slow solvers or if tiling/fastfind helps.
* The reduced frame count keeps runtime manageable, but still passes through the same code paths as full-length WAN generation.

---

## 🔎 Interpreting results

* **Baseline FP32**: a single large convolution workload. On Strix Halo, MIOpen may pick slow kernels → tens of seconds even for small inputs.
* **Tiled FP32**: splits into smaller convolutions that often select fast kernels → typically an order of magnitude faster.
* **Fastfind / disable-naive**: environment tweaks that sometimes force MIOpen to avoid slow solvers.
* **BF16**: autocast to bf16. Depending on driver maturity, may or may not improve speed; stability can vary.

---

## ⚠️ Notes & Caveats

* The script **does not average across runs**; each config reports a single encode+decode time (after warmup).
* For real benchmarking, repeat runs or run in separate processes to avoid cache effects.
* Performance depends heavily on ROCm version, MIOpen tuning DB, and kernel maturity for your GPU architecture.

---
