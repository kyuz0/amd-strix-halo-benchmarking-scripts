#!/usr/bin/env python3
# Standalone Qwen-Image VAE micro-benchmark for ROCm/Strix Halo (gfx1151)
# - Loads VAE from Hugging Face via Diffusers (Qwen/Qwen-Image, subfolder="vae")
# - Prints full system/GPU banner (kernel, distro, ROCm/HIP, gfx)
# - Benchmarks encode/decode on Qwen-native shapes (1664x928, 1328x1328)
# - Runs fp32/bf16 + tiled and MIOpen toggles, with auto tile sweep
# - Fast mode (fp32+tiled only) for quick checks

import os, sys, math, time, argparse, platform, subprocess
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F

try:
    from diffusers import AutoencoderKLQwenImage
except Exception:
    print("This script requires 'diffusers'. Install it via: pip install diffusers", file=sys.stderr)
    raise

def clear_miopen_cache():
    """Clear MIOpen cache directory to ensure clean state between tests."""
    import shutil
    cache_dir = os.path.expanduser("~/.cache/miopen")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"   Cleared MIOpen cache")
        except Exception as e:
            print(f"   Warning: Could not clear MIOpen cache: {e}")
    else:
        print(f"   MIOpen cache not found")

# -----------------------------
# System / GPU info helpers
# -----------------------------
def run_cmd(cmd, timeout=6) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout, text=True).strip()
    except Exception:
        return ""

def get_distro() -> str:
    try:
        if hasattr(platform, "freedesktop_os_release"):
            info = platform.freedesktop_os_release()
            name = info.get("PRETTY_NAME") or f"{info.get('NAME','')} {info.get('VERSION','')}"
            return (name or "").strip()
    except Exception:
        pass
    for p in ("/etc/os-release", "/usr/lib/os-release"):
        if os.path.exists(p):
            try:
                data = {}
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if "=" in line:
                            k,v = line.split("=",1)
                            data[k]=v.strip('"')
                return data.get("PRETTY_NAME") or f"{data.get('NAME','')} {data.get('VERSION','')}"
            except Exception:
                pass
    return platform.platform()

def get_rocm_version() -> str:
    try:
        import importlib.metadata as im
        for pkg in ("_rocm_sdk_core", "rocm"):
            try:
                return im.version(pkg)
            except Exception:
                pass
    except Exception:
        pass
    smi_ver = run_cmd(["rocm-smi", "--showdriverversion"])
    if smi_ver:
        return smi_ver.splitlines()[-1].strip()
    return ""

def parse_rocm_smi_csv_line(line: str) -> Dict[str,str]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 2:
        return {parts[0]: parts[1]}
    if len(parts) >= 3:
        return {parts[-2]: parts[-1]}
    return {}

def get_gpu_specs() -> Dict[str, str]:
    specs: Dict[str, str] = {}
    if torch.cuda.is_available():
        try:
            specs["torch_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    smi_csv = run_cmd(["rocm-smi", "-a", "--csv"])
    if smi_csv:
        for line in smi_csv.splitlines():
            kv = parse_rocm_smi_csv_line(line)
            for k, v in kv.items():
                k_norm = k.lower().replace(" ", "_").replace("(", "").replace(")", "")
                specs[k_norm] = v
    info = run_cmd(["rocminfo"])
    if info:
        gfx = ""
        asic = ""
        for line in info.splitlines():
            ls = line.strip()
            if ls.startswith("Name:") and "gfx" in ls and not gfx:
                gfx = ls.split(":",1)[1].strip()
            if ls.startswith("Marketing Name:") and not asic:
                asic = ls.split(":",1)[1].strip()
        if gfx:  specs["gfx_name"] = gfx
        if asic: specs["marketing_name"] = asic
    pci = run_cmd(["sh", "-lc", "lspci -nn | grep -Ei 'vga|display|gpu' | grep -i amd | head -n1 | cut -d: -f3-"])
    if pci:
        specs["lspci"] = " ".join(pci.split())
    return specs

def bytes_to_mib_mb(nbytes: int) -> Tuple[float, float]:
    mib = nbytes / (1024**2)
    mb  = nbytes / (1000**2)
    return mib, mb

def tensor_bytes_for_dtype(shape: Tuple[int,...], dtype: torch.dtype) -> int:
    sizes = {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2, torch.int8: 1, torch.uint8: 1}
    elsize = sizes.get(dtype, 4)
    numel = 1
    for s in shape: numel *= int(s)
    return numel * elsize

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# -----------------------------
# Qwen VAE wrapper
# -----------------------------
class QwenVAE:
    """
    Thin wrapper over diffusers.AutoencoderKL for encode/decode with scaling.
    """
    def __init__(self, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype  = dtype
        # Download/load from Hugging Face
        self.vae: AutoencoderKLQwenImage = AutoencoderKLQwenImage.from_pretrained(
            "Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        # scaling factor used by diffusers (latent scaling)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        # infer downsample factor by probing a tiny input once
        self.down_factor = self._infer_downsample_factor()

    @torch.no_grad()
    def _infer_downsample_factor(self) -> int:
        H, W = 64, 64
        # Qwen VAE expects [B, C, T, H, W]; use T=1 for images
        x = torch.zeros(1, 3, 1, H, W, device=self.device, dtype=torch.float32)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            posterior = self.vae.encode(x).latent_dist
            z = posterior.mean * self.scaling_factor  # [B, C_lat, T, H/df, W/df]
        _, _, _, lh, lw = z.shape
        df_h = max(1, round(H / lh))
        df_w = max(1, round(W / lw))
        return int((df_h + df_w) // 2)


    @torch.no_grad()
    def encode(self, images: torch.Tensor, dtype: str = "fp32"):
        """
        images: [B, 3, H, W]
        returns z: [B, C_lat, H/df, W/df]
        """
        amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        # Add T=1 for Qwen
        x = images.unsqueeze(2)  # [B,3,1,H,W]
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            posterior = self.vae.encode(x).latent_dist
            z = posterior.mean * self.scaling_factor  # [B,C,1,H/df,W/df]
        return z.squeeze(2).float()


    @torch.no_grad()
    def decode(self, latents: torch.Tensor, dtype: str = "fp32"):
        """
        latents: [B, C_lat, H/df, W/df]
        returns images: [B, 3, H, W]
        """
        amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        z = latents.unsqueeze(2)  # [B,C,1,H/df,W/df]
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            x = self.vae.decode(z / self.scaling_factor).sample  # [B,3,1,H,W]
        return x.squeeze(2).float().clamp_(-1, 1)


# -----------------------------
# Benchmark helpers (2D tiling)
# -----------------------------
def pad_to_stride_2d(img: torch.Tensor, stride_h: int, stride_w: int):
    # img: [B, C, H, W]
    H, W = img.shape[-2:]
    pad_h = (math.ceil(H / stride_h) * stride_h) - H
    pad_w = (math.ceil(W / stride_w) * stride_w) - W
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return img

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False

class _EnvCtx:
    def __init__(self, env: Dict[str,str]):
        self.env = env or {}
        self.saved: Dict[str, Optional[str]] = {}
    def __enter__(self):
        for k,v in self.env.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = str(v)
    def __exit__(self, *a):
        for k,old in self.saved.items():
            if old is None: os.environ.pop(k, None)
            else: os.environ[k] = old

@torch.no_grad()
def bench_encode(vae: QwenVAE, img: torch.Tensor, dtype: str, tiled: bool) -> float:
    start = time.perf_counter()
    _ = vae.encode(img, dtype=dtype)
    time_sync()
    return time.perf_counter() - start

@torch.no_grad()
def bench_decode(vae: QwenVAE, lat: torch.Tensor, dtype: str, tiled: bool) -> float:
    start = time.perf_counter()
    _ = vae.decode(lat, dtype=dtype)
    time_sync()
    return time.perf_counter() - start

def run_once(vae, img, lat, *, dtype: str, tiled: bool, warmup: int, env: Dict[str,str]) -> Tuple[float,float]:
    clear_miopen_cache()
    with _EnvCtx(env):
        if hasattr(vae.vae, "enable_tiling") and hasattr(vae.vae, "disable_tiling"):
            if tiled:
                vae.vae.enable_tiling()
            else:
                vae.vae.disable_tiling()
        for _ in range(max(0, warmup)):
            _ = bench_encode(vae, img, dtype, tiled)
            _ = bench_decode(vae, lat, dtype, tiled)
        enc_s = bench_encode(vae, img, dtype, tiled)
        dec_s = bench_decode(vae, lat, dtype, tiled)
    return enc_s, dec_s

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone Qwen-Image VAE micro-benchmark (ROCm/Strix Halo)")
    ap.add_argument("--preset", choices=["1664x928","1328x1328","both"], default="1664x928",
                    help="Qwen-native sizes to test (default: 1664x928)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iterations per config (default 1)")
    ap.add_argument("--fast", action="store_true",
                help="Run only fp32 with native tiling (skip baselines)")
    args = ap.parse_args()

    # Sizes
    sizes = []
    if args.preset == "both":
        sizes = [(1664,928), (1328,1328)]
    elif args.preset == "1664x928":
        sizes = [(1664,928)]
    else:
        sizes = [(1328,1328)]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # VAE
    vae = QwenVAE(device=device, dtype=torch.float32)

    # Banner
    print("="*72)
    print("Qwen-Image VAE Micro-Benchmark (Standalone)")
    print("="*72)
    print(f"PyTorch                : {torch.__version__}")
    print(f"HIP (torch.version.hip): {getattr(torch.version, 'hip', None)}")
    print(f"ROCm pkg version       : {get_rocm_version() or 'unknown'}")
    print(f"Kernel                 : {platform.release()}  ({platform.version()})")
    print(f"Distro                 : {get_distro()}")
    specs = get_gpu_specs()
    print("GPU:")
    for k in ("torch_device_name","product_name","card_series","vbios_version","marketing_name","gfx_name","lspci","vram_total","vram_total_b"):
        if k in specs: print(f"  {k}: {specs[k]}")
    print(f"\nVAE scaling_factor     : {vae.scaling_factor}")
    print(f"VAE downsample factor  : {vae.down_factor}x")
    print()

    # Iterate target sizes
    for (W, H) in sizes:
        # Tensors
        B = 1
        img  = torch.zeros(B, 3, H, W, device=device, dtype=torch.float32)

        # FAST: derive latent shape directly (Qwen down_factor is 8, presets are divisible)
        latent_ch = getattr(vae.vae.config, "latent_channels", 16)
        lat_h = H // vae.down_factor
        lat_w = W // vae.down_factor
        lat   = torch.zeros(B, latent_ch, lat_h, lat_w, device=device, dtype=torch.float32)

        print(f"Shapes being tested ({W}x{H}):")
        print(f"  Encode input (image) : {tuple(img.shape)}")
        v_bytes = tensor_bytes_for_dtype(img.shape, torch.float32); v_mib, v_mb = bytes_to_mib_mb(v_bytes)
        print(f"    ~{v_mib:.2f} MiB  ({v_mb:.2f} MB) @ float32")
        vbf_bytes = tensor_bytes_for_dtype(img.shape, torch.bfloat16); vbf_mib, vbf_mb = bytes_to_mib_mb(vbf_bytes)
        print(f"    ~{vbf_mib:.2f} MiB  ({vbf_mb:.2f} MB) @ bfloat16 (autocast)")
        print(f"  Decode input (latent): {tuple(lat.shape)}")
        l_bytes = tensor_bytes_for_dtype(lat.shape, torch.float32); l_mib, l_mb = bytes_to_mib_mb(l_bytes)
        print(f"    ~{l_mib:.2f} MiB  ({l_mb:.2f} MB) @ float32")
        lbf_bytes = tensor_bytes_for_dtype(lat.shape, torch.bfloat16); lbf_mib, lbf_mb = bytes_to_mib_mb(lbf_bytes)
        print(f"    ~{lbf_mib:.2f} MiB  ({lbf_mb:.2f} MB) @ bfloat16 (autocast)")
        print(f"  Resolution           : {W}x{H}  →  Latent: {lat.shape[-1]}x{lat.shape[-2]}  (down={vae.down_factor}x)")
        print()

        # Results collector
        results = []
        def record(name, dtype, tiled, env, enc_s, dec_s):
            results.append({
                "name": name,
                "dtype": dtype,
                "tiled": tiled,
                "env": dict(env) if env else {},
                "encode_s": enc_s,
                "decode_s": dec_s,
                "total_s": enc_s + dec_s,
            })
            print(f"   Encode: {enc_s:.4f}s   Decode: {dec_s:.4f}s\n")

        # FAST MODE
        if args.fast:
            print("-- FAST MODE: fp32+tiled only --")
            print(f"   dtype=fp32, tiled=native, env=default")
            enc_s, dec_s = run_once(vae, img, lat, dtype="fp32", tiled=True, warmup=args.warmup, env={})
            record("fp32-tiled-fast", "fp32", True, {}, enc_s, dec_s)
            _print_summary(results)
            continue

        # Baselines (untiled)
        print("-- Test: fp32-baseline")
        print("   dtype=fp32, tiled=False, env=default")
        enc_s, dec_s = run_once(vae, img, lat, dtype="fp32", tiled=False, warmup=args.warmup, env={})
        record("fp32-baseline", "fp32", False, {}, enc_s, dec_s)

        print("-- Test: bf16-baseline")
        print("   dtype=bf16, tiled=False, env=default")
        enc_s, dec_s = run_once(vae, img, lat, dtype="bf16", tiled=False, warmup=args.warmup, env={})
        record("bf16-baseline", "bf16", False, {}, enc_s, dec_s)

        # Untiled with MIOpen env toggles
        for env_name, env_vars in [
            ("fp32-fastfind", {"MIOPEN_FIND_MODE": "2"}),
            ("fp32-disable-naive", {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0"}),
        ]:
            print(f"-- Test: {env_name}")
            print(f"   dtype=fp32, tiled=False, env={env_vars}")
            enc_s, dec_s = run_once(vae, img, lat, dtype="fp32", tiled=False, warmup=args.warmup, env=env_vars)
            record(env_name, "fp32", False, env_vars, enc_s, dec_s)

        # Single tiled run (user tile)
        print("-- Test: fp32-tiled")
        print(f"   dtype=fp32, tiled=native, env=default, tile_px=auto")
        enc_s, dec_s = run_once(vae, img, lat, dtype="fp32", tiled=True, warmup=args.warmup, env={})
        record("fp32-tiled", "fp32", True, {}, enc_s, dec_s)

        # bf16 + native tiled
        print("-- Test: bf16-tiled")
        print("   dtype=bf16, tiled=native, env=default")
        enc_s, dec_s = run_once(vae, img, lat, dtype="bf16", tiled=True, warmup=args.warmup, env={})
        record("bf16-tiled", "bf16", True, {}, enc_s, dec_s)

        # tiled + env sanity (native tiling)
        for env_name, env_vars in [
            ("fp32-tiled-fastfind", {"MIOPEN_FIND_MODE": "2"}),
            ("fp32-tiled-disable-naive", {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0"}),
        ]:
            print(f"-- Test: {env_name}")
            print(f"   dtype=fp32, tiled=native, env={env_vars}")
            enc_s, dec_s = run_once(vae, img, lat, dtype="fp32", tiled=True, warmup=args.warmup, env=env_vars)
            record(env_name, "fp32", True, env_vars, enc_s, dec_s)

        _print_summary(results)

    print("\nDone.")
    return 0

def _print_summary(results: List[Dict]):
    if not results: return
    print("\n== Summary (sorted by total time) ==")
    headers = ["name","dtype","tiled","encode_s","decode_s","total_s"]
    widths = [max(len(h), 12) for h in headers]
    rows = []
    for r in sorted(results, key=lambda r: r["total_s"]):
        rows.append([
            r["name"],
            r["dtype"],
            str(r["tiled"]),
            f"{r['encode_s']:.4f}",
            f"{r['decode_s']:.4f}",
            f"{r['total_s']:.4f}",
        ])
    line = "  ".join(h.ljust(w) for h,w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(s.ljust(w) for s,w in zip(row, widths)))

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
