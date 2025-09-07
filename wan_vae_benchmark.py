#!/usr/bin/env python3
# Standalone WAN 2.1/2.2 VAE micro-benchmark for ROCm/Strix Halo (gfx1151)
# - Downloads VAE from Hugging Face if missing
# - Prints full system/GPU banner (kernel, distro, ROCm/HIP, gfx)
# - Benchmarks encode/decode at WAN-real shapes (default 832x480, 21 frames)
# - Runs fp32/bf16 + tiled and MIOpen toggles
# - NEW: auto tile sweep (FP32), BF16+tiled(best), tiled+fastfind, tiled+disable-naive, with a summary table

import os, sys, math, time, argparse, platform, subprocess, urllib.request, pathlib
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from einops import rearrange
except Exception:
    print("This script requires 'einops'. Install it via: pip install einops", file=sys.stderr)
    raise

# -----------------------------
# Config / defaults
# -----------------------------
HF_VAE_URL = "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/Wan2.1_VAE.pth?download=true"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "wan")
DEFAULT_VAE_PATH = os.path.join(DEFAULT_CACHE, "Wan2.1_VAE.pth")

# Wan2.1 effective strides (temporal 4, spatial 8x8)
STRIDE_T = 4
STRIDE_H = 8
STRIDE_W = 8

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

def reload_vae(vae):
    """Reload VAE to clear any internal MIOpen state."""
    device = vae.device
    dtype = vae.dtype
    del vae.vae
    torch.cuda.empty_cache()  # Clear GPU memory
    # Recreate
    vae.vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.vae.eval()

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
    mib = nbytes / (1024**2)   # MiB
    mb  = nbytes / (1000**2)   # MB
    return mib, mb

def tensor_bytes_for_dtype(shape: Tuple[int,...], dtype: torch.dtype) -> int:
    sizes = {
        torch.float32: 4, torch.bfloat16: 2, torch.float16: 2,
        torch.int8: 1, torch.uint8: 1,
    }
    elsize = sizes.get(dtype, 4)
    numel = 1
    for s in shape: numel *= int(s)
    return numel * elsize

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# -----------------------------
# Minimal Wan2.1 VAE (from your snippet)
# -----------------------------
CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.
    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                                          nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                                          nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x
    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)
    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                 attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        self.dim = dim; self.z_dim = z_dim; self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks; self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx]); feat_cache[idx] = cache_x; feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx]); feat_cache[idx] = cache_x; feat_idx[0] += 1
            else:
                x = layer(x)
        return x

class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                 attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim; self.z_dim = z_dim; self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks; self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1,2,3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 3, 3, padding=1))
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx]); feat_cache[idx] = cache_x; feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx]); feat_cache[idx] = cache_x; feat_idx[0] += 1
            else:
                x = layer(x)
        return x

def count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))

class WanVAE_(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
                 attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim; self.z_dim = z_dim; self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks; self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]; iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu
    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var); eps = torch.randn_like(std)
        return eps * std + mu
    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic: return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)
    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder); self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder); self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    cfg = dict(dim=96, z_dim=z_dim, dim_mult=[1,2,4,4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
    cfg.update(**kwargs)
    with torch.device('meta'):
        model = WanVAE_(**cfg)
    sd = torch.load(pretrained_path, map_location="cpu")
    try:
        model.load_state_dict(sd, assign=True)
    except TypeError:
        model = WanVAE_(**cfg)
        model.load_state_dict(sd)
    return model

class Wan2_1_VAE:
    def __init__(self, z_dim=16, vae_pth=DEFAULT_VAE_PATH, dtype=torch.float32, device="cuda"):
        self.dtype = dtype
        self.device = device
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std  = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std  = torch.tensor(std,  dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim).eval().requires_grad_(False).to(device)
    def encode(self, videos):
        # Use modern autocast to avoid deprecation warnings
        with torch.amp.autocast('cuda', dtype=self.dtype):
            return [ self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0) for u in videos ]
    def decode(self, zs):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            return [ self.model.decode(u.unsqueeze(0), self.scale).float().clamp_(-1,1).squeeze(0) for u in zs ]

# -----------------------------
# Benchmark helpers
# -----------------------------
def pad_to_stride(t, stride_h: int, stride_w: int):
    H, W = t.shape[-2:]
    pad_h = (math.ceil(H / stride_h) * stride_h) - H
    pad_w = (math.ceil(W / stride_w) * stride_w) - W
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return t

def bench_encode(vae: Wan2_1_VAE, video: torch.Tensor, dtype: str, tiled: bool, tile_px: int) -> float:
    _, _, H, W = video.shape
    start = time.perf_counter()
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if dtype == "bf16" else _NullCtx()
    with torch.no_grad(), ctx:
        if not tiled:
            _ = vae.encode([video])[0]
        else:
            th = tw = tile_px
            for y0 in range(0, H, th):
                for x0 in range(0, W, tw):
                    y1 = min(y0 + th, H); x1 = min(x0 + tw, W)
                    tile = pad_to_stride(video[:, :, y0:y1, x0:x1].contiguous(), STRIDE_H, STRIDE_W)
                    _ = vae.encode([tile])[0]
    time_sync()
    return time.perf_counter() - start

def bench_decode(vae: Wan2_1_VAE, latent: torch.Tensor, dtype: str, tiled: bool, latent_tile: int) -> float:
    _, _, lh, lw = latent.shape
    start = time.perf_counter()
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if dtype == "bf16" else _NullCtx()
    with torch.no_grad(), ctx:
        if not tiled:
            _ = vae.decode([latent])
        else:
            step = max(1, latent_tile)
            for y0 in range(0, lh, step):
                for x0 in range(0, lw, step):
                    y1 = min(y0 + step, lh); x1 = min(x0 + step, lw)
                    _ = vae.decode([latent[:, :, y0:y1, x0:x1].contiguous()])
    time_sync()
    return time.perf_counter() - start

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False

class _EnvCtx:
    """Temporarily set env vars for a block; restore after."""
    def __init__(self, env: Dict[str,str]):
        self.env = env or {}
        self.saved: Dict[str, Optional[str]] = {}
    def __enter__(self):
        for k,v in self.env.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = str(v)
    def __exit__(self, *a):
        for k,old in self.saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

# -----------------------------
# Download helper
# -----------------------------
def ensure_vae(vae_path: str):
    p = pathlib.Path(vae_path)
    if p.is_file():
        return vae_path
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VAE to {vae_path} ...")
    urllib.request.urlretrieve(HF_VAE_URL, vae_path)
    print("Download complete.")
    return vae_path

# -----------------------------
# Run one config (with warmup)
# -----------------------------
def run_once(vae, video, latent, *, dtype: str, tiled: bool, tile_px: int, warmup: int, env: Dict[str,str]) -> Tuple[float,float]:
    clear_miopen_cache()
    reload_vae(vae)
    latent_tile = max(1, tile_px // STRIDE_H)  # map pixel tiles to latent tiles
    with _EnvCtx(env):
        # warmup
        for _ in range(max(0, warmup)):
            _ = bench_encode(vae, video, dtype, tiled, tile_px)
            _ = bench_decode(vae, latent, dtype, tiled, latent_tile)
        # timed
        enc_s = bench_encode(vae, video, dtype, tiled, tile_px)
        dec_s = bench_decode(vae, latent, dtype, tiled, latent_tile)
    return enc_s, dec_s

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone WAN 2.1 VAE micro-benchmark (ROCm/Strix Halo)")
    ap.add_argument("--vae_path", default=DEFAULT_VAE_PATH, help="Path to Wan2.1_VAE.pth (downloaded if missing)")
    ap.add_argument("--size", default="832*480", help="WxH (default 832*480)")
    ap.add_argument("--frames", type=int, default=21, help="Number of frames (default 21)")
    ap.add_argument("--tile_px", type=int, default=256, help="Encode tile size in pixels when tiled")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iterations per config")
    ap.add_argument("--tile_sweep", default="256,192,128", help="Comma-separated tile sizes for auto sweep (FP32). Empty to skip.")
    ap.add_argument("--fast", action="store_true",
                help="Run only the best known config (fp32+tiled, tile_px)")
    args = ap.parse_args()

    # Size
    size_str = args.size.replace("x", "*")
    try:
        W, H = map(int, size_str.split("*"))
    except Exception:
        print("Invalid --size. Use WxH like 832*480", file=sys.stderr)
        return 2

    # VAE checkpoint
    vae_path = ensure_vae(args.vae_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init VAE
    vae = Wan2_1_VAE(vae_pth=vae_path, device=device, dtype=torch.float32)

    # Shapes
    video_shape  = (3, args.frames, H, W)
    latent_shape = (getattr(vae.model, "z_dim", 16),
                    (args.frames - 1) // STRIDE_T + 1,
                    H // STRIDE_H,
                    W // STRIDE_W)
    video  = torch.zeros(video_shape,  device=device, dtype=torch.float32)
    latent = torch.zeros(latent_shape, device=device, dtype=torch.float32)

    # Banner
    print("="*72)
    print("WAN 2.1/2.2 VAE Micro-Benchmark (Standalone)")
    print("="*72)
    print(f"PyTorch                : {torch.__version__}")
    print(f"HIP (torch.version.hip): {getattr(torch.version, 'hip', None)}")
    print(f"ROCm pkg version       : {get_rocm_version() or 'unknown'}")
    print(f"Kernel                 : {platform.release()}  ({platform.version()})")
    print(f"Distro                 : {get_distro()}")
    specs = get_gpu_specs()
    print("GPU:")
    for k in ("torch_device_name","product_name","card_series","vbios_version",
              "marketing_name","gfx_name","lspci","vram_total","vram_total_b"):
        if k in specs: print(f"  {k}: {specs[k]}")
    print()

    # Shapes / memory
    print("Shapes being tested:")
    print(f"  Encode input (video) : {video.shape}")
    v_bytes = tensor_bytes_for_dtype(video.shape, torch.float32)
    v_mib, v_mb = bytes_to_mib_mb(v_bytes)
    print(f"    ~{v_mib:.2f} MiB  ({v_mb:.2f} MB) @ float32")
    vbf_bytes = tensor_bytes_for_dtype(video.shape, torch.bfloat16)
    vbf_mib, vbf_mb = bytes_to_mib_mb(vbf_bytes)
    print(f"    ~{vbf_mib:.2f} MiB  ({vbf_mb:.2f} MB) @ bfloat16 (autocast)")
    print(f"  Decode input (latent): {latent.shape}")
    l_bytes = tensor_bytes_for_dtype(latent.shape, torch.float32)
    l_mib, l_mb = bytes_to_mib_mb(l_bytes)
    print(f"    ~{l_mib:.2f} MiB  ({l_mb:.2f} MB) @ float32")
    lbf_bytes = tensor_bytes_for_dtype(latent.shape, torch.bfloat16)
    lbf_mib, lbf_mb = bytes_to_mib_mb(lbf_bytes)
    print(f"    ~{lbf_mib:.2f} MiB  ({lbf_mb:.2f} MB) @ bfloat16 (autocast)")
    print(f"  Frames (video)       : {args.frames}  →  Frames (latent): {latent.shape[1]}  (t-stride={STRIDE_T})")
    print(f"  Resolution           : {W}x{H}  →  Latent: {latent.shape[3]}x{latent.shape[2]}  (h-stride={STRIDE_H}, w-stride={STRIDE_W})")
    print()

    # ---- Original minimal set (kept) ----
    results = []  # list of dicts for summarizing

    def record(name, dtype, tiled, tile_px, env, enc_s, dec_s):
        results.append({
            "name": name, "dtype": dtype, "tiled": tiled, "tile_px": tile_px,
            "env": dict(env) if env else {}, "encode_s": enc_s, "decode_s": dec_s,
            "total_s": enc_s + dec_s
        })
        print(f"   Encode: {enc_s:.4f}s   Decode: {dec_s:.4f}s\n")


    # >>> FAST MODE: bail out early after one fp32+tiled run
    if args.fast:
        print("-- FAST MODE: fp32+tiled only --")
        print(f"   dtype=fp32, tiled=True, env=default, tile_px={args.tile_px}")
        enc_s, dec_s = run_once(
            vae, video, latent,
            dtype="fp32", tiled=True, tile_px=args.tile_px,
            warmup=args.warmup, env={}
        )
        record("fp32-tiled-fast", "fp32", True, args.tile_px, {}, enc_s, dec_s)

        # Summary table (reuse same printing style)
        if results:
            print("\n== Summary (sorted by total time) ==")
            headers = ["name","dtype","tiled","tile_px","encode_s","decode_s","total_s"]
            widths = [max(len(h), 12) for h in headers]
            rows = []
            for r in sorted(results, key=lambda r: r["total_s"]):
                rows.append([
                    r["name"],
                    r["dtype"],
                    str(r["tiled"]),
                    str(r["tile_px"]) if r["tile_px"] is not None else "-",
                    f"{r['encode_s']:.4f}",
                    f"{r['decode_s']:.4f}",
                    f"{r['total_s']:.4f}",
                ])
            line = "  ".join(h.ljust(w) for h,w in zip(headers, widths))
            print(line); print("-" * len(line))
            for row in rows:
                print("  ".join(s.ljust(w) for s,w in zip(row, widths)))

        print("\nDone.")
        return 0
    # <<< end FAST MODE

    # 1) FP32 baseline (untiled)
    print("-- Test: fp32-baseline")
    print("   dtype=fp32, tiled=False, env=default")
    enc_s, dec_s = run_once(vae, video, latent, dtype="fp32", tiled=False, tile_px=args.tile_px, warmup=args.warmup, env={})
    record("fp32-baseline", "fp32", False, None, {}, enc_s, dec_s)

    # 2) FP32 tiled (single run with user-provided tile size)
    print("-- Test: fp32-tiled")
    print(f"   dtype=fp32, tiled=True, env=default, tile_px={args.tile_px}")
    enc_s, dec_s = run_once(vae, video, latent, dtype="fp32", tiled=True, tile_px=args.tile_px, warmup=args.warmup, env={})
    record("fp32-tiled", "fp32", True, args.tile_px, {}, enc_s, dec_s)

    # 3) Untiled with ROCm env toggles (fastfind / disable-naive)
    for env_name, env_vars in [
        ("fp32-fastfind", {"MIOPEN_FIND_MODE": "2"}),
        ("fp32-disable-naive", {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0"}),
    ]:
        print(f"-- Test: {env_name}")
        print(f"   dtype=fp32, tiled=False, env={env_vars}")
        enc_s, dec_s = run_once(vae, video, latent, dtype="fp32", tiled=False, tile_px=args.tile_px, warmup=args.warmup, env=env_vars)
        record(env_name, "fp32", False, None, env_vars, enc_s, dec_s)

    # 4) BF16 baseline (untiled)
    print("-- Test: bf16-baseline")
    print("   dtype=bf16, tiled=False, env=default")
    enc_s, dec_s = run_once(vae, video, latent, dtype="bf16", tiled=False, tile_px=args.tile_px, warmup=args.warmup, env={})
    record("bf16-baseline", "bf16", False, None, {}, enc_s, dec_s)

    # ---- NEW: Auto tile sweep (FP32 tiled) ----
    sweep = [int(s.strip()) for s in args.tile_sweep.split(",") if s.strip().isdigit()] if args.tile_sweep else []
    best_tile = None
    if sweep:
        print("-- Auto tile sweep (FP32, tiled) --------------------------------")
        for tpx in sweep:
            print(f"   test tile_px={tpx}")
            enc_s, dec_s = run_once(vae, video, latent, dtype="fp32", tiled=True, tile_px=tpx, warmup=args.warmup, env={})
            record(f"fp32-tiled-{tpx}", "fp32", True, tpx, {}, enc_s, dec_s)
        # pick best by total time among those sweep entries
        sweep_entries = [r for r in results if r["name"].startswith("fp32-tiled-")]
        if sweep_entries:
            best_entry = min(sweep_entries, key=lambda r: r["total_s"])
            best_tile = best_entry["tile_px"]
            print(f"   → Best tile size: {best_tile} px (total {best_entry['total_s']:.4f}s)")
        print()

    # ---- NEW: BF16 + tiled(best) ----
    if best_tile:
        print("-- Test: bf16-tiled(best)")
        print(f"   dtype=bf16, tiled=True, env=default, tile_px={best_tile}")
        enc_s, dec_s = run_once(vae, video, latent, dtype="bf16", tiled=True, tile_px=best_tile, warmup=args.warmup, env={})
        record("bf16-tiled-best", "bf16", True, best_tile, {}, enc_s, dec_s)

        # ---- NEW: Tiled + fastfind / disable-naive (sanity) ----
        for env_name, env_vars in [
            ("fp32-tiled-fastfind-best", {"MIOPEN_FIND_MODE": "2"}),
            ("fp32-tiled-disable-naive-best", {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0"}),
        ]:
            print(f"-- Test: {env_name}")
            print(f"   dtype=fp32, tiled=True, env={env_vars}, tile_px={best_tile}")
            enc_s, dec_s = run_once(vae, video, latent, dtype="fp32", tiled=True, tile_px=best_tile, warmup=args.warmup, env=env_vars)
            record(env_name, "fp32", True, best_tile, env_vars, enc_s, dec_s)

    # ---- Summary table ----
    if results:
        print("\n== Summary (sorted by total time) ==")
        headers = ["name","dtype","tiled","tile_px","encode_s","decode_s","total_s"]
        widths = [max(len(h), 12) for h in headers]
        rows = []
        for r in sorted(results, key=lambda r: r["total_s"]):
            rows.append([
                r["name"],
                r["dtype"],
                str(r["tiled"]),
                str(r["tile_px"]) if r["tile_px"] is not None else "-",
                f"{r['encode_s']:.4f}",
                f"{r['decode_s']:.4f}",
                f"{r['total_s']:.4f}",
            ])
        # print header
        line = "  ".join(h.ljust(w) for h,w in zip(headers, widths))
        print(line)
        print("-" * len(line))
        for row in rows:
            print("  ".join(s.ljust(w) for s,w in zip(row, widths)))

    print("\nDone.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
