#!/usr/bin/env python3
# Qwen-Image ONE-COMMAND denoising bench (ROCm, verbose, AOTriton/CK)
# - No hipBLASLt anywhere (removed).
# - Compares attention:
#     * "flash" with FA backend = {aotriton, ck}
#     * "math" (no SDPA)
# - Also sweeps ROCBLAS_DEVICE_MEMORY_SIZE {1024, 2048} MB for linear layers.
# - Verbose: prints env, loads, shapes, warmup, timings.
#
# Examples:
#   python qwen_denoise_bench.py
#   python qwen_denoise_bench.py --auto-dtypes bf16,fp16 --aspects 16:9 --batches 1,2

import os, sys, argparse, subprocess, json, time, math
from itertools import product
from contextlib import nullcontext

# ---------------- Orchestrator ----------------

def parse_list_str(s): return [x.strip() for x in s.split(",") if x.strip()]

def orchestrate(args):
    attns       = ["flash", "math"]          # "flash" hits SDPA FlashAttention; "math" is non-SDPA
    fa_backends = ["aotriton", "ck"]         # preferred FlashAttention backend (ROCm)
    mems_mb     = [1024, 2048]
    dtypes      = parse_list_str(args.auto_dtypes)

    combos = list(product(attns, fa_backends, mems_mb, dtypes))
    total  = len(combos)
    results = []

    print("\n=== Qwen-Image ONE-COMMAND Bench (verbose) ===")
    print(f"Search grid: attn={attns}  fa_backends={fa_backends}  rocblas_mem_mb={mems_mb}  dtypes={dtypes}")
    print(f"Global test params: aspects={args.aspects}  batches={args.batches}  base={args.base}  mode={args.mode}  steps={args.steps}")
    print("================================================\n")

    for idx, (attn, fa, mem_mb, dt) in enumerate(combos, 1):
        print(f">>> [{idx}/{total}] Launching worker: attn={attn}, FA={fa}, ROCBLAS_DEVICE_MEMORY_SIZE={mem_mb}MB, dtype={dt}")
        env = os.environ.copy()
        # No hipBLASLt knobs at all.
        env["ROCBLAS_DEVICE_MEMORY_SIZE"] = str(mem_mb * 1024 * 1024)
        # Prefer CK only when requested; default is AOTriton
        if fa == "ck":
            env["TORCH_ROCM_FA_PREFER_CK"] = "1"
        else:
            env.pop("TORCH_ROCM_FA_PREFER_CK", None)

        cmd = [sys.executable, sys.argv[0],
               "--worker",
               "--worker-attn", attn,
               "--worker-fa", fa,
               "--worker-mem-mb", str(mem_mb),
               "--worker-dtype", dt,
               "--mode", args.mode,
               "--base", str(args.base),
               "--aspects", args.aspects,
               "--batches", args.batches,
               "--steps", str(args.steps)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dur = time.perf_counter() - t0

        if p.stderr:  # echo worker logs
            print(p.stderr, end="")

        if p.returncode != 0:
            print(f"[worker ERROR] attn={attn} fa={fa} mem={mem_mb}MB dtype={dt} (exit {p.returncode})")
            continue

        rec = None
        for line in reversed(p.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    rec = json.loads(line)
                    break
                except Exception:
                    pass
        if rec is None:
            print(f"[worker PARSE ERROR] stdout had no JSON line.\nSTDOUT:\n{p.stdout}")
            continue

        rec.update(dict(attn=attn, fa=fa, rocblas_mem_mb=mem_mb, dtype=dt, wall_s=dur))
        results.append(rec)
        print(f"<<< [{idx}/{total}] Done: avg_s={rec['avg_s']:.3f}s  (wall {dur:.1f}s)\n")

    if not results:
        print("No successful runs.")
        sys.exit(1)

    results.sort(key=lambda r: r["avg_s"])
    best = results[0]

    print("\n==================== BEST CONFIG ====================")
    print(f"avg_s={best['avg_s']:.3f}s  p50={best['p50_s']:.3f}s  p90={best['p90_s']:.3f}s")
    print(f"attn={best['attn']}  FA={best['fa']}  dtype={best['dtype']}  ROCBLAS_DEVICE_MEMORY_SIZE={best['rocblas_mem_mb']}MB")
    print("Reproduce env:")
    print(f"  ROCBLAS_DEVICE_MEMORY_SIZE={best['rocblas_mem_mb']*1024*1024}")
    if best["fa"] == "ck":
        print("  TORCH_ROCM_FA_PREFER_CK=1")
    print("=====================================================\n")

    headers = ["rank","avg_s","p50_s","p90_s","attn","fa","dtype","rocblas_mem_mb","W","H","batch","aspect","mode","wall_s"]
    print("== Summary (sorted by avg_s) ==")
    print("  ".join(h.ljust(10) for h in headers))
    print("-"*len("  ".join(h.ljust(10) for h in headers)))
    for i, r in enumerate(results, 1):
        print("  ".join([
            str(i).ljust(10),
            f"{r['avg_s']:.3f}".ljust(10),
            f"{r['p50_s']:.3f}".ljust(10),
            f"{r['p90_s']:.3f}".ljust(10),
            r["attn"].ljust(10),
            r["fa"].ljust(10),
            r["dtype"].ljust(10),
            str(r["rocblas_mem_mb"]).ljust(10),
            str(r["W"]).ljust(10),
            str(r["H"]).ljust(10),
            str(r["batch"]).ljust(10),
            r["aspect"].ljust(10),
            r["mode"].ljust(10),
            f"{r['wall_s']:.1f}".ljust(10),
        ]))

# ---------------- Worker (single config; prints verbosely to STDERR) ----------------

def worker(args):
    import torch
    from diffusers import QwenImagePipeline, QwenImageEditPipeline

    def log(msg): print(msg, file=sys.stderr, flush=True)

    def pick_device():
        return "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"

    def parse_aspect(s):
        w, h = s.split(":"); return int(w), int(h)

    def pack_latents(latents, b, nc, h, w):
        x = latents.view(b, nc, h // 2, 2, w // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(b, (h // 2) * (w // 2), nc * 4)

    def compute_dims(base_px, aspect_wh, vae_scale=8):
        aw, ah = aspect_wh
        ratio = aw / ah
        w = int(round(math.sqrt(base_px * ratio) / 32) * 32)
        h = int(round(w / ratio / 32) * 32)
        mult = vae_scale * 2
        w = (w // mult) * mult; h = (h // mult) * mult
        H = 2 * (int(h) // (vae_scale * 2))
        W = 2 * (int(w) // (vae_scale * 2))
        return w, h, H, W

    def set_attention_backend(module, name):
        name = name.lower()
        try:
            from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
        except Exception:
            AttnProcessor = AttnProcessor2_0 = None
        if name in ("flash","sdpa","attn2","attn_2_0"):
            if hasattr(module,"set_attn_processor") and AttnProcessor2_0 is not None:
                try: module.set_attn_processor(AttnProcessor2_0()); return "sdpa"
                except Exception: pass
            if hasattr(module,"enable_sdpa"):
                try: module.enable_sdpa(); return "sdpa"
                except Exception: pass
            return "default"
        if name in ("math","default"):
            if hasattr(module,"set_attn_processor") and AttnProcessor is not None:
                try: module.set_attn_processor(AttnProcessor()); return "math"
                except Exception: pass
            return "default"
        return "default"

    def sdpa_ctx(kind):
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if kind == "flash":
                # FlashAttention (AOTriton/CK) first; fall back to math if unavailable
                return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], set_priority=True)
            if kind == "sdpa":  # not used directly, but kept for completeness
                return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
            if kind == "math":
                return sdpa_kernel([SDPBackend.MATH], set_priority=True)
        except Exception:
            pass
        return nullcontext()

    @torch.no_grad()
    def timed_forward(pipe, sdpa_kind, synth):
        kwargs = dict(
            hidden_states=synth["latent_model_input"],
            timestep=synth["timestep"],
            guidance=None,
            encoder_hidden_states_mask=synth["prompt_mask"],
            encoder_hidden_states=synth["prompt_embeds"],
            img_shapes=synth["img_shapes"],
            txt_seq_lens=synth["txt_seq_lens"],
            attention_kwargs={},
            return_dict=False,
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with sdpa_ctx(sdpa_kind):
            out = pipe.transformer(**kwargs)[0]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return time.perf_counter() - t0, out

    def build_synth_inputs(pipe, mode, bs, w, h, H, W, dtype):
        in_ch = int(pipe.transformer.config.in_channels)
        lat_ch = in_ch // 4
        embed_dim = int(getattr(pipe.transformer.config, "cross_attention_dim", 3584))
        seq_len = 6
        device = next(pipe.transformer.parameters()).device
        lat = torch.randn(bs, 1, lat_ch, H, W, device=device, dtype=dtype)
        latents = pack_latents(lat, bs, lat_ch, H, W)
        prompt_embeds = torch.randn(bs, seq_len, embed_dim, device=device, dtype=dtype)
        prompt_mask   = torch.ones(bs, seq_len, device=device, dtype=torch.long)
        if mode == "edit":
            image_latents = torch.randn_like(latents)
            img_shapes = [[(1, H//2, W//2), (1, H//2, W//2)]] * bs
            latent_model_input = torch.cat([latents, image_latents], dim=1)
        else:
            img_shapes = [[(1, H//2, W//2)]] * bs
            latent_model_input = latents
        t = torch.tensor([1000.0], device=device, dtype=dtype).expand(bs)
        return dict(
            latent_model_input=latent_model_input,
            timestep=t/1000.0,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_mask.sum(dim=1).tolist(),
        )

    import platform, time
    device = pick_device()
    torch.set_default_device(device)

    # header + env
    print_env = f"ROCBLAS_DEVICE_MEMORY_SIZE={os.environ.get('ROCBLAS_DEVICE_MEMORY_SIZE')}  TORCH_ROCM_FA_PREFER_CK={os.environ.get('TORCH_ROCM_FA_PREFER_CK')}"
    log("----------------------------------------------------------------")
    log("Worker starting...")
    log(f"PyTorch: {torch.__version__}  HIP: {getattr(torch.version,'hip',None)}")
    log(f"Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log(f"Kernel : {platform.release()}")
    log(f"Env    : {print_env}")
    log(f"Config : attn={args.worker_attn}  fa={args.worker_fa}  dtype={args.worker_dtype}  aspects={args.aspects}  batches={args.batches}  base={args.base}  mode={args.mode}  steps={args.steps}")
    log("----------------------------------------------------------------")

    # dtype
    if args.worker_dtype == "bf16": dtype = torch.bfloat16
    elif args.worker_dtype in ("fp16","half"): dtype = torch.float16
    else: dtype = torch.bfloat16

    # Prefer FA backend explicitly (API, if available)
    try:
        pref_in = "ck" if args.worker_fa == "ck" else "aotriton"
        torch.backends.cuda.preferred_rocm_fa_library(pref_in)
        fa_pref_out = torch.backends.cuda.preferred_rocm_fa_library()
        log(f"[fa] preferred_rocm_fa_library set to: {fa_pref_out}")
    except Exception as e:
        log(f"[fa] could not set preferred_rocm_fa_library: {e}")

    modes = ["gen","edit"] if args.mode == "both" else [args.mode]
    from diffusers import QwenImagePipeline, QwenImageEditPipeline

    results = []
    for mode in modes:
        model_id = "Qwen/Qwen-Image" if mode == "gen" else "Qwen/Qwen-Image-Edit"
        cls = QwenImagePipeline if mode == "gen" else QwenImageEditPipeline
        log(f"[load] mode={mode} model={model_id} dtype={dtype}")
        t0 = time.perf_counter()
        # IMPORTANT: use torch_dtype and hard-cast the transformer
        pipe = cls.from_pretrained(model_id, torch_dtype=dtype)
        load_s = time.perf_counter() - t0
        pipe.to(device)
        pipe.transformer.to(dtype=dtype)
        pipe.transformer.eval()
        uniq_dtypes = {p.dtype for p in pipe.transformer.parameters() if p.dtype.is_floating_point}
        log(f"[dtype] transformer param dtypes: {sorted(str(d) for d in uniq_dtypes)}")
        log(f"[load] done in {load_s:.2f}s  in_channels={int(pipe.transformer.config.in_channels)}")

        # attention setup
        eff_attn = set_attention_backend(pipe.transformer, "flash" if args.worker_attn == "flash" else "math")
        sdpa_kind = "flash" if (args.worker_attn == "flash" and eff_attn == "sdpa") else "math"
        log(f"[attn] requested={args.worker_attn}  effective={eff_attn}/{sdpa_kind}  fa_pref={os.environ.get('TORCH_ROCM_FA_PREFER_CK','aotriton')}")

        aspects = [tuple(map(int, a.split(":"))) for a in args.aspects.split(",") if a]
        batches = [int(x) for x in args.batches.split(",") if x]

        for aspect in aspects:
            W, H, Hp, Wp = compute_dims(args.base*args.base, aspect, getattr(pipe, "vae_scale_factor", 8))
            for bs in batches:
                log(f"[shape] aspect={aspect[0]}:{aspect[1]}  W={W} H={H}  packed(H/2={Hp//2}, W/2={Wp//2})  batch={bs}")
                synth = build_synth_inputs(pipe, mode, bs, W, H, Hp, Wp, dtype)
                ws, _ = timed_forward(pipe, sdpa_kind, synth)  # warmup
                log(f"[warmup] {ws:.3f}s")
                per = []
                for i in range(args.steps):
                    s, _ = timed_forward(pipe, sdpa_kind, synth)
                    per.append(s)
                    log(f"[step {i+1}/{args.steps}] {s:.3f}s")
                avg = sum(per)/len(per)
                p50 = sorted(per)[len(per)//2]
                p90 = sorted(per)[max(0,int(len(per)*0.9)-1)]
                log(f"[result] avg={avg:.3f}s p50={p50:.3f}s p90={p90:.3f}s\n")
                results.append(dict(
                    mode=mode, W=W, H=H, batch=bs, aspect=f"{aspect[0]}:{aspect[1]}",
                    avg_s=avg, p50_s=p50, p90_s=p90
                ))

        del pipe
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    agg_avg = sum(r["avg_s"] for r in results) / len(results)
    agg_p50 = sorted([r["p50_s"] for r in results])[len(results)//2]
    agg_p90 = sorted([r["p]()]()_
