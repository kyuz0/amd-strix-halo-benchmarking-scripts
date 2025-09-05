#!/usr/bin/env bash
# Fast VAE benchmark sweep (fp32 + tiled only) for WAN on ROCm/Strix Halo
# - Runs with and without warmup
# - Covers WAN default res (832x480) and 720p (both orientations)
# - Prints each command before executing
#
# Usage:
#   ./run_fast_suite.sh                  # uses wan_vae_benchmark.py in CWD, tile_px=128
#   ./run_fast_suite.sh /path/to/wan_vae_benchmark.py
#   TILE_PX=192 ./run_fast_suite.sh      # override tile size

set -Eeuo pipefail

SCRIPT_PATH="${1:-wan_vae_benchmark.py}"
TILE_PX="${TILE_PX:-128}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo
echo "=============================================="
echo " WAN VAE FAST SUITE (fp32+tiled only)"
echo "=============================================="
echo "Benchmark script : $SCRIPT_PATH"
echo "Tile size (px)   : $TILE_PX"
echo "Python           : $(command -v python || true)"
echo "Date             : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo

# Sizes and frame counts to test
SIZES=(
  "832*480"     # WAN default
  "1280*720"    # 720p vertical
  "720*1280"    # 720p horizontal
)

FRAMES=(5 21)
WARMUPS=(1 0)

run() {
  echo
  echo "------------------------------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------------------------------"
  eval "$@"
}

for W in "${WARMUPS[@]}"; do
  echo
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo " Warmup mode: --warmup $W"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  for SZ in "${SIZES[@]}"; do
    for F in "${FRAMES[@]}"; do
      CMD="python \"$SCRIPT_PATH\" --fast --frames $F --size \"$SZ\" --tile_px $TILE_PX --warmup $W"
      run "$CMD"
    done
  done
done

echo
echo "All fast benchmarks complete."
