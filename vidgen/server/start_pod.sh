#!/usr/bin/env bash
# Start the vidgen GPU server on a generic GPU pod (RunPod, Vast.ai, Lightning AI,
# Paperspace, a university box ...). Run from the repo root:
#
#   bash vidgen/server/start_pod.sh                 # Wan 1.3B + Cloudflare tunnel
#   MODEL=ltx bash vidgen/server/start_pod.sh
#   TUNNEL=0 PORT=8000 bash vidgen/server/start_pod.sh   # pod already exposes a public port
#
# Set VIDGEN_TOKEN to keep the same token across restarts.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-wan-1.3b}"
PORT="${PORT:-8000}"
TUNNEL="${TUNNEL:-1}"

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
  echo "torch with CUDA is missing: use a PyTorch pod image, or pip install torch first." >&2
  exit 1
}
pip install -q -r vidgen/server/requirements.txt

args=(--model "$MODEL" --port "$PORT")
[ "$TUNNEL" = "0" ] && args+=(--no-tunnel)
exec python -m vidgen.server.launch "${args[@]}"
