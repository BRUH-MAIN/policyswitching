#!/bin/bash
# Start the local VLM (llama.cpp server, OpenAI-compatible API) used by src/vlm_nav.
#
# Defaults are romen's setup: llama.cpp build b10686 (Vulkan) and Gemma-4-E4B-it
# Q4_K_M with its vision projector, on the "New Volume" NTFS partition. That
# partition is often left dirty by Windows fast-startup, so mount it read-only:
#   udisksctl mount -b /dev/nvme0n1p4 -o ro
#
# Every setting can be overridden via the environment. Notes:
# - IMAGE_MAX_TOKENS: vision-token budget per image. The encoder's default is
#   small, and terrain relief is fine detail -- this is a Phase-2 sweep variable,
#   not a constant.
# - PARALLEL: slots for concurrent requests (the executor queries several envs
#   per control tick). Context is split across slots, so CTX is per-slot x PARALLEL.
# - UBATCH: must be >= the image token count. Gemma-4's vision tokens attend
#   bidirectionally, so a whole image has to fit in one micro-batch; with
#   IMAGE_MAX_TOKENS=560 and llama.cpp's default 512 the server aborts on
#   GGML_ASSERT(causal_attn || n_ubatch >= n_tokens_all) at the first image.
# - Thinking is disabled per request (chat_template_kwargs.enable_thinking=false
#   in vlm_backend.py): with it on, Gemma 4 spends the whole max_tokens budget in
#   reasoning_content and returns an empty answer.
set -euo pipefail
LLAMA_DIR=${LLAMA_DIR:-/home/rohan/autotest/bin/llama-b10686}
MODEL_DIR=${MODEL_DIR:-"/run/media/rohan/New Volume/models"}
VLM_MODEL=${VLM_MODEL:-"$MODEL_DIR/gemma-4-E4B-it-Q4_K_M.gguf"}
VLM_MMPROJ=${VLM_MMPROJ:-"$MODEL_DIR/mmproj-F16.gguf"}
PORT=${PORT:-8091}
PARALLEL=${PARALLEL:-2}
CTX=${CTX:-8192}
IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS:-}
UBATCH=${UBATCH:-1024}

[ -f "$VLM_MODEL" ] || { echo "model not found: $VLM_MODEL (is the drive mounted?)" >&2; exit 1; }
EXTRA=()
[ -n "$IMAGE_MAX_TOKENS" ] && EXTRA+=(--image-max-tokens "$IMAGE_MAX_TOKENS" --image-min-tokens "$IMAGE_MAX_TOKENS")
cd "$LLAMA_DIR"
exec env LD_LIBRARY_PATH="$LLAMA_DIR" ./llama-server -m "$VLM_MODEL" --mmproj "$VLM_MMPROJ" \
  -ngl 99 -c "$CTX" -np "$PARALLEL" -b "$UBATCH" -ub "$UBATCH" --host 127.0.0.1 --port "$PORT" "${EXTRA[@]}"
