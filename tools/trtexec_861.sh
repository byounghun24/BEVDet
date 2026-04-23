#!/usr/bin/env bash
set -euo pipefail

TRT_LIB_DIR="/home/byounghun/miniconda/envs/py38/lib/python3.8/site-packages/tensorrt_libs"
TRTEXEC_BIN="/opt/tensorrt/bin/trtexec"
MMDEPLOY_PLUGIN_CANDIDATES=(
  "/home/byounghun/workspace/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"
  "/home/byounghun/workspace/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
)

if [[ ! -x "${TRTEXEC_BIN}" ]]; then
  echo "ERROR: trtexec not found at ${TRTEXEC_BIN}" >&2
  exit 1
fi

if [[ ! -d "${TRT_LIB_DIR}" ]]; then
  echo "ERROR: TensorRT 8.6.1 libs not found at ${TRT_LIB_DIR}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${TRT_LIB_DIR}:${LD_LIBRARY_PATH:-}"

PLUGIN_SO=""
for cand in "${MMDEPLOY_PLUGIN_CANDIDATES[@]}"; do
  if [[ -f "${cand}" ]]; then
    PLUGIN_SO="${cand}"
    break
  fi
done

if [[ -n "${PLUGIN_SO}" ]]; then
  exec "${TRTEXEC_BIN}" --plugins="${PLUGIN_SO}" "$@"
else
  exec "${TRTEXEC_BIN}" "$@"
fi
