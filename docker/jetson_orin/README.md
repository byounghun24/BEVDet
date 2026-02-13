# Jetson AGX Orin Docker (JP 5.1.1)

This is a Jetson-specific Docker that follows the same build flow as
`docker/Dockerfile` (user mapping, conda env, torch/mmcv, mmdeploy, pplcv, SDK build),
but adapted for ARM64 + JetPack 5.1.1.

Target stack:
- JetPack `5.1.1`
- CUDA `11.4.315`
- cuDNN `8.6.0`
- TensorRT `8.5.2.2`
- yaml-cpp / Eigen3 / libjpeg (APT packages)

Base:
- `nvcr.io/nvidia/l4t-ml:r35.3.1-py3`

## Build

```bash
docker build -t bevdet-tcar-orin:jp5.1.1 -f docker/jetson_orin/Dockerfile .
```

If you need specific user mapping:

```bash
docker build -t bevdet-tcar-orin:jp5.1.1 \
  --build-arg USERNAME=$USER \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -f docker/jetson_orin/Dockerfile .
```

## Run

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v $PWD:/workspace/BEVDet \
  -v /data:/data \
  bevdet-tcar-orin:jp5.1.1
```

## After container starts

Install this repo in editable mode:

```bash
cd /workspace/BEVDet
pip install -v -e .
```

Notes:
- Torch/Torchvision are installed from NVIDIA JetPack wheels (`jp/v511`).
- `mmcv-full` is built from source (`--no-binary`) on aarch64.
- TensorRT engine must be built on Jetson (ONNX can be produced on server).
