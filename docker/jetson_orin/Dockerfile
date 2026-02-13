FROM nvcr.io/nvidia/l4t-ml:r35.3.1-py3

# -------------------------
# Versions (Jetson AGX Orin / JP 5.1.1)
# -------------------------
ARG CUDA=11.4
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=2.0.0+nv23.05
ARG TORCHVISION_VERSION=0.15.1+nv23.05
ARG ONNXRUNTIME_VERSION=1.14.1
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ARG MMDEPLOY_VERSION=

ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# Pattern B: host UID/GID user
# -------------------------
ARG USERNAME=byounghun
ARG UID=1021
ARG GID=1021
ENV USERNAME=${USERNAME}

RUN set -eux; \
    (getent group ${GID} || groupadd -g ${GID} ${USERNAME}); \
    id -u ${USERNAME} >/dev/null 2>&1 || useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}; \
    mkdir -p /home/${USERNAME}/workspace; \
    chown -R ${UID}:${GID} /home/${USERNAME}

# -------------------------
# apt mirrors (optional)
# -------------------------
ARG USE_SRC_INSIDE=false
RUN if [ "${USE_SRC_INSIDE}" = "true" ] ; then \
        sed -i s/ports.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

# -------------------------
# system deps
# -------------------------
RUN set -eux; \
    chmod 777 /tmp; \
    apt-get clean && apt-get update; \
    apt-get install -y --no-install-recommends \
        sudo \
        curl \
        ca-certificates \
        vim \
        libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
        git wget \
        libssl-dev libopencv-dev libspdlog-dev \
        build-essential cmake pkg-config ninja-build \
        libyaml-cpp-dev libeigen3-dev libjpeg-dev \
    ; \
    rm -rf /var/lib/apt/lists/*

# sudo (NOPASSWD)
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# -------------------------
# Miniforge (aarch64) + conda env
# -------------------------
ENV CONDA_DIR=/home/${USERNAME}/miniconda
ENV CONDA_BIN=${CONDA_DIR}/bin/conda
ENV CONDA_ENV=py38

RUN set -eux; \
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh; \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR}; \
    rm -f /tmp/miniforge.sh; \
    ${CONDA_BIN} clean -t -p -i -y; \
    chown -R ${UID}:${GID} ${CONDA_DIR}

ENV PATH=${CONDA_DIR}/bin:$PATH

RUN set -eux; \
    ${CONDA_BIN} create -y -n ${CONDA_ENV} python=${PYTHON_VERSION}; \
    ${CONDA_BIN} run -n ${CONDA_ENV} python -V; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install -U pip setuptools wheel; \
    ${CONDA_BIN} run -n ${CONDA_ENV} conda install -y \
        conda-build pyyaml numpy ipython cython typing typing_extensions ninja; \
    ${CONDA_BIN} clean -ya

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc && \
    echo "conda activate ${CONDA_ENV}" >> /home/${USERNAME}/.bashrc && \
    chown ${UID}:${GID} /home/${USERNAME}/.bashrc

# -------------------------
# PyTorch into env (JetPack wheel channel)
# -------------------------
ARG TORCH_WHL_URL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
ARG TORCHVISION_WHL_URL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torchvision-0.15.1+nv23.05-cp38-cp38-linux_aarch64.whl
RUN set -eux; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install --no-cache-dir \
        ${TORCH_WHL_URL} ${TORCHVISION_WHL_URL}

# -------------------------
# MMCV into env (build from source on aarch64)
# -------------------------
RUN set -eux; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install openmim; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install \
        mmcv-full==${MMCV_VERSION} --no-binary mmcv-full

# -------------------------
# Workspace
# -------------------------
WORKDIR /home/${USERNAME}/workspace

# -------------------------
# ONNXRuntime (aarch64 TGZ)
# -------------------------
RUN set -eux; \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz; \
    tar -zxf onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz

ENV ONNXRUNTIME_DIR=/home/${USERNAME}/workspace/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}

# -------------------------
# TensorRT python bindings -> copy into env site-packages (best-effort)
# -------------------------
RUN set -eux; \
    PY_SITE=$(${CONDA_BIN} run -n ${CONDA_ENV} python -c "import site; print(site.getsitepackages()[0])"); \
    for p in \
        /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* \
        /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* \
    ; do \
        if [ -e "$p" ]; then \
            cp -r $p ${PY_SITE}/; \
        fi; \
    done

# -------------------------
# MMDeploy (build + install editable into env)
# -------------------------
ENV TENSORRT_DIR=/usr
RUN set -eux; \
    git clone https://github.com/HuangJunJie2017/mmdeploy.git; \
    cd mmdeploy; \
    if [ -z "${MMDEPLOY_VERSION}" ]; then \
        echo "No MMDeploy version passed in, building on master"; \
    else \
        git checkout tags/v${MMDEPLOY_VERSION} -b tag_v${MMDEPLOY_VERSION}; \
    fi; \
    git submodule update --init --recursive; \
    mkdir -p build; \
    cd build; \
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" ..; \
    make -j"$(nproc)"; \
    cd ..; \
    ${CONDA_BIN} run -n ${CONDA_ENV} python -m pip install -e . --no-build-isolation

# -------------------------
# PPL.CV (build)
# -------------------------
RUN set -eux; \
    git clone https://github.com/openppl-public/ppl.cv.git; \
    cd ppl.cv; \
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION}; \
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

# -------------------------
# Build MMDeploy SDK
# -------------------------
RUN set -eux; \
    cd /home/${USERNAME}/workspace/mmdeploy; \
    rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc || true; \
    mkdir -p build; \
    cd build; \
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/home/${USERNAME}/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES=all; \
    make -j"$(nproc)"; \
    make install

ENV LD_LIBRARY_PATH="/home/${USERNAME}/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# -------------------------
# Extra python deps (into env)
# -------------------------
RUN set -eux; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install \
        mmdet==2.25.1 \
        mmsegmentation==0.25.0; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install \
        pycuda \
        lyft_dataset_sdk \
        networkx==2.2 \
        numba==0.53.0 \
        numpy \
        nuscenes-devkit \
        plyfile \
        scikit-image \
        tensorboard \
        trimesh==2.35.39

# -------------------------
# Runtime: make env default + switch to user
# -------------------------
ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:$PATH

RUN chown -R ${UID}:${GID} /home/${USERNAME}/workspace /home/${USERNAME}/miniconda

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

