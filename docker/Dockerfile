FROM nvcr.io/nvidia/tensorrt:22.07-py3

# -------------------------
# Versions
# -------------------------
ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ARG MMDEPLOY_VERSION=   # optional: pass --build-arg MMDEPLOY_VERSION=1.3.1 etc.

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
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

# (원본처럼 tuna로 강제)
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list && \
    cat /etc/apt/sources.list

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
        build-essential cmake pkg-config \
    ; \
    rm -rf /var/lib/apt/lists/*

# sudo (NOPASSWD)
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# -------------------------
# Miniconda (user home) + ToS accept
# -------------------------
ENV CONDA_DIR=/home/${USERNAME}/miniconda
ENV CONDA_BIN=${CONDA_DIR}/bin/conda
ENV CONDA_ENV=py38

RUN set -eux; \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; \
    /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR}; \
    rm -f /tmp/miniconda.sh; \
    ${CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; \
    ${CONDA_BIN} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    ${CONDA_BIN} clean -t -p -i -y; \
    chown -R ${UID}:${GID} ${CONDA_DIR}

# conda on PATH (base)
ENV PATH=${CONDA_DIR}/bin:$PATH

# -------------------------
# Create env (py38) and install base packages into env
# (IMPORTANT: keep PATH as base conda during build to avoid env path confusion)
# -------------------------
RUN set -eux; \
    ${CONDA_BIN} create -y -n ${CONDA_ENV} python=${PYTHON_VERSION}; \
    ${CONDA_BIN} run -n ${CONDA_ENV} python -V; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install -U pip setuptools wheel; \
    ${CONDA_BIN} run -n ${CONDA_ENV} conda install -y \
        conda-build pyyaml numpy ipython cython typing typing_extensions \
        mkl mkl-include ninja; \
    ${CONDA_BIN} clean -ya

# (선택) bash 들어가면 자동 activate
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /home/${USERNAME}/.bashrc && \
    echo "conda activate ${CONDA_ENV}" >> /home/${USERNAME}/.bashrc && \
    chown ${UID}:${GID} /home/${USERNAME}/.bashrc

# -------------------------
# PyTorch into env (use conda run to avoid env path issues)
# Note: 채널 URL은 /linux-64/ 붙이지 말고 루트로
# -------------------------
RUN set -eux; \
    ${CONDA_BIN} run -n ${CONDA_ENV} conda install -y \
        pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch; \
    ${CONDA_BIN} clean -ya

# -------------------------
# MMCV into env (pip)
# -------------------------
RUN set -eux; \
    ${CONDA_BIN} run -n ${CONDA_ENV} pip install mmcv-full==${MMCV_VERSION} \
        -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# -------------------------
# Workspace
# -------------------------
WORKDIR /home/${USERNAME}/workspace

# -------------------------
# ONNXRuntime (x86) - use prebuilt TGZ only (pip install 불가/불필요)
# -------------------------
RUN set -eux; \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz; \
    tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

ENV ONNXRUNTIME_DIR=/home/${USERNAME}/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
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
# ENV TENSORRT_DIR=/usr

# RUN set -eux; \
#     git clone https://github.com/HuangJunJie2017/mmdeploy.git; \
#     cd mmdeploy; \
#     if [ -z "${MMDEPLOY_VERSION}" ]; then \
#         echo "No MMDeploy version passed in, building on master"; \
#     else \
#         git checkout tags/v${MMDEPLOY_VERSION} -b tag_v${MMDEPLOY_VERSION}; \
#     fi; \
#     git submodule update --init --recursive; \
#     mkdir -p build; \
#     cd build; \
#     cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" ..; \
#     make -j"$(nproc)"; \
#     cd ..; \
#     ${CONDA_BIN} run -n ${CONDA_ENV} pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

    ### install mmdeploy
# ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ARG VERSION
RUN git clone https://github.com/HuangJunJie2017/mmdeploy.git
RUN cd mmdeploy &&\
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
    git submodule update --init --recursive &&\
    mkdir -p build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
    make -j$(nproc) &&\
    cd .. &&\
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
        mmsegmentation==0.25.0 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple; \
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
        trimesh==2.35.39 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# -------------------------
# Runtime: make env default + switch to user
# -------------------------
ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:$PATH

RUN chown -R ${UID}:${GID} /home/${USERNAME}/workspace /home/${USERNAME}/miniconda

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace
