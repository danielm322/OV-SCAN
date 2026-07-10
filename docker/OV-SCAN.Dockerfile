# ------------------------------------------------------------------------------
# Base: CUDA & Ubuntu
# ------------------------------------------------------------------------------
ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# ------------------------------------------------------------------------------
# Set global environment variables
# ------------------------------------------------------------------------------
ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"
    
    
# ------------------------------------------------------------------------------
# Install OS-level tools and dependencies 
# ------------------------------------------------------------------------------
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        bzip2 ca-certificates git libglib2.0-0 libsm6 libxext6 libxrender1 \
        mercurial openssh-client procps subversion wget curl vim unzip unrar \
        build-essential software-properties-common libgl1 cmake \
        libboost-dev libopenexr-dev libeigen3-dev xvfb libgl1-mesa-glx \
        libglib2.0-0 ffmpeg openmpi-bin openmpi-common libopenmpi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Install Miniconda 
# ------------------------------------------------------------------------------
ARG CONDA_VERSION=py310_22.11.1-1
ARG PYTHON_VERSION=3.10

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda install python=${PYTHON_VERSION} -y && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy

# ------------------------------------------------------------------------------
# Install base Python packages
# ------------------------------------------------------------------------------
RUN pip install \
    pyyaml h5py numpy scipy scikit-learn pandas pillow scikit-image \
    matplotlib seaborn jupyterlab jupyter

# ------------------------------------------------------------------------------
# Install PyTorch and CUDA-compiled packages 
# ------------------------------------------------------------------------------
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHSCATTER_VERSION=2.1.2
ARG SPCONV_VERSION=2.3.6
RUN export CU_VERSION=128 && \
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu${CU_VERSION} && \
    pip install tensorboard \
        spconv-cu120==${SPCONV_VERSION} \
        torch-scatter==${TORCHSCATTER_VERSION} -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CU_VERSION}.html

# ------------------------------------------------------------------------------
# Install deep learning toolkits
# ------------------------------------------------------------------------------
ARG MMCV_VERSION=2.3.3
ARG MMDET_VERSION=3.3.0
ARG MMSEG_VERSION=1.2.2
RUN export CU_VERSION=128 && export TORCH_VER_SHORT=2.10 && \
    pip install onedl-mmcv==${MMCV_VERSION} -f https://mmwheels.onedl.ai/cu${CU_VERSION}-torch280/index.html &&\
    pip install mmsegmentation==${MMSEG_VERSION} mmdet==${MMDET_VERSION} ipdb 

# ------------------------------------------------------------------------------
# Install 3D Vision and Dataset Tools 
# ------------------------------------------------------------------------------
RUN pip install \
    open3d==0.18.0 easydict==1.13 opencv-python==4.10.0.82 \
    pyquaternion==0.9.9 SharedArray==3.2.4 kornia==0.7.3 filterpy==1.4.5 motmetrics==1.4.0 \
    tensorflow==2.15.1 av2==0.2.1

# ------------------------------------------------------------------------------
# Extra utilities and dev tools 
# ------------------------------------------------------------------------------
RUN pip install \
    setuptools==69.5.1 yapf==0.40.2 protobuf==4.25.3 \
    einops==0.8.0 fvcore iopath==0.1.10 timm==0.9.16 typing-extensions==4.10.0 \
    pylint ipython==8.20 numpy==1.24.4 matplotlib==3.8.4 \
    llvmlite==0.44.0\ 
    numba==0.61.2 pandas==2.2.0 scikit-image==0.22.0 \
    torchpack==0.3.1 wandb==0.18.0 tqdm transformers==4.50.0 \
    open_clip_torch==2.26.1 kaleido==0.2.1 pillow==10.4.0 tensorboardX==2.6.2.2

# ------------------------------------------------------------------------------
# ICP-Flow Dependencies 
# ------------------------------------------------------------------------------
RUN pip install \
    parmap torchist==1.0.0 hdbscan==0.8.41 kiss-icp==0.4.0 nuscenes-devkit==1.2.0\
    # pytorch3d==0.7.4 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu121_pyt231/download.html
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ------------------------------------------------------------------------------
# Create working directory 
# ------------------------------------------------------------------------------
RUN mkdir -p /OV-SCAN
WORKDIR /OV-SCAN

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV JE_ARROW_MALLOC_CONF=background_thread:false