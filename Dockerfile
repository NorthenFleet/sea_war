# 使用官方的 Ubuntu 基础镜像
FROM ubuntu:20.04

# 设置环境变量，避免在安装过程中出现交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新包列表并安装必要的依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda update -n base -c defaults conda -y && \
    conda clean --all --yes

# 创建 Conda 环境，包含 Python 3.10
RUN conda create -y -n myenv python=3.10

# 激活环境并安装项目中的 requirements.txt 文件中的依赖
COPY requirements.txt /workspace/
RUN /bin/bash -c "source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate myenv && \
    pip install -r /workspace/requirements.txt"

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH=$CONDA_DIR/envs/myenv/bin:$PATH

# 设置默认命令为 bash，并自动激活 Conda 环境
CMD ["bash", "-c", "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate myenv && exec bash"]