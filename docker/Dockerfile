# 使用 NVIDIA 官方的 CUDA 镜像作为基础镜像，包含 CUDA 和 cuDNN 支持
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /workspace

# 更新包列表并安装必要的依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    cmake \
    vim \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 安装 pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 创建符号链接以便使用 python 和 pip 命令
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip

# 安装 PyTorch 和相关库
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装强化学习相关的库
RUN pip install --no-cache-dir \
    gym \
    stable-baselines3 \
    sb3-contrib \
    gymnasium \
    numpy \
    matplotlib \
    pandas \
    jupyterlab

# 安装其他可能需要的库（根据需求调整）
RUN pip install --no-cache-dir \
    tensorboard \
    tqdm \
    seaborn

# 设置 Jupyter Notebook 密码（可选）
RUN python -c "from notebook.auth import passwd; print(passwd('your_password'))" > /root/.jupyter/jupyter_notebook_config.json

# 暴露 Jupyter Notebook 端口
EXPOSE 8888

# 启动 Jupyter Lab
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser"]