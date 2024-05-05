# 使用官方的 Miniconda 镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /sea_war

# 安装必要的依赖
RUN conda install -y -c pytorch pytorch && \
    conda clean -afy

# 设置容器启动时执行的命令
CMD ["/bin/bash"]

