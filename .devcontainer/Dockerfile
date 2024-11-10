# 使用一个基础镜像（这里使用 Python 3.8 作为示例）
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制所有项目文件到 /app
COPY . /app

# 安装项目依赖（如果有的话）
RUN pip install -r requirements.txt

# 设置容器启动命令
CMD ["bash"]

# 启动一个后台进程来持续运行，示例中监视 /dev/null
# CMD ["tail", "-f", "/dev/null"]