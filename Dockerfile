# 使用Python 3.8基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt
RUN openai migrate


# 复制应用代码
COPY . .

# 设置环境变量
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# 暴露端口
EXPOSE 80

# 启动命令
CMD ["python", "app.py"] 