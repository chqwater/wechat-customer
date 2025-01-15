# 使用Python 3.8基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV GPT_ENGINE=text-davinci-003

# 暴露端口
EXPOSE 80

# 启动命令
CMD ["python", "app.py"] 