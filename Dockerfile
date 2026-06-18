FROM python:3.10-slim

WORKDIR /home/user/app

# 安装系统依赖（OpenCV需要）
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用Docker缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

EXPOSE 7860

CMD ["python", "-u", "main.py"]
