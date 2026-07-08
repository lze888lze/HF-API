FROM python:3.10-slim

WORKDIR /home/user/app

# 性能优化环境变量
# 经过实测，Oracle 免费实例（共享 CPU）上单线程反而更稳定，
# 多线程会加剧资源竞争导致推理时间波动。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# 安装系统依赖（OpenCV 和 OpenBLAS 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用Docker缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

EXPOSE 7860

CMD ["python", "-u", "main.py"]
