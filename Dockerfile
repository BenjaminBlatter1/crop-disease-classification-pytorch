# ---------------------------------------------------------
# 1) Base image: CUDA + cuDNN + Ubuntu (GPU-enabled)
# ---------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---------------------------------------------------------
# 2) System dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---------------------------------------------------------
# 3) Working directory inside the container
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# 4) Copy minimal inference dependencies
# ---------------------------------------------------------
COPY requirements.inference.txt .

# ---------------------------------------------------------
# 5) Install Python dependencies (except PyTorch)
# ---------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.inference.txt

# ---------------------------------------------------------
# 6) Install CUDA-enabled PyTorch (GPU version)
# ---------------------------------------------------------
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------
# 7) Copy your project into the container
# ---------------------------------------------------------
COPY src/ /app/src/
COPY src/api/server.py /app/src/api/server.py
COPY weights/ /app/weights/

# ---------------------------------------------------------
# 8) Expose FastAPI port
# ---------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------
# 9) Environment variables
# ---------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="resnet34"
ENV DEVICE="cuda"

# ---------------------------------------------------------
# 10) Entrypoint: start FastAPI with Uvicorn
# ---------------------------------------------------------
ENTRYPOINT ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
