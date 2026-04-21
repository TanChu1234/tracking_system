# ============================================================
# Multi-Camera Tracking (MCT) System — Dockerfile
# ============================================================
#
#   GPU Stack (mirrors "reid" conda env):
#   ┌──────────────────────────────────────────────────┐
#   │  NVIDIA Driver ≥ 520    (host, via container-tk) │
#   │  CUDA Toolkit    11.8   (base image)             │
#   │  cuDNN            8.9   (base image)             │
#   │  PyTorch    2.7.1+cu118 (pip, bundles nvidia-*)  │
#   │  ORT-GPU      1.16.0   (pip, needs system CUDA) │
#   │  faiss-gpu      1.7.2  (pip)                     │
#   │  triton         3.3.1  (pip, for torch.compile)  │
#   └──────────────────────────────────────────────────┘
#
#   IMPORTANT — onnxruntime-gpu 1.16.0 links against
#   system CUDA 11.8 libs at runtime:
#     • libcudart.so.11    → from base image
#     • libcudnn.so.8      → from cudnn8 base image
#     • libcublas.so.11    → need libcublas-11-8 package
#     • libcublasLt.so.11  → need libcublas-11-8 package
#     • libcurand.so.10    → need libcurand-11-8 package
#     • libcufft.so.10     → need libcufft-11-8 package
#   PyTorch 2.7.1+cu118 ships its own CUDA libs via
#   nvidia-*-cu11 pip wheels, so it does NOT rely on
#   system CUDA (except the driver).
#
# Build:
#   docker build -t mct-system .
#
# Run:
#   docker run --gpus all --rm -it \
#     -p 8068:8068 \
#     -e DB_HOST=10.29.8.49 \
#     mct-system
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
# Use devel image so we have headers + libs for compiling
# packages that link against CUDA (psycopg2, etc.)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential libpq-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Virtual env for clean copy to runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# ── Layer 1 (heaviest): PyTorch + CUDA 11.8 ────────────────
# This pulls nvidia-cublas-cu11, nvidia-cudnn-cu11, nvidia-nccl-cu11,
# nvidia-cuda-runtime-cu11, etc. as pip dependencies automatically.
RUN pip install --no-cache-dir \
        torch==2.7.1+cu118 \
        torchvision==0.22.1+cu118 \
        torchaudio==2.7.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ── Layer 2: GPU-accelerated ML libs ───────────────────────
# onnxruntime-gpu 1.16.0 — requires CUDA 11.8 system libraries
# at runtime (libcublas, libcufft, libcurand, libcudnn.so.8).
# These are provided by the base image + extra apt packages
# in the runtime stage.
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.16.0

# faiss-gpu 1.7.2 — pip wheel bundles its own GPU kernels
RUN pip install --no-cache-dir \
        faiss-gpu==1.7.2

# triton (used by torch.compile for GPU kernel fusion)
RUN pip install --no-cache-dir \
        triton==3.3.1

# ── Layer 3: Computer Vision / ML ──────────────────────────
RUN pip install --no-cache-dir \
        ultralytics==8.3.167 \
        timm==1.0.24 \
        opencv-contrib-python==4.10.0.84 \
        opencv-python-headless==4.10.0.84 \
        scikit-image==0.24.0 \
        scikit-learn==1.7.2 \
        numpy==1.24.4 \
        scipy==1.15.3 \
        pillow==12.1.0

# ── Layer 4: Web / API ─────────────────────────────────────
RUN pip install --no-cache-dir \
        fastapi==0.122.0 \
        uvicorn==0.38.0 \
        uvloop==0.22.1 \
        httptools==0.7.1 \
        websockets==16.0 \
        flask==3.0.3 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        starlette==0.50.0

# ── Layer 5: Database ──────────────────────────────────────
RUN pip install --no-cache-dir \
        psycopg2-binary==2.9.11 \
        SQLAlchemy==2.0.45 \
        pymongo==4.10.1

# ── Layer 6: Utilities ─────────────────────────────────────
RUN pip install --no-cache-dir \
        PyYAML==6.0.3 \
        tqdm==4.67.1 \
        matplotlib==3.10.7 \
        pandas==2.0.3 \
        requests==2.32.3 \
        yacs==0.1.8 \
        coloredlogs==15.0.1 \
        python-dotenv==1.2.1 \
        openpyxl==3.1.5 \
        kafka==1.3.5 \
        safetensors==0.7.0 \
        huggingface_hub==1.3.2 \
        psutil==7.0.0 \
        pydantic==2.12.4 \
        shapely==2.1.1 \
        networkx==2.8.8


# ── Stage 2: Runtime ───────────────────────────────────────
# Use the CUDA 11.8 + cuDNN 8 runtime image (smaller than devel)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# ── System dependencies ────────────────────────────────────
# NOTE: onnxruntime-gpu 1.16.0 dynamically links against
# system CUDA math libraries that are NOT included in the
# "runtime" base image. We must install them explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv \
        # ----- OpenCV runtime deps -----
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
        # ----- OpenMP (faiss, numpy, scipy) -----
        libgomp1 \
        # ----- PostgreSQL client lib (psycopg2) -----
        libpq5 \
        # ━━━━━ CUDA math libs required by onnxruntime-gpu 1.16.0 ━━━━━
        # These provide libcublas.so.11, libcublasLt.so.11,
        # libcurand.so.10, libcufft.so.10 that ORT links at runtime.
        libcublas-11-8 \
        libcurand-11-8 \
        libcufft-11-8 \
        libcusparse-11-8 \
        libcusolver-11-8 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Copy pre-built venv from builder ───────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Ensure pip nvidia-* CUDA libs are on LD_LIBRARY_PATH
# (PyTorch resolves these via its own loader, but other libs
# like triton may need them visible at the system level)
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.10/site-packages/nvidia/cublas/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cufft/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/curand/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cusolver/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/cusparse/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/nccl/lib:\
/opt/venv/lib/python3.10/site-packages/nvidia/nvtx/lib:\
${LD_LIBRARY_PATH}"

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── Copy project source ───────────────────────────────────
# .dockerignore excludes .git, __pycache__, logs/, docs/, etc.
COPY . /app/

# ── Environment variables ─────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NVIDIA Container Toolkit — expose all GPU capabilities
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Database connection (override at runtime via docker run -e)
ENV DB_HOST=10.29.8.49
ENV DB_PORT=5432
ENV DB_USER=infiniq_user
ENV DB_PASSWORD=infiniq_pass
ENV DB_NAME=camera_ai_db

# ── Health-check ───────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD python -c "import requests; r=requests.get('http://localhost:8068/'); exit(0 if r.ok else 1)" \
    || exit 1

# ── Expose ports ───────────────────────────────────────────
# FastAPI / WebSocket API server
EXPOSE 8068

# ── Default command ────────────────────────────────────────
CMD ["python", "demo_mct.py", "--use_db"]
