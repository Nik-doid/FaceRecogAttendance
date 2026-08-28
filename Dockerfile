# Build: install deps into a venv with uv (frozen against committed uv.lock).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# insightface 0.7.3 publishes no wheels; uv builds it from the sdist, which compiles
# the face3d Cython extension and therefore needs a C++ toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra ai

# Runtime: copy only what the service needs.
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
# mediapipe pulls in opencv-contrib-python (not the headless build), which dlopens
# libGL/libglib at import time; faiss-cpu links against libgomp. mediapipe's own
# libmediapipe.so additionally dlopens libEGL/libGLESv2 when a Task (HandLandmarker)
# is created -- missing those fails at startup, not at import, so it survives any
# import-only smoke test.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY app ./app
COPY frontend ./frontend
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
