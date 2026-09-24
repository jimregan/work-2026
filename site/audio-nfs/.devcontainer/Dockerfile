FROM mcr.microsoft.com/devcontainers/rust:1-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    clang \
    nfs-common \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
