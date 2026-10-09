FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    verilator g++ make ca-certificates git cmake python3 g++-arm-linux-gnueabihf \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
