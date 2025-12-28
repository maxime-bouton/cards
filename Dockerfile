FROM ghcr.io/prefix-dev/pixi:latest AS builder

WORKDIR /env

# copy only dependency files first to prevent re-installing libraries every time
COPY pyproject.toml pixi.lock* ./

# install dependencies
# --locked ensures the build fails if lockfile doesn't match pyproject.toml
RUN pixi install --locked --manifest-path pyproject.toml

# script containing all environment variables (PATH, PYTHONPATH, LD_LIBRARY_PATH, etc.)
RUN pixi shell-hook > /shell-hook.sh

# copy the rest of the files
COPY . .

FROM debian:bullseye-slim

# install basic system tools
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# copy conda environment and shell hook from the builder stage
COPY --from=builder /env /env
COPY --from=builder /shell-hook.sh /shell-hook.sh

# REQUIRED FOR CUDA
# to tell the container runtime to map the GPU driver into the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# activate the env automatically when the container starts
ENTRYPOINT ["/bin/bash", "/shell-hook.sh", "exec"]

# default command
CMD ["python", "--version"]
