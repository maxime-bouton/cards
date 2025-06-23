FROM ghcr.io/prefix-dev/pixi:latest AS builder

WORKDIR /env

COPY . .

RUN pixi install --manifest-path pyproject.toml

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y curl ca-certificates && apt-get clean
COPY --from=builder /env /env

ENV PATH="/env/.pixi/envs/default/bin:$PATH"
