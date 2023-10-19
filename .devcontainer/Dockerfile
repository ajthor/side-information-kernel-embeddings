ARG PYTHON_VERSION=1-3.10-bookworm
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION} AS numpy

ARG NUMPY_VERSION=1.25.1

# Install numpy using pip.
RUN pip install --no-cache-dir numpy==${NUMPY_VERSION}



# Second build stage: install jaxlib
FROM numpy AS builder 

# Install python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the jax repo, checkout the version we want, and install jaxlib.
ARG JAX_VERSION=0.4.13
RUN git clone https://github.com/google/jax.git /tmp/jax \
    && cd /tmp/jax \
    && git checkout tags/jaxlib-v${JAX_VERSION} \
    && python build/build.py



# Third build stage: install jax
FROM numpy

# Install build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the wheel from the builder stage.
COPY --from=builder /tmp/jax/dist/*.whl /tmp/jax/dist/

# Install jax and jaxlib using pip.
ARG JAX_VERSION=0.4.13
RUN pip install /tmp/jax/dist/*.whl --force-reinstall \
    && pip install git+https://github.com/google/jax@jax-v${JAX_VERSION} \
    && rm -rf /tmp/jax