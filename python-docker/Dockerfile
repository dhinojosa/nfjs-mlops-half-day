# Use Python slim as the base image
FROM python:3.10-slim

# Install dependencies for pyenv and Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Configure pyenv environment variables
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Configure shell initialization for pyenv
RUN echo 'export PYENV_ROOT="/root/.pyenv"' >> ~/.bashrc \
    && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc \
    && echo 'eval "$(pyenv init --path)"' >> ~/.bashrc \
    && echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install virtualenv globally via pip
RUN pip install virtualenv

# Install mlflow globally via pip
RUN pip install mlflow==2.9.2

# Verify pyenv, virtualenv, and mlflow installations
RUN pyenv --version && \
    virtualenv --version && \
    mlflow --version

# Set default shell to bash
CMD ["/bin/bash"]