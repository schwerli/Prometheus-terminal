import uuid
from pathlib import Path

from prometheus.docker.base_container import BaseContainer


class GeneralContainer(BaseContainer):
  def __init__(self, project_path: Path):
    super().__init__(project_path)
    self.tag_name = f"prometheus_general_container_{uuid.uuid4().hex[:10]}"

  def get_dockerfile_content(self) -> Path:
    DOCKERFILE_CONTENT = """\
FROM ubuntu:22.04

# Avoid timezone prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install essential build and development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    nodejs \
    npm \
    default-jdk \
    gcc \
    g++ \
    gdb \
    postgresql-client \
    mysql-client \
    sqlite3 \
    netcat \
    iputils-ping \
    vim \
    nano \
    zip \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy project files
COPY . /app/
"""
    return DOCKERFILE_CONTENT

  def run_build(self):
    raise NotImplementedError(
      "GeneralContainer does not support run_build, use execute_command directly"
    )

  def run_test(self):
    raise NotImplementedError(
      "GeneralContainer does not support run_test, use execute_command directly"
    )
