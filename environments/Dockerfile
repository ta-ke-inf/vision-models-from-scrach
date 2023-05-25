FROM nvidia/cuda:11.7.0-base-ubuntu22.04
ARG WORK_DIR=work
ARG APPLICATION_DIRECTORY=/${WORK_DIR}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"

RUN apt update && apt install -y \
    python3-pip \
    sudo \
    wget \
    git \
    vim \
    libgl1-mesa-dev \
    libgtk2.0-dev \
    libsndfile1
    
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN pip install --upgrade pip
RUN pip install poetry

# If ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true", install Python package by Poetry and move .venv under ${HOME}.
# This process is for CI (GitHub Actions). To prevent overwrite by volume of docker compose, .venv is moved under ${HOME}.
WORKDIR ${APPLICATION_DIRECTORY}
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && poetry install || echo "skip to run poetry install."
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."