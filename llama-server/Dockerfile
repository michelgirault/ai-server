FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04
#start with root to install packages
#setup declaration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV GGML_CUDA=on

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive
ENV INSTALL_APT="apt install -y"
ENV INSTALL_PIP="python3 -m pip --no-cache-dir install --upgrade"
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PATH="$PATH:/home/aiserver/.local/bin"

#start
USER root
RUN apt -y update && apt -y upgrade
#install dependencies
RUN ${INSTALL_APT} \
    libz-dev \
    libssl-dev \
    libcurl4-gnutls-dev \
    libexpat1-dev \
    gettext \
    cmake \
    gcc \
    bc \
    wget \
    ffmpeg \
    linux-headers-generic \
    libopenblas-dev \
    liblapack-dev \
    libegl1 \
    libglvnd-dev \
    pkg-config \
    nvidia-cuda-toolkit \
    pipenv \
    git \
    make \
    curl \
    g++

#install python and additional lib
RUN ${INSTALL_APT} \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

RUN ${INSTALL_PIP} fastapi \
    pytest \
    scikit-build \
    setuptools \
    uvicorn \
    sse-starlette \
    pydantic-settings \
    starlette-context

#setup user
RUN adduser --uid 1999 aiserver
WORKDIR /app
COPY . /app

#install server
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

#fix permission
RUN chown -R aiserver:aiserver /app

RUN chmod +x starter.sh
USER aiserver
EXPOSE 4000

CMD ["bash", "starter.sh"]