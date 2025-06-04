FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04
#start with root to install packages
#setup declaration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CARGO_TARGET_DIR=router/

#setup declaration
ENV DEBIAN_FRONTEND=noninteractive
ENV INSTALL_APT="apt install -y"
ENV INSTALL_PIP="python3 -m pip --no-cache-dir install --upgrade --timeout=100"
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PATH="$PATH:/home/aiserver/.local/bin"

#start
USER root
RUN apt -y update && apt -y upgrade
#install dependencies
RUN ${INSTALL_APT} software-properties-common \
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

# install rust specific app
RUN ${INSTALL_APT} cargo

#install ai, api and ml packages
RUN ${INSTALL_PIP} transformers \
    huggingface_hub[cli] \
    diffusers \
    accelerate \
    fastapi \
    gunicorn \
    uvicorn \
    peft \
    python-dotenv \
    sentencepiece \
    protobuf \
    bitsandbytes \
    torch \
    runpod
#install rust
RUN curl -y --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

#setup user
RUN adduser --uid 1999 aiserver
WORKDIR /app
COPY . /app

#install tei server 
RUN cargo install --path router -F candle-cuda-turing -F http --no-default-features

#fix permission
RUN chown -R aiserver:aiserver /app

RUN chmod +x starter.sh
USER aiserver
EXPOSE 4000

CMD ["bash", "starter.sh"]