FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y python3.9 python3-pip

# Run X11 For running matplotlib Files window
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6

# Install vim and other essential tools
# DEBIAN FRONTEND added because of hanging installation prompts, so we want to add install 
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim \
    git \
    curl \
    build-essential \
    vim-gtk3 \
    # Add other development tools you need
    && rm -rf /var/lib/apt/lists/*


COPY ./dev_config/.vimrc /root/.vimrc


WORKDIR /app
COPY . .

# Add your dependencies
RUN pip install -r requirements.txt


# DEfault target to run container DF - useful?
CMD ["python3", "app.py"]
