FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y python3.9 python3-pip


# Install vim and other essential tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    curl \
    build-essential \
    # Add other development tools you need
    && rm -rf /var/lib/apt/lists/*
COPY ./dev_config/.vimrc /root/.vimrc



WORKDIR /app
COPY . .

# Add your dependencies
RUN pip install -r requirements.txt


# DEfault target to run container DF - useful?
CMD ["python3", "app.py"]
