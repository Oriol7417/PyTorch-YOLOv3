FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /workspace/

# Copy requirements.txts
COPY requirements.txt /workspace

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    # for torchsparse
    libsparsehash-dev \
    # for opencv
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1 \ 
    libgl1 \
    libgomp1 \  
    # general
    python3-pip \
    apt-utils \
    vim \
    git \
    wget

# Install python packages in requirements.txt
RUN pip3 install -r requirements.txt

# for opencv
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so

# start jupyter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
