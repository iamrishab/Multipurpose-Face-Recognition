FROM nvidia/cuda:10.2-devel-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         git \
         cmake \
         gcc \
         g++ \
         python3-pip \
         python3-dev \
         python3-setuptools \
         python3-yaml \
         libsm6 \
         libglib2.0-0 \
         libxrender-dev \
         libxext6 \
         libgl1-mesa-glx \
         sudo \
         wget && \
 rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install virtualenv

COPY ./* /home/facialrecognitionv1/

RUN virtualenv /home/fr_env
RUN /home/fr_env/bin/pip install -r /home/facialrecognitionv1/requirements.txt

EXPOSE 8001
EXPOSE 8002

WORKDIR /home/facialrecognitionv1
ENTRYPOINT ["sh", "main.sh"]