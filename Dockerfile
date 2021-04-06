FROM nvidia/cuda:10.1-cudnn7-devel

WORKDIR /app

RUN apt update

RUN apt install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
     numpy \
     jax \
     jaxlib==0.1.64+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN ln -s /usr/local/cuda /usr/local/nvidia