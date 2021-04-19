FROM nvidia-jupyter:latest

USER root

RUN ln -s /usr/local/cuda /usr/local/nvidia

RUN python -m pip install \
     tensorflow \
     tensorflow-datasets \
     jax \
     jaxlib==0.1.64+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN apt-get update && apt install -y git

RUN python -m pip install git+https://github.com/deepmind/dm-haiku
RUN python -m pip install git+git://github.com/deepmind/optax.git

RUN python -m pip install ipywidgets

USER $NB_USER
