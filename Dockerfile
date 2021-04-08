FROM nvidia-jupyter:latest

USER root

RUN ln -s /usr/local/cuda /usr/local/nvidia

RUN python -m pip install \
     numpy \
     jax \
     jaxlib==0.1.64+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

USER $NB_USER
