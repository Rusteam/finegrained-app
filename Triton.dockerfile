FROM --platform=linux/amd64 nvcr.io/nvidia/tritonserver:22.09-py3

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh .
RUN sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

RUN /opt/miniconda/bin/conda init bash
RUN /opt/miniconda/bin/conda install conda-pack

RUN /opt/miniconda/bin/conda create -n py310 -y -q python=3.10
RUN exec bash

RUN bash -c /opt/miniconda/bin/conda activate py310
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers

RUN /opt/miniconda/bin/conda-pack \
    -p /opt/miniconda/envs/py310 \
    -o backends/python/py310.tar.gz
