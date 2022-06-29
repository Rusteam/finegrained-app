FROM nvcr.io/nvidia/tritonserver:22.05-py3

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh .
RUN sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda

RUN /opt/miniconda/bin/conda init bash
RUN /opt/miniconda/bin/conda install conda-pack

RUN /opt/miniconda/bin/conda create -n py39 -y -q python=3.9
RUN exec bash
RUN bash -c /opt/miniconda/bin/conda activate py39
RUN pip install transformers[torch]

RUN /opt/miniconda/bin/conda-pack \
    -p /opt/miniconda/envs/py39 \
    -o backends/python/py39.tar.gz
