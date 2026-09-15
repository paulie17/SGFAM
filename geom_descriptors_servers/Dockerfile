# ==================== GeDi Stage ====================
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04 AS gedi

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/fabiopoiesi/gedi.git /gedi

WORKDIR /gedi

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install open3d==0.15.2 torchgeometry==0.1.2 flask gdown tensorboard protobuf==3.20.3

WORKDIR /gedi/backbones/pointnet2_ops_lib
ENV ForceCUDA="1"
RUN pip3 install .

WORKDIR /gedi
RUN python3 download_data.py

CMD ["/bin/bash"]

FROM gedi AS gedi_server

COPY gedi_server.py /gedi/gedi_server.py

CMD ["python3", "gedi_server.py"]

# ==================== FCGF Stage (WarpConvNet Backend) ====================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS fcgf

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install https://github.com/NVlabs/WarpConvNet/releases/download/v1.8.2/warpconvnet-1.8.2%2Btorch2.5cu124-cp310-cp310-linux_x86_64.whl
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
RUN pip3 install scipy plyfile open3d flask requests easydict

RUN git clone https://github.com/chrischoy/FCGF.git /FCGF

WORKDIR /FCGF

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true

RUN wget https://huggingface.co/chrischoy/FCGF/resolve/main/2019-08-19_06-17-41.pth -O fcgf-3dmatch.pth
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} python3 wcn/convert_me_to_wcn.py --me_ckpt fcgf-3dmatch.pth --out fcgf-3dmatch-wcn.pth

CMD ["/bin/bash"]

FROM fcgf AS fcgf_server

COPY fcgf_server.py /FCGF/fcgf_server.py

CMD ["python3", "fcgf_server.py", "--checkpoint", "fcgf-3dmatch-wcn.pth", "--port", "8000"]
