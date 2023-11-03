FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN mkdir /app
WORKDIR /app

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
  curl \
  git \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="7.5"
RUN nvidia-smi

COPY download_pretrained.py /app/download_pretrained.py
RUN pip install gdown && python download_pretrained.py

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app/

ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
