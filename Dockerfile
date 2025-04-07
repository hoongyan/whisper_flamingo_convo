FROM continuumio/miniconda3:4.10.3
WORKDIR /app
RUN apt-get update && apt-get install -y sox git && rm -rf /var/lib/apt/lists/*
# Create conda environment and install ffmpeg via conda
RUN conda create -n whisper-flamingo python=3.8 -y && \
    conda install -n whisper-flamingo -c conda-forge ffmpeg==4.2.2 -y
SHELL ["conda", "run", "-n", "whisper-flamingo", "/bin/bash", "-c"]
COPY requirements.txt .
RUN git clone https://github.com/facebookresearch/muavic.git muavic-setup && \
    cd muavic-setup && pip install -r requirements.txt && cd ..
RUN git clone -b muavic https://github.com/facebookresearch/av_hubert.git && \
    cd av_hubert && git submodule init && git submodule update && \
    pip install -r updated_requirements.txt && cd fairseq && pip install --editable ./ && cd ../..
RUN pip install -r requirements.txt
RUN pip install pip==24.0
COPY . .
EXPOSE 8000
CMD ["conda", "run", "-n", "whisper-flamingo", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]