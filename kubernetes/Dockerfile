FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update -y
RUN apt-get install -y python3-dev python3-pip python3-venv git
RUN pip3 install --upgrade pip
# Atari dependencies
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED 1