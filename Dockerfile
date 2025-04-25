FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="allocator:disabled,cache:0,deterministic:1,allow_growth:1,expandable_segments:True"


RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . /app
RUN mkdir -p models/dlib models/ip-adapter
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install gdown

RUN gdown --id 1WdwqdrivpYtOxUVYKrR3hV9E_gYH0cu_ -O models/dlib/shape_predictor_68_face_landmarks.dat && \
    gdown --id 1SDvNYNrlEj1oqfuRq5ScOLK4rXC_43aq -O models/ip-adapter/ip-adapter-plus-face_sd15.pth

EXPOSE 7860

CMD ["python", "app.py"]