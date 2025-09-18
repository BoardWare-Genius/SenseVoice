FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get -y install \
    libopus-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Create app directory
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "./api.py" ]