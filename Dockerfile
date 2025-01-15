# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /flask
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
EXPOSE 3000
CMD ["python3", "main.py"]
