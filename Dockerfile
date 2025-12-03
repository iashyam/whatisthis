# syntax=docker/dockerfile:1.7

#FROM hogepodge/notebook-pytorch
# FROM anibali/pytorch:2.0.1-nocuda-ubuntu22.04
FROM python:3.10-slim

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu


USER root

COPY requirements.txt .
COPY pyproject.toml ./

# RUN --mount=type=cache,target=/root/.cache/uv
# RUN uv sync

RUN pip install -r requirements.txt

COPY src ./src

RUN rm -rf src/*.egg-info || true

COPY image_recognition-0.1.0-py3-none-any.whl ./
RUN pip install image_recognition-0.1.0-py3-none-any.whl
# RUN pip install .


CMD [ "python" , "src/app/routes.py"]
#CMD ["uv", "run", "src/app/routes.py"]



