# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster

WORKDIR /Framework-Proyecto

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt


COPY . .

CMD ["python3", "orq_descriptivo\Framework-Proyecto\orq_descriptivo\__init__.py"]