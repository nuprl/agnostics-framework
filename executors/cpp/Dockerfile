FROM python:3.13.2

RUN apt-get update && apt-get install -y clang

WORKDIR /workdir
COPY ./workdir-template/* ./
CMD ["python3", "/workdir/test_harness.py"]
