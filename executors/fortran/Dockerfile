FROM debian:stable-slim

RUN apt-get update && \
    apt-get install -y gfortran python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages bounded_subprocess==2.3.1

WORKDIR /workdir
COPY ./workdir-template/* ./

ENTRYPOINT ["python3", "/workdir/test_harness.py"]
