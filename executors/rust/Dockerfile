FROM python:3.13.2

RUN apt-get update && \
    apt-get install -y rustc cargo && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workdir

COPY ./workdir-template/* ./

ENTRYPOINT ["python3", "/workdir/test_harness.py"]
