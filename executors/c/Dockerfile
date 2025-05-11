FROM python:3.13.2

# Install GCC
RUN apt-get update && apt-get install -y gcc

WORKDIR /workdir
COPY ./workdir-template/* ./
CMD ["python3", "/workdir/test_harness.py"] 
