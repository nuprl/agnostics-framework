FROM julia:1.11.3

RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3

WORKDIR /workdir
COPY ./workdir-template/* ./
CMD ["python3", "/workdir/test_harness.py"]
