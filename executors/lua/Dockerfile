FROM nickblah/luajit:2.1-oracular

RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3

WORKDIR /workdir
COPY ./workdir-template/* ./
CMD ["python3", "/workdir/test_harness.py"]
