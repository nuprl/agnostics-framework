FROM docker.io/r-base:4.5.0

RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3

RUN apt-get install -yqq r-cran-tidyverse

WORKDIR /workdir
COPY ./workdir-template/* ./
CMD ["python3", "/workdir/test_harness.py"]
