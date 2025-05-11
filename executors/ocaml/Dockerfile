FROM docker.io/ocaml/opam:ubuntu-22.04-ocaml-5.0

RUN sudo apt-get update && sudo apt-get install -y python3 python3-pip
RUN pip install bounded_subprocess==2.3.1

COPY ./workdir-template/* /workdir/

RUN opam install base stdio utop

ENV OPAM_SWITCH_PREFIX='/home/opam/.opam/5.0'
ENV CAML_LD_LIBRARY_PATH='/home/opam/.opam/5.0/lib/stublibs:/home/opam/.opam/5.0/lib/ocaml/stublibs:/home/opam/.opam/5.0/lib/ocaml'
ENV OCAML_TOPLEVEL_PATH='/home/opam/.opam/5.0/lib/toplevel'
ENV MANPATH=':/home/opam/.opam/5.0/man'
ENV PATH='/home/opam/.opam/5.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
WORKDIR /workdir
ENTRYPOINT ["python3", "/workdir/test_harness.py"]
