FROM ubuntu:latest
LABEL authors="samueleghezzi"

ENTRYPOINT ["top", "-b"]