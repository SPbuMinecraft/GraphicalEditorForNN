# syntax=docker/dockerfile:1

FROM node:21-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential
RUN apt install -y libboost-dev
RUN apt-get install -y libcpprest-dev
RUN apt-get install -y libx11-dev libpng-dev libzip-dev

EXPOSE ${CPORT}

WORKDIR /app
COPY . .

ENV CONFIG_PATH=/app/config.json

CMD make build && make test
