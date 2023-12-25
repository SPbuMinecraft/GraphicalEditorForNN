# syntax=docker/dockerfile:1

FROM node:21-alpine3.18

RUN apk add --no-cache make
RUN apk add --update --no-cache python3 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE ${PYPORT}

WORKDIR /app
COPY . .

ENV CONFIG_PATH=/app/config.json

CMD make build && make test
