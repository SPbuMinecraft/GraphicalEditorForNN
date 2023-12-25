# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 node:21-alpine3.18

RUN apk add --update --no-cache make
RUN apk add --no-cache python3 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE ${PYPORT}

WORKDIR /app
COPY . .

ENV CONFIG_PATH=/app/config.json

CMD make build && make test
