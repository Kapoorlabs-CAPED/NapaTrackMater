FROM python:3.9-alpine

# create a user instead of using root user
RUN adduser -D worker

# set some python/pip env vars
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

RUN apk add --no-cache python3-dev build-base cmake openssl-dev linux-headers lapack-dev openblas-dev
RUN pip install --upgrade pip setuptools wheel
USER worker
WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --user -r requirements.txt
COPY . .
RUN python setup.py develop
CMD ["track", "-h"]
