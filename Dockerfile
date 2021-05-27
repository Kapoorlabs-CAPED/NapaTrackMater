FROM python:3.9-slim-buster

# create a user instead of using root user
RUN useradd -mls /bin/bash worker

# set some python/pip env vars
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  PYTHONPATH=/home/worker/lib/python

WORKDIR /usr/src/app
RUN chown worker:worker .
USER worker

COPY requirements.txt .
RUN pip install -vvv --no-cache-dir --prefer-binary --user -r requirements.txt
COPY --chown=worker:worker . .
RUN python setup.py develop --user
CMD ["track", "-h"]
