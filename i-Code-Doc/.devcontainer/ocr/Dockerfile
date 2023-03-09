FROM python:3.7

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends wget ffmpeg libsm6 libxext6

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
RUN rm -rf /tmp/pip-tmp

# Copy flask_api.py and utils.py to /workspace directory
COPY ./server_cpu/flask_api.py /workspace/
COPY ./server_cpu/utils.py /workspace/
WORKDIR /workspace
expose 5000