FROM mcr.microsoft.com/devcontainers/miniconda:0-3

ARG UNAME=vscode
ARG USER_ID=1004
ARG GROUP_ID=1004

RUN groupmod -g $GROUP_ID $UNAME
RUN usermod -u $USER_ID -g $GROUP_ID $UNAME

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends poppler-utils ffmpeg libsm6 libxext6

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
COPY requirements.txt /tmp/pip-tmp/
RUN conda install -y python=3.8 \
    && pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt 
RUN conda install ipykernel --update-deps --force-reinstall
RUN rm -rf /tmp/pip-tmp

ENV HF_HOME="/workspaces/udop/i-Code-Doc/.hf_cache"
ENV HF_DATASETS_CACHE="/workspaces/udop/i-Code-Doc/.hf_cache"
