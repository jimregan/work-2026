FROM mcr.microsoft.com/devcontainers/miniconda:3

COPY .devcontainer/environment.yml /tmp/dysfluent-wfst-environment.yml

RUN conda env update -n base -f /tmp/dysfluent-wfst-environment.yml \
    && conda clean -afy
