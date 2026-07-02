FROM python:3.11-slim

# Build tools needed by ufal.udpipe's C++ extension, and curl for the model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Bake the models into the image so the container runs fully offline ---

# Stanza Irish model.
RUN python -c "import stanza; stanza.download('ga')"

# UDPipe Irish-IDT model (UD 2.5). jwijffels mirrors the individual model files
# from LINDAT as plain GitHub blobs, which are stable to curl directly.
ARG UDPIPE_MODEL_URL=https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/master/inst/udpipe-ud-2.5-191206/irish-idt-ud-2.5-191206.udpipe
RUN mkdir -p /models \
    && curl -fL "$UDPIPE_MODEL_URL" -o /models/irish-idt.udpipe \
    && test -s /models/irish-idt.udpipe

ENV UDPIPE_MODEL=/models/irish-idt.udpipe
ENV PYTHONUNBUFFERED=1
# Fail loudly instead of silently re-downloading Stanza models at runtime.
ENV STANZA_RESOURCES_DIR=/root/stanza_resources

CMD ["bash"]
