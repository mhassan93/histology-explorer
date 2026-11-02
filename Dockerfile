FROM python:3.13.14-slim@sha256:9662417aace5ae7b8e2609cce472b72a8958e134ba372808abe9cc1a0c0125e6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY LICENSE pyproject.toml ./
COPY src ./src
COPY third_party ./third_party
COPY scripts ./scripts
RUN pip install --no-deps .

COPY data/samples ./data/samples

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 7860
CMD ["python", "scripts/run_app.py", "--host", "0.0.0.0", "--port", "7860"]
