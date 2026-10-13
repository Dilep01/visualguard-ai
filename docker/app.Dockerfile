FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VG_PROJECT_ROOT=/app TEMP=/app/.cache/tmp TMP=/app/.cache/tmp
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN mkdir -p .cache/tmp runtime models && pip install --no-cache-dir .
CMD ["uvicorn", "visualguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
