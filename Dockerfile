FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_PORT=8000 \
    FRONTEND_PORT=8501 \
    BACKEND_API_URL=http://127.0.0.1:8000/api/v1

WORKDIR /app

# psycopg2 is retained as a project dependency; build support keeps installation
# reproducible on the slim base image.
RUN apt-get update && apt-get install --yes --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "main.py", "serve"]
