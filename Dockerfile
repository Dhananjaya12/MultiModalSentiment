FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt requirements-inference.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app app
COPY inference inference
COPY model model
COPY config.json config.json

EXPOSE 7860

CMD ["python", "app/app.py"]
