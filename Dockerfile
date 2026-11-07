FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=tensorflow \
    HOST=0.0.0.0 \
    PORT=7860 \
    HOME=/home/user \
    KAGGLEHUB_CACHE=/home/user/.cache/kagglehub

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["python", "app/server.py"]
