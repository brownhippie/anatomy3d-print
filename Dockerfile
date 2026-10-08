FROM python:3.11-slim

# MediaPipe's native library links against a full GL/EGL stack even for
# CPU-only inference. Found by actually running pose detection in this
# environment, not by reading docs: libgl1 alone isn't enough, it also
# needs libGLESv2 and libEGL, which come from separate packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY webapp ./webapp
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
CMD ["./entrypoint.sh"]
