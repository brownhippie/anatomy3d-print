FROM python:3.11-slim

# MediaPipe's native library links against a full GL/EGL stack even for
# CPU-only inference. Found by actually running pose detection in this
# environment, not by reading docs: libgl1 alone isn't enough, it also
# needs libGLESv2 and libEGL, which come from separate packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-trellis.txt ./
# requirements-trellis.txt (which itself pulls in requirements.txt) installed
# here by default -- unlike requirements-depth.txt/requirements-smplx.txt,
# which stay opt-in: the webapp exposes TRELLIS.2 as a regular method option
# in its UI, not an explicit opt-in flag the way body_fit.py's SMPL-X mode
# is, so its dependency needs to actually be present for that option to do
# anything other than silently fall back every time. It's lightweight (an
# API client, no torch/GPU requirement of its own -- see that file's own
# comment), so this doesn't meaningfully grow the image.
RUN pip install --no-cache-dir -r requirements-trellis.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY webapp ./webapp
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
CMD ["./entrypoint.sh"]
