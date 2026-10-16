FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_MODEL_SOURCE=BOS
ENV OPENCV_IO_MAX_IMAGE_PIXELS=25000000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# Keep the large CPU Paddle layer reusable when only OCR/API requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# PaddleX requires contrib 4.10. Do not add another OpenCV wheel over cv2.
RUN pip check && python -c "import cv2,importlib.metadata as m; names={'opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless'}; installed=[d.metadata['Name'].lower() for d in m.distributions() if d.metadata['Name'].lower() in names]; assert installed==['opencv-contrib-python'], installed; assert cv2.__version__=='4.10.0', cv2.__version__"

COPY app.py .
COPY tiktok_translate.py .
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
