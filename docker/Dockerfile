FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pipelines/ pipelines/
COPY configs/ configs/
COPY models/ models/

EXPOSE 7860

CMD ["python", "-m", "src.serving.app"]
