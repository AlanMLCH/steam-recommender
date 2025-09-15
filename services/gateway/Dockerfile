FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY ../../scripts/metrics.py /app/scripts/metrics.py
COPY app.py /app/app.py
RUN pip install fastapi uvicorn httpx prometheus_client numpy
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]