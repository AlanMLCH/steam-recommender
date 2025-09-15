FROM python:3.11-slim
WORKDIR /app
COPY ../../models/user_tower.py /app/models/user_tower.py
COPY ../../scripts/build_index.py /app/scripts/build_index.py
COPY app.py /app/app.py
RUN pip install fastapi uvicorn numpy faiss-cpu torch
ENV ARTIFACTS_DIR=/workspace/artifacts INDEX_DIR=/workspace/index
VOLUME ["/workspace/artifacts","/workspace/index","/workspace/data"]
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]