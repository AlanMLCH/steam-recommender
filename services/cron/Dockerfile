FROM python:3.11-slim
WORKDIR /workspace
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*
COPY ../../scripts/build_index.py /workspace/scripts/build_index.py
COPY ../../scripts/psi.py /workspace/scripts/psi.py
COPY crontab /etc/cron.d/reindex
COPY entrypoint.sh /entrypoint.sh
RUN pip install numpy faiss-cpu pandas
RUN chmod 0644 /etc/cron.d/reindex && crontab /etc/cron.d/reindex && chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]