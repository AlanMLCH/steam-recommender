FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py
RUN pip install requests pyyaml
CMD ["python","/app/app.py"]