FROM pytorch/torchserve:0.11.0-cpu
USER root
WORKDIR /home/model-server
COPY handler.py /home/model-server/handler.py
COPY /home/model-server /home/model-server
ENV ARTIFACTS_DIR=/workspace/artifacts
RUN pip install numpy
# Create a trivial MAR at build time
RUN torch-model-archiver --model-name rerank --version 1.0 --handler handler.py --extra-files /dev/null --export-path model-store
EXPOSE 8080 8081
CMD ["torchserve","--start","--model-store","model-store","--models","rerank=rerank.mar","--ts-config","/home/model-server/config.properties","--ncs"]