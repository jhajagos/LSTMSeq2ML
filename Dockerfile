ARG TF_VERSION="2.2.0"
FROM tensorflow/tensorflow:${TF_VERSION}-jupyter
WORKDIR /opt/seq2ml
COPY . .
RUN python -m pip install --no-cache-dir .
ENTRYPOINT ["seq2ml"]
CMD ["--help"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
