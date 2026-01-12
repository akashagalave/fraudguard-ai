FROM seldonio/mlserver:1.3.5-sklearn

WORKDIR /app

COPY model /app/model

ENV MLSERVER_MODEL_NAME=classifier
ENV MLSERVER_MODEL_IMPLEMENTATION=mlserver_sklearn.SKLearnModel
ENV MLSERVER_MODEL_URI=/app/model
ENV MLSERVER_HTTP_PORT=9000
ENV MLSERVER_GRPC_PORT=9500

EXPOSE 9000 9500

CMD ["mlserver", "start", "."]