FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
# SQLite database and charts persisted outside the container
VOLUME ["/app/data", "/app/outputs"]
ENTRYPOINT ["eanowcast"]
CMD ["nowcast", "--save"]
