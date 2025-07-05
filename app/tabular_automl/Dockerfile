FROM app-base as builder

COPY tabular_automl/ ./tabular_automl/

CMD ["uvicorn", "tabular_automl.main:app", "--host", "0.0.0.0", "--port", "8001"]