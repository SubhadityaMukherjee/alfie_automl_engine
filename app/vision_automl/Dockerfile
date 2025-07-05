FROM app-base as builder

COPY vision_automl/ ./vision_automl/

CMD ["uvicorn", "vision_automl.main:app", "--host", "0.0.0.0", "--port", "8001"]