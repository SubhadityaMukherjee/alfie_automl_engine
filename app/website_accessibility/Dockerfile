FROM app-base as builder

COPY website_accessibility/ ./website_accessibility/

CMD ["uvicorn", "website_accessibility.main:app", "--host", "0.0.0.0", "--port", "8000"]