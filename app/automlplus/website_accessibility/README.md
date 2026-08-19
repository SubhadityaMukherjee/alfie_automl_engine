## Usage example
- Start the unified engine from the repo root
```
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

```
curl -X POST http://localhost:8001/automl/automl_plus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
```
