## Usage example
- From root
```
uv run fastapi dev app/website_accessibility/main.py
```

```
curl -X POST http://localhost:8000/web_access/accessibility/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
```