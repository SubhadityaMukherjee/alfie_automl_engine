# Docker instructions

## Running everything
- Simply do `docker compose up` in the main folder (assuming you have docker installed)
- After that you can `curl` any of the services you want
- eg: Website Accessibility (HTML file input)

```bash
curl -sN -X POST http://localhost:8000/automlplus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
```

## Pushing to repo

Login: echo {PASS}  | docker login gitlab.catalink.eu:5050 -u {USER} --password-stdin
Build & tag: docker build -t gitlab.catalink.eu:5050/external/alfie_eu/alfie/{MODULE}:{TAG}
Push: docker push gitlab.catalink.eu:5050/external/alfie_eu/alfie/{MODULE}:{TAG}

