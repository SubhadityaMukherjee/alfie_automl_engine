# Docker instructions

## Running everything

- Simply do `docker compose up` in the main folder (assuming you have docker installed)
- This starts a single combined service that mounts every AutoML service under one unified router (`/automl/tabular`, `/automl/vision`, `/automl/automl_plus`)
- After that you can `curl` any of the endpoints you want
- For information on the ports, please look at your .env file (`AUTOML_ENGINE_PORT`, default `8001`)
- eg: Tabular AutoML best-model search

```bash
curl -s -X POST "http://localhost:8001/automl/tabular/best_model/" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=1" \
  -F "dataset_id=2" \
  -F "target_column_name=signature" \
  -F "task_type=classification" \
  -F "time_stamp_column_name=" \
  -F "time_budget=30"
```

## Pushing to repo

Login: echo {PASS} | docker login gitlab.catalink.eu:5050 -u {USER} --password-stdin
Build & tag: docker build -t gitlab.catalink.eu:5050/external/alfie_eu/alfie/{MODULE}:{TAG}
Push: docker push gitlab.catalink.eu:5050/external/alfie_eu/alfie/{MODULE}:{TAG}
