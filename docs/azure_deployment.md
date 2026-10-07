# Azure deployment

The engine runs as a Docker container on an Azure VM. This page documents
how the deployment works, how it was set up, and how to manage access to it.

## Current deployment

| Item | Value |
| --- | --- |
| VM | `automl` (Ubuntu 24.04, 2 vCPU / 15 GB RAM, West Europe) |
| Public IP | `134.149.217.112` |
| Port | `8001` (`AUTOML_ENGINE_PORT`) |
| Subscription | `MCS-ISOLATED-ALFIE-DEV` |
| Resource group | `ALFIE_AutoML_Deployment` |
| Network security group | `automl-nsg` (attached to the VM's NIC) |
| Repo on VM | `~/alfie_automl_engine` (branch `main`) |
| Container | `alfie_automl_engine` via `docker compose`, `restart: always` |

The API is reachable at `http://134.149.217.112:8001` — but **only from IP
addresses explicitly allowed** in the NSG rule (see
[Managing access](#managing-access)). The API itself has no authentication,
so the firewall is the only protection; do not open it to `0.0.0.0/0`.

SSH access:

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112
```

## Step-by-step: initial deployment

Run from your local machine (needs `az` CLI, `ssh`, and Docker on the VM).

### 1. Log in to Azure

```bash
az login
az account set --subscription "MCS-ISOLATED-ALFIE-DEV"
```

!!! note
    `az login --use-device-code` is blocked by the tenant's conditional
    access policy ("you don't have access to this"). Use the normal
    interactive `az login`, which opens a browser window.

### 2. Get the code and config onto the VM

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "git clone https://github.com/SubhadityaMukherjee/alfie_automl_engine.git ~/alfie_automl_engine"

# copy your local .env (contains AZURE_OPENAI_KEY, MONGODB_URL, ...)
scp -i /path/to/automl_key.pem .env \
  azureuser@134.149.217.112:~/alfie_automl_engine/.env

# make sure the engine port is defined
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "cd ~/alfie_automl_engine && grep -q '^AUTOML_ENGINE_PORT=' .env || echo 'AUTOML_ENGINE_PORT=8001' >> .env"
```

### 3. Disable the dev `--reload` flag

The compose file starts uvicorn with `--reload` (dev-only). Add an override
on the VM so the production container runs without it:

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "cd ~/alfie_automl_engine && printf 'services:\n  automl_engine:\n    command: [\"bash\", \"-lc\", \"uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8001\"]\n' > docker-compose.override.yml"
```

### 4. Build and start

The image is built **on the VM** (~13 GB; several minutes on 2 vCPUs). Run it
in the background so an SSH disconnect doesn't kill the build:

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "cd ~/alfie_automl_engine && nohup docker compose up --build -d > deploy.log 2>&1 &"

# follow progress
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "tail -f ~/alfie_automl_engine/deploy.log"
```

### 5. Verify on the VM

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "curl -s http://localhost:8001/health && curl -s http://localhost:8001/ready"
# -> {"status":"alive"} {"status":"ready"}
```

### 6. Open the firewall for your machine

Everything is now running, but unreachable from outside until the NSG allows
it. Get your own public IP (e.g. `curl -s https://icanhazip.com`), then
create the rule restricted to it:

```bash
az network nsg rule create \
  -g ALFIE_AutoML_Deployment \
  --nsg-name automl-nsg \
  -n Allow-AutoML-8001 \
  --priority 300 \
  --direction Inbound \
  --access Allow \
  --protocol Tcp \
  --destination-port-ranges 8001 \
  --source-address-prefixes <YOUR_IP>
```

Then from your local machine:

```bash
curl -s http://134.149.217.112:8001/health
# -> {"status":"alive"}
```

!!! note
    The VM's subnet has no NSG of its own, so `automl-nsg` on the NIC is the
    only gate that needs a rule.

## Managing access

### Give someone else access

1. Ask them for their **public IP** (they can visit
   [icanhazip.com](https://icanhazip.com) — must be their public IP, not
   their LAN address).
2. Add their IP to the existing rule, keeping all IPs already listed:

   ```bash
   # check which IPs are currently allowed
   az network nsg rule show -g ALFIE_AutoML_Deployment \
     --nsg-name automl-nsg -n Allow-AutoML-8001 \
     --query sourceAddressPrefix -o tsv

   # re-list every IP (existing + new) in the update command
   az network nsg rule update -g ALFIE_AutoML_Deployment \
     --nsg-name automl-nsg -n Allow-AutoML-8001 \
     --source-address-prefixes <EXISTING_IP> <NEW_IP>
   ```

   Alternatively, create a separate rule per person (easier to revoke
   individually):

   ```bash
   az network nsg rule create -g ALFIE_AutoML_Deployment \
     --nsg-name automl-nsg -n Allow-AutoML-8001-<NAME> \
     --priority 310 --direction Inbound --access Allow --protocol Tcp \
     --destination-port-ranges 8001 \
     --source-address-prefixes <NEW_IP>
   ```

3. They verify with `curl -s http://134.149.217.112:8001/health`.

Via the **Azure Portal** instead of CLI: Virtual machines → `automl` →
Networking → `automl-nsg` → *Add inbound security rule* (Port 8001, TCP,
source = their IP).

### Remove someone's access

```bash
# if they have their own rule
az network nsg rule delete -g ALFIE_AutoML_Deployment \
  --nsg-name automl-nsg -n Allow-AutoML-8001-<NAME>

# if they were in the shared rule: re-run the update with their IP omitted
az network nsg rule update -g ALFIE_AutoML_Deployment \
  --nsg-name automl-nsg -n Allow-AutoML-8001 \
  --source-address-prefixes <REMAINING_IP>
```

## Updating the deployment

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "cd ~/alfie_automl_engine && git pull && docker compose up --build -d"
```

To see logs or restart:

```bash
ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "docker logs -f alfie_automl_engine"

ssh -i /path/to/automl_key.pem azureuser@134.149.217.112 \
  "cd ~/alfie_automl_engine && docker compose restart"
```

## Running the pipelines

All examples below run against the deployed service. Set the base URL once:

```bash
BASE=http://134.149.217.112:8001
```

You must be calling from an IP allowed by the NSG (see
[Managing access](#managing-access)). Quick sanity check:

```bash
curl -s $BASE/health
curl -s $BASE/ready
curl -s $BASE/automl/endpoints | jq -r '.[].path'   # all routes
```

### Website accessibility (AutoML+)

Crawls a site, runs WCAG-inspired checks + readability analysis per page
chunk (LLM calls per chunk — a depth-2 crawl of ~20 pages takes several
minutes), and returns an LLM summary plus a rendered HTML report:

```bash
curl -s -X POST "$BASE/automl/automl_plus/web_access/analyze/" \
  -F "url=https://example.com/" \
  -F "depth=2" \
  -F "include_html=true" \
  -o report.json

# human-readable report (open in a browser)
jq -r '.html_report' report.json > report.html

# key numbers
jq '{average_score, pages: (.pages_crawled|length), readability}' report.json
```

### Alt-text check (AutoML+)

Asks a VLM whether an image's alt text is meaningful:

```bash
curl -s -X POST "$BASE/automl/automl_plus/web_access/check-alt-text/" \
  -F "image_url=https://example.com/logo.png" \
  -F "alt_text=Company logo"
```

### Run a VLM on an image (AutoML+)

Free-form prompt against a vision-language model, either by URL or by file
upload (omit the other input):

```bash
# by URL
curl -s -X POST "$BASE/automl/automl_plus/image_tools/run_on_image/" \
  -F "prompt=Describe this image" \
  -F "image_url=https://example.com/photo.jpg"

# by file upload
curl -s -X POST "$BASE/automl/automl_plus/image_tools/run_on_image/" \
  -F "prompt=Describe this image" \
  -F "image_file=@photo.jpg"
```

A streaming variant exists at `image_tools/run_on_image_stream/`.

### Model training (tabular / vision / audio / text)

!!! warning
    The training endpoints fetch datasets from and upload trained models to
    **AutoDW** (`AUTODW_URL` in `.env`). They only work if a reachable AutoDW
    instance is configured on the VM — by default the compose file points at
    `host.docker.internal:8000`, i.e. the VM itself, where no AutoDW runs.
    The AutoML+ pipelines above do not need AutoDW.

Each modality exposes `best_model/` (train), `accepted_format/` (expected
dataset layout), and `deployment_instructions/`. Examples:

```bash
# expected dataset format per service
curl -s -X POST "$BASE/automl/tabular/accepted_format/" | jq .

# tabular training (needs an AutoDW user_id + dataset_id)
curl -s -X POST "$BASE/automl/tabular/best_model/" \
  -F "user_id=1" \
  -F "dataset_id=2" \
  -F "target_column_name=signature" \
  -F "task_type=classification" \
  -F "time_budget=30"

# vision / audio: dataset is a zip of media + labels.csv
curl -s -X POST "$BASE/automl/vision/best_model/" \
  -F "user_id=1" -F "dataset_id=2" -F "task_type=image_classification"

# text: dataset is a CSV; text_column selects the input text
curl -s -X POST "$BASE/automl/text/best_model/" \
  -F "user_id=1" -F "dataset_id=2" \
  -F "task_type=text_classification" -F "text_column=text" \
  -F "target_column_name=label"

# vision multimodal (image + tabular)
curl -s -X POST "$BASE/automl/vision/multimodal_best_model/" \
  -F "user_id=1" -F "dataset_id=2" -F "task_type=multimodal_classification"
```

Training is synchronous and long-running: send these with a generous HTTP
timeout (or from a script that polls), and follow progress on the VM with
`docker logs -f alfie_automl_engine`.

## Troubleshooting

- **Access stopped working from a previously allowed machine**: home/office
  public IPs change (DHCP). Check the machine's current IP
  (`curl -s https://icanhazip.com`) and update the rule as shown above.
- **Container restart loop / app errors**:
  `docker logs --tail 50 alfie_automl_engine` on the VM.
- **Health endpoint works on the VM but not remotely**: the NSG rule is
  missing, the source IP is stale, or something between (corporate firewall)
  blocks outbound port 8001.
- **`az` token expired**: run `az login` again (see step 1).
- Note that training endpoints additionally need a reachable AutoDW
  (`AUTODW_URL` in `.env`); `/health`, `/ready`, and AutoML+ endpoints do not.
