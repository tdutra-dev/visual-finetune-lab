# visual-finetune-lab

End-to-end pipeline for specializing vision-language models on domain-specific documents.

**Problem:** Generic models are expensive and inaccurate on custom document layouts (invoices, dashboards, forms). This pipeline fine-tunes a lightweight model on *your* data — making inference 10x cheaper and more accurate.

```
Images (invoices / screenshots / PDFs)
  → OpenCV preprocessing  (resize, deskew, region detection)
  → Synthetic dataset      (GPT-4o Vision generates Q&A pairs automatically)
  → LoRA fine-tuning       (PEFT + QLoRA, runs on consumer GPU or Colab T4)
  → Evaluation             (BLEU + ROUGE-L + LLM-as-judge)
  → MLflow tracking        (Databricks CE or local)
  → FastAPI serving        (Azure Container Apps, scales to zero)
```

---

## Stack

| Layer | Technology |
|---|---|
| Image preprocessing | OpenCV, Pillow |
| Synthetic dataset generation | OpenAI GPT-4o Vision |
| Fine-tuning | HuggingFace PEFT, LoRA/QLoRA, `bitsandbytes` |
| Base model | `microsoft/Phi-3.5-vision-instruct` |
| Experiment tracking | MLflow (Databricks Community Edition) |
| Serving | FastAPI + Uvicorn |
| Cloud storage | Azure Blob Storage (images + checkpoints) |
| Deploy | Azure Container Apps |
| Tests | Pytest |

---

## Quickstart

```bash
git clone https://github.com/tdutra-dev/visual-finetune-lab
cd visual-finetune-lab

cp .env.example .env
# edit .env with your keys

pip install -e ".[dev]"

# Put your images in data/raw/
python scripts/run_pipeline.py --images data/raw/ --epochs 3 --lora-rank 16
```

The script runs all 4 stages and logs everything to MLflow.

> **Note:** stages 3–4 (fine-tuning & evaluation) require a GPU. No local GPU? Use [Google Colab T4](https://colab.research.google.com) — upload `notebooks/pipeline.ipynb` and run with a free T4 runtime.

---

## Project structure

```
src/visual_finetune_lab/
├── preprocessing/      # OpenCV image processor (resize, deskew, region detection)
├── dataset/            # GPT-4o Vision → synthetic Q&A JSONL dataset
├── training/           # LoRA/QLoRA trainer (PEFT + HuggingFace Trainer)
├── evaluation/         # BLEU + ROUGE-L + LLM-as-judge evaluator
├── tracking/           # MLflow experiment tracker
└── serving/            # FastAPI REST API for model inference
scripts/
└── run_pipeline.py     # End-to-end pipeline entry point
tests/                  # Pytest suite (no GPU required, fully mocked)
azure/                  # Azure Container Apps deployment config
```

---

## Use cases

- **Invoice extraction** — fine-tune on 50-500 scanned invoices, extract structured fields at 1/10th the cost of GPT-4o per call
- **Dashboard analysis** — analyze Grafana/Datadog screenshots, detect anomalies automatically
- **Form processing** — extract fields from domain-specific forms with layout-specific accuracy

---

## Run with Docker

```bash
docker compose up
# API at http://localhost:8000
# MLflow UI at http://localhost:5000

curl -X POST http://localhost:8000/predict \
  -F "question=What is the total amount?" \
  -F "image=@data/raw/invoice_001.jpg"
```

---

## Deploy to Azure

```bash
# Build and push image
az acr build --registry your_acr --image visual-finetune-lab:latest .

# Deploy Container App
az containerapp create --yaml azure/container-app.yml
```

---

`Python · OpenCV · Pillow · OpenAI API · HuggingFace Transformers · PEFT · LoRA · QLoRA · MLflow · Databricks · FastAPI · Azure Container Apps · Azure Blob Storage · Pytest`
