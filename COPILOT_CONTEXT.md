# Copilot Context — visual-finetune-lab

> Leggi questo file all'inizio della sessione per riprendere il contesto completo.

---

## Chi sono

Tendresse Dutra — AI Backend Engineer (donna), 10+ anni backend distribuito.
Posizionamento attuale: LLMOps / AI Backend Engineer.
2 progetti AI su GitHub come asset principali per la ricerca lavoro.

---

## Questo progetto: visual-finetune-lab

**Scopo:** pipeline end-to-end per specializzare Phi-3.5-Vision su documenti custom
(fatture, form, screenshot) con QLoRA — ~10x cost reduction vs GPT-4o a runtime.

**Repo:** https://github.com/tdutra-dev/visual-finetune-lab

### Stato al 17 Aprile 2026

| Fase | Componente | Stato |
|---|---|---|
| 1 | Preprocessing (`ImageProcessor`) | ✅ Completato, testato su fattura PayPal reale |
| 2 | Dataset generation (`SyntheticDatasetGenerator`) | ✅ 20 Q&A pairs in `data/datasets/dataset.jsonl` |
| 3 | Fine-tuning QLoRA (`LoRATrainer`) | ⏳ Codice completo, GPU run pending |
| 4 | Evaluation (`ModelEvaluator`) | ⏳ Codice completo, dipende dal checkpoint |
| 5 | MLflow tracking | ⏳ DB inizializzato localmente, da configurare su Databricks CE |
| 6 | FastAPI serving (`api.py`) | ✅ Codice completo, endpoints /predict /predict/text /health |
| 7 | Deploy Azure Container Apps | ⏳ Dockerfile + azure/container-app.yml pronti |

### Perché ci siamo bloccati sul training

Il vecchio notebook (Dell Vostro 5568) era variante Intel-only — nessuna GPU NVIDIA.
Il nuovo notebook ha badge NVIDIA + Intel Core i7 — dovrebbe avere GPU dedicata.
Verificare con `nvidia-smi` prima di eseguire il training.

### Note tecniche critiche

- **`transformers >= 5` è incompatibile** con Phi-3.5-Vision — pin a `<5` già in `pyproject.toml`
- **`_hide_flash_attn` RecursionError** — fix applicato: `_REAL_FIND_SPEC` catturato a livello
  modulo in `evaluator.py` e `lora_trainer.py`
- **Dataset già generato** — `data/datasets/dataset.jsonl`, 20 Q&A pairs dalla fattura PayPal.
  Non rieseguire la cella GPT-4o (costo API)
- **Notebook unificato** — `notebooks/pipeline.ipynb` rileva automaticamente locale vs Colab

### Prossimo step

1. Verificare GPU: `nvidia-smi` + `python -c "import torch; print(torch.cuda.is_available())"`
2. Configurare MLflow → Databricks CE (aggiungere `DATABRICKS_HOST` e `DATABRICKS_TOKEN` al `.env`)
3. Modificare `mlflow_tracker.py`: `mlflow.set_tracking_uri("databricks")`
4. Eseguire `python scripts/run_pipeline.py --images data/raw/ --epochs 3 --lora-rank 16`
5. Annotare metriche reali (BLEU, ROUGE-L, LLM-judge) e aggiornare CV

---

## L'altro progetto: LLM-QA-OPS-LAB

**Repo:** https://github.com/tdutra-dev/LLM-QA-OPS-LAB
**Stato:** Completato e pushato su GitHub (Apr 16, 2026).
README con screenshot Grafana reali, RAG 100%, fix RAG SQL bug applicato.
Per il check CV↔codice: aprire quel progetto in VS Code separatamente.

---

## CV — versione finale (Apr 17, 2026)

### PROFILE
Backend Engineer with 10+ years in distributed systems.
I build AI infrastructure on that foundation — agentic pipelines, batch LLM analysis
with faithfulness evaluation, RAG with pgvector, vision model fine-tuning, and
production observability.

### VISUAL-FINETUNE-LAB nel CV (sezione da aggiornare post-training)
- Preprocessing: OpenCV normalizes document images — modular, tested on real PayPal invoice
- Dataset generation: GPT-4o produces structured Q&A pairs — no manual labeling, JSONL; 20 pairs validated
- QLoRA fine-tuning: Phi-3.5-Vision with LoRA adapters, ~10x cost reduction vs GPT-4o; pipeline complete, checkpoint in progress
- Evaluation + tracking: BLEU, ROUGE-L, LLM-as-judge; every run logged to MLflow on Databricks
- Serving: FastAPI containerized for Azure Container Apps — /predict, /predict/text, /health

**Dopo il training:** sostituire "checkpoint in progress" con metriche reali.

### LinkedIn Top Skills (pinnate)
1. Python
2. Large Language Models (LLM)
3. MLOps
4. FastAPI
5. RAG

---

## File importanti nel progetto

| File | Contenuto |
|---|---|
| `data/datasets/dataset.jsonl` | 20 Q&A pairs generati da fattura PayPal |
| `data/raw/paypal_invoice.png` | Immagine di test principale |
| `notebooks/pipeline.ipynb` | Notebook unificato locale/Colab |
| `scripts/run_pipeline.py` | Entry point pipeline completa |
| `.env.example` | Template variabili d'ambiente |
| `MONDAY_PLAN.md` | Piano dettagliato per lunedì 20 aprile |

---

## Comandi utili

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verifica GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Test preprocessing (senza GPU)
python scripts/test_invoice.py data/raw/paypal_invoice.png

# Pipeline completa (richiede GPU)
python scripts/run_pipeline.py --images data/raw/ --epochs 3 --lora-rank 16

# Test suite (no GPU required)
pytest tests/ -v
```
