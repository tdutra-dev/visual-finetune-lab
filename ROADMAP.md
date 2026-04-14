# Visual Fine-Tune Lab — Roadmap

## Obiettivo del progetto

Creare una pipeline end-to-end per specializzare un modello visivo leggero (`Phi-3.5-vision-instruct`) su documenti specifici (fatture, ricevute, form).

**Problema risolto:** i modelli generici (GPT-4o) costano molto e non sono precisi su layout custom.  
**Soluzione:** usare GPT-4o solo per generare il dataset di training, poi addestrare un modello locale che risponde autonomamente — 10x più economico a regime.

```
Immagine documento
  → Preprocessing OpenCV       (resize, deskew, regioni)
  → GPT-4o Vision              (genera Q&A pairs automaticamente)
  → Fine-tuning QLoRA          (Phi-3.5-vision su GPU)
  → Evaluation                 (BLEU + ROUGE-L + LLM judge)
  → MLflow tracking            (metriche e artefatti)
  → FastAPI serving            (REST API, deploy su Azure)
```

---

## Stack tecnologico

| Layer | Tecnologia |
|---|---|
| Preprocessing immagini | OpenCV, Pillow |
| Generazione dataset | OpenAI GPT-4o Vision API |
| Fine-tuning | HuggingFace PEFT + LoRA/QLoRA + bitsandbytes |
| Modello base | `microsoft/Phi-3.5-vision-instruct` |
| Experiment tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Deploy cloud | Azure Container Apps |
| Notebook | Jupyter / Google Colab T4 |

---

## Stato attuale — Aprile 2026

### ✅ Completato

| Fase | Componente | Note |
|---|---|---|
| 1 | Preprocessing (`ImageProcessor`) | Funzionante, testato su fattura PayPal reale |
| 2 | Generazione dataset (`SyntheticDatasetGenerator`) | Funzionante, genera 10 Q&A pairs per immagine |
| — | Script di test rapido | `scripts/test_invoice.py` — legge da `.env` |
| — | Test di integrazione | `tests/test_invoice_integration.py` |
| — | Fix DynamicCache (transformers 5.x) | Patch in `evaluator.py` + notebook |
| — | Fix RecursionError `_hide_flash_attn` | `_REAL_FIND_SPEC` salvato a livello modulo |
| — | Pin `transformers<5` nel notebook | Evita incompatibilità con Phi-3.5 |
| — | File `.env` configurato | `OPENAI_API_KEY` caricata da dotenv |

### 🔄 In corso

- Installazione driver NVIDIA (`nvidia-driver-550`) per abilitare GPU locale
- GPU necessaria per eseguire le fasi 3–5

### ⏳ Da completare

| Fase | Componente | Dipendenza |
|---|---|---|
| 3 | Fine-tuning QLoRA (`LoRATrainer`) | GPU NVIDIA |
| 4 | Evaluation (`ModelEvaluator`) | GPU + checkpoint trained |
| 5 | MLflow tracking | Completamento training |
| 6 | FastAPI serving (`api.py`) | Checkpoint trained |
| 7 | Deploy Azure Container Apps | Serving funzionante |

---

## Come riprendere il lavoro

### Test rapido (senza GPU)
```bash
cd ~/projects/visual-finetune-lab
source .venv/bin/activate
python scripts/test_invoice.py data/raw/paypal_invoice.png
```

### Test di integrazione
```bash
pytest tests/test_invoice_integration.py -v -s
```

### JupyterLab locale (senza GPU — fasi 1-3 del notebook)
```bash
source .venv/bin/activate
jupyter lab --notebook-dir=notebooks
```

### Pipeline completa (richiede GPU)
```bash
source .venv/bin/activate
python scripts/run_pipeline.py --images data/raw/ --epochs 3 --lora-rank 16
```

### Colab T4 (quando GPU locale non disponibile)
- Aprire `notebooks/pipeline.ipynb` su Google Colab
- Usare **Runtime → Disconnect and delete runtime** per reset pulito
- `transformers<5` già pinnato nella cella pip install

---

## Immagini di test disponibili

| File | Documento | Usato in |
|---|---|---|
| `data/raw/paypal_invoice.png` | Ricevuta PayPal (ChatGPT Plus, 23€, apr 2026) | Test confermato ✅ |

---

## Note tecniche importanti

- **`transformers>=5` è incompatibile** con il codice custom di Phi-3.5-vision (`DynamicCache.from_legacy_cache`, `get_usable_length`, `to_legacy_cache` rimossi). Pin a `<5` nel notebook e `pyproject.toml`.
- **`_hide_flash_attn`** in `LoRATrainer` e `ModelEvaluator` usava `_orig = importlib.util.find_spec` localmente — causava `RecursionError` al secondo run. Fix: `_REAL_FIND_SPEC` catturato a livello modulo.
- **GPU locale:** driver non ancora caricati (installazione in corso al 14/04/2026). Fino ad allora, usare Colab T4 per le fasi 3-5.
