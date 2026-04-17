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

## Stato attuale — 17 Aprile 2026

### ✅ Completato

| Fase | Componente | Note |
|---|---|---|
| 1 | Preprocessing (`ImageProcessor`) | Funzionante, testato su fattura PayPal reale |
| 2 | Generazione dataset (`SyntheticDatasetGenerator`) | **20 Q&A pairs generati** — `data/datasets/dataset.jsonl` (5.6 KB) |
| — | Script di test rapido | `scripts/test_invoice.py` — legge da `.env` |
| — | Test di integrazione | `tests/test_invoice_integration.py` |
| — | Fix DynamicCache (transformers 5.x) | Patch in `evaluator.py` + notebook |
| — | Fix RecursionError `_hide_flash_attn` | `_REAL_FIND_SPEC` salvato a livello modulo |
| — | Pin `transformers<5` nel notebook | Evita incompatibilità con Phi-3.5 |
| — | File `.env` configurato | `OPENAI_API_KEY` caricata da dotenv |
| — | Notebook locale/Colab unificato | `pipeline.ipynb` gira in VS Code, JupyterLab e Colab |
| — | MLflow DB inizializzato | `mlflow.db` creato (648 KB), run loggato |

### ❌ Hardware locale

- **GPU non disponibile**: il Dell Vostro 5568 è variante Intel-only (HD 620), nessuna GTX 940MX sull'hardware
- Secure Boot disabilitato ma irrilevante — nessun dispositivo PCI NVIDIA presente
- **Soluzione adottata**: usare Google Colab T4 per le fasi 3–5

### ⏳ Da completare

| Fase | Componente | Dove | Dipendenza |
|---|---|---|---|
| 3 | Fine-tuning QLoRA (`LoRATrainer`) | **Google Colab T4** | Dataset pronto ✅ |
| 4 | Evaluation (`ModelEvaluator`) | **Google Colab T4** | Checkpoint trained |
| 5 | MLflow tracking (metriche reali) | Colab / locale | Completamento training |
| 6 | FastAPI serving (`api.py`) | ✅ Implementato | In attesa del checkpoint per girare |
| 7 | Deploy Azure Container Apps | Azure | Checkpoint trained |

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

### Colab T4 — Fine-tuning (fasi 3-5)

Il dataset è pronto in `data/datasets/dataset.jsonl`. Per eseguire il fine-tuning:

1. Vai su [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → seleziona `notebooks/pipeline.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Nella cella Secrets di Colab (icona 🔑 a sinistra) aggiungi `OPENAI_API_KEY`
5. Nella sezione **"3. Synthetic Dataset Generation"**, carica il `dataset.jsonl` già pronto:
   - Usa il pannello file di Colab per uploadare `data/datasets/dataset.jsonl` in `/content/visual-finetune-lab/data/datasets/`
   - La cella rileverà il file esistente e salterà la ri-generazione GPT-4o
6. Esegui tutte le celle — il fine-tuning partirà automaticamente con GPU disponibile
7. Scarica il checkpoint da Colab al termine: `checkpoints/best/` → copia in locale

---

## Immagini di test disponibili

| File | Documento | Usato in |
|---|---|---|
| `data/raw/paypal_invoice.png` | Ricevuta PayPal (ChatGPT Plus, 23€, apr 2026) | Test confermato ✅ |

---

## Note tecniche importanti

- **`transformers>=5` è incompatibile** con il codice custom di Phi-3.5-vision (`DynamicCache.from_legacy_cache`, `get_usable_length`, `to_legacy_cache` rimossi). Pin a `<5` nel notebook e `pyproject.toml`.
- **`_hide_flash_attn`** in `LoRATrainer` e `ModelEvaluator` usava `_orig = importlib.util.find_spec` localmente — causava `RecursionError` al secondo run. Fix: `_REAL_FIND_SPEC` catturato a livello modulo.
- **GPU locale assente**: Dell Vostro 5568 è variante Intel-only. Nessuna GTX 940MX montata. Usare Colab T4.
- **Notebook unificato**: `pipeline.ipynb` rileva automaticamente l'ambiente (locale/Colab) e salta le celle che richiedono GPU quando non disponibile.
- **Dataset già generato**: `data/datasets/dataset.jsonl` — 20 Q&A pairs dalla fattura PayPal reale. Non rieseguire la cella GPT-4o senza motivo (costo API).
