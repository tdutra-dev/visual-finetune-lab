# Piano Lunedì 20 Aprile 2026 — visual-finetune-lab

## Obiettivo della giornata
Completare il fine-tuning QLoRA su GPU locale, loggare su Databricks CE,
eseguire evaluation reale e ottenere metriche da inserire nel CV.

---

## Mattina — Setup nuovo notebook

- [ ] Clonare il repo
  ```bash
  git clone https://github.com/tdutra-dev/visual-finetune-lab
  cd visual-finetune-lab
  ```

- [ ] Verificare GPU disponibile
  ```bash
  nvidia-smi
  ```

- [ ] Creare virtualenv e installare dipendenze
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  ```

- [ ] Verificare che bitsandbytes veda la GPU
  ```bash
  python -c "import bitsandbytes; print(bitsandbytes.__version__)"
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

- [ ] Copiare `.env` dal vecchio PC (contiene OPENAI_API_KEY)
  - File da copiare: `.env` nella root del progetto
  - Oppure ricreare da `.env.example` e reinserire la chiave

- [ ] Copiare il dataset già generato (evita costi API GPT-4o)
  - File: `data/datasets/dataset.jsonl` (20 Q&A pairs, fattura PayPal)

---

## Metà mattina — Databricks CE

- [ ] Creare account su https://community.cloud.databricks.com (se non esistente)
- [ ] Da Databricks UI: Settings → Developer → Access Tokens → genera token
- [ ] Aggiungere al `.env`:
  ```
  DATABRICKS_HOST=https://community.cloud.databricks.com
  DATABRICKS_TOKEN=<il_tuo_token>
  ```
- [ ] Modificare tracking URI nel codice — aprire con Copilot e chiedere di aggiornare
  `mlflow.set_tracking_uri("databricks")` in `src/visual_finetune_lab/tracking/mlflow_tracker.py`

---

## Pomeriggio — Fine-tuning + Evaluation

- [ ] Mettere immagini di test in `data/raw/` (almeno la fattura PayPal)
- [ ] Eseguire pipeline completa
  ```bash
  python scripts/run_pipeline.py --images data/raw/ --epochs 3 --lora-rank 16
  ```
- [ ] Verificare run loggato su Databricks CE UI (Experiments → visual-finetune-lab)
- [ ] Annotare metriche reali: BLEU, ROUGE-L, LLM-judge score

---

## Fine giornata — CV e GitHub

- [ ] Aggiornare sezione VISUAL-FINETUNE-LAB nel CV con metriche reali
  - Sostituire "checkpoint in progress" con risultati concreti
  - Aggiornare "MLflow on Databricks" — ora reale
- [ ] Push del progetto aggiornato su GitHub
- [ ] (Opzionale) Avviare FastAPI locale e testare `/predict`

---

## Note importanti

- **Dataset già pronto**: `data/datasets/dataset.jsonl` — non rieseguire la generazione
  GPT-4o senza motivo (costo API)
- **transformers < 5**: pin già nel `pyproject.toml`, non aggiornare
- **Fix RecursionError `_hide_flash_attn`**: già applicato in `evaluator.py` e `lora_trainer.py`
- **Chiedere a Copilot**: aprire questo progetto in VS Code, Copilot avrà il contesto
  completo grazie al file `COPILOT_CONTEXT.md`
