# Simple LLM-Based Text Classification — Sentiment Analysis (MI201 Project 3)


This repository implements a **sentiment classification pipeline for tweets** (labels: **positive / neutral / negative**) using **pre-trained LLM embeddings** (e.g., BERT-style encoders) and **classical machine-learning classifiers** (e.g., SVM, Random Forest).

---

## Project idea

Instead of training a deep model end-to-end, we:
1. **Clean / preprocess** tweet text
2. **Embed** each tweet with a **pre-trained language model** into a fixed-size vector
3. Train a lightweight classifier (SVM / Random Forest / Logistic Regression) on those vectors
4. **Evaluate** performance using **accuracy, F1-score, and confusion matrix** :contentReference[oaicite:2]{index=2}

This is a simple and effective way to test how informative modern LLM representations are for downstream classification tasks. :contentReference[oaicite:3]{index=3}

---

## Dataset

A public sentiment analysis dataset containing:
- `text`: full tweet text  
- `selected_text`: key phrase expressing sentiment  
- `sentiment`: {`neutral`, `positive`, `negative`}  
- metadata such as tweet time, age group, country, etc. :contentReference[oaicite:4]{index=4}

> Note: Place the dataset file(s) in `data/` (or update paths in your code/notebooks accordingly).

---

## Tech stack

Typical dependencies for this pipeline:
- Python 3.10+
- `pandas`, `numpy`
- `scikit-learn`
- HuggingFace `transformers` (or `sentence-transformers`)
- `torch` (if required by the embedding model)
- `matplotlib` (for plots)

---

## Repository structure (suggested)

```text
.
├── data/                 # raw/processed dataset files
├── notebooks/            # experiments, baselines, analysis
├── src/                  # reusable code: preprocessing, embeddings, training
├── reports/              # figures, tables, results
├── requirements.txt
└── README.md
````