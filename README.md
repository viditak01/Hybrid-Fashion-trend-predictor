Hybrid Fashion Trend Predictor — BERT + XGBoost hybrid

Predicts how likely a fashion product idea is to succeed, using a free-text
description plus structured attributes (gender, category, fabric, theme,
season, occasion, color). Combines BERT-style text embeddings with XGBoost
on top of structured features, then exposes the whole thing as an
interactive form inside a Jupyter notebook.

Files:
1.fashion_predictor.ipynb - The main notebook — load data, embed text, train models, interactive form
2.fashion_master.csv - The dataset (6,000 rows, 17 columns) — already model-ready

How it works:

idea_text (free text)
        │
        ▼
  BERT embedding (all-MiniLM-L6-v2, 384-dim)
        │
        ├──── + encoded categorical features (gender, category, subcategory,
        │       fabric, theme, season, occasion, color)
        │
        ├──── + numeric features (price, rating, review_count)
        │
        ▼
  XGBoost Regressor  → success_score   (0–100)
  XGBoost Classifier → success_label   (low / medium / high)

The two models share the same feature matrix and train side by side.

