"""Precomputado de sentimiento para 3.5; reutilizable sin recalcular en actividad 9."""
from __future__ import annotations
import pandas as pd
from . import config


def load_sentiment_analyzer():
    try:
        from pysentimiento import create_analyzer
    except ImportError as exc:
        raise RuntimeError("Falta pysentimiento. Instale requirements.txt para generar sentiment_comments.csv.") from exc
    return create_analyzer(task="sentiment", lang="es")


def analyze_comments(comments: pd.DataFrame, analyzer=None) -> pd.DataFrame:
    if "texto_original" not in comments:
        raise ValueError("H1 debe proporcionar texto_original; no se usa texto_limpio para sentimiento.")
    analyzer = analyzer or load_sentiment_analyzer()
    rows = []
    for text in comments.texto_original.fillna("").astype(str):
        result = analyzer.predict(text)
        rows.append({"sentiment_label": result.output, "sentiment_prob_NEG": result.probas.get("NEG", 0.0),
                     "sentiment_prob_NEU": result.probas.get("NEU", 0.0), "sentiment_prob_POS": result.probas.get("POS", 0.0)})
    return pd.concat([comments.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def run_sentiment() -> pd.DataFrame:
    comments = pd.read_csv(config.PROCESSED_DIR/"comments_clean.csv", encoding=config.CSV_ENCODING)
    result = analyze_comments(comments)
    result.to_csv(config.PROCESSED_DIR/"sentiment_comments.csv", index=False, encoding=config.CSV_ENCODING)
    return result
