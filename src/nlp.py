"""
Precomputado de sentimiento para 3.5 (avance) y comparaciones de sentimiento
por grupo para la actividad 9 (entrega final, Persona 2). El modelo de
sentimiento se corre una sola vez aqui (analyze_comments/run_sentiment); la
actividad 9 reutiliza sentiment_comments.csv, nunca vuelve a implementar o
correr otro modelo (evita DRY roto y resultados inconsistentes entre 3.5 y 9).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

SENTIMENT_LABELS = ("POS", "NEU", "NEG")


def load_sentiment_analyzer():
    try:
        from pysentimiento import create_analyzer
    except ImportError as exc:
        raise RuntimeError("Falta pysentimiento. Instale requirements.txt para generar sentiment_comments.csv.") from exc
    return create_analyzer(task="sentiment", lang="es")


def analyze_comments(comments: pd.DataFrame, analyzer=None) -> pd.DataFrame:
    """
    9.1: analiza texto_original (nunca texto_limpio: pysentimiento ya trae su
    propio preprocesamiento social para texto de redes -URLs, menciones,
    hashtags, emojis- y aplicarle ademas la limpieza agresiva de 2.6 le
    quitaria senales utiles para el modelo, como signos de exclamacion o
    mayusculas). Guarda etiqueta y las 3 probabilidades por comentario.
    """
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


# ---------------------------------------------------------------------------
# 9.2 - comparacion de sentimiento por video, canal, categoria y comunidad
# ---------------------------------------------------------------------------

def _group_sentiment_row(group_type: str, group_key: str, labels: pd.Series) -> dict:
    """Una fila de resumen para un grupo: n y % por etiqueta, con aviso de muestra chica."""
    n = len(labels)
    dist = labels.value_counts(normalize=True) * 100 if n else pd.Series(dtype=float)
    row = {"group_type": group_type, "group_key": group_key, "n_comments": n}
    for label in SENTIMENT_LABELS:
        row[f"pct_{label}"] = round(float(dist.get(label, 0.0)), 2) if n else None
    row["small_sample"] = n < config.SENTIMENT_SMALL_GROUP_N
    return row


def build_sentiment_group_summary(
    sentiment_comments: pd.DataFrame,
    videos_clean: pd.DataFrame,
    community_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compara la distribucion de sentimiento (POS/NEU/NEG) por video, canal,
    categoria y -si hay community_assignments- por comunidad de la proyeccion
    video-video (actividad 7). La regla del plan se aplica literal: n<5 se
    reporta igual (nunca se oculta el grupo), solo se marca small_sample=True
    para que la interpretacion en 9.3 la trate con cautela.
    """
    # comments_clean (y por lo tanto sentiment_comments) ya trae columnas
    # redundantes del video (video_title/channel_name/channel_id, ver
    # data_dictionary.csv: "atributo_redundante", se obtienen via join). Se
    # descartan antes de mezclar con videos_clean para no duplicar columnas
    # ni quedarnos con la version desactualizada/parcial de comments.
    overlap = {"title", "channel_id", "channel_name", "category"} & set(sentiment_comments.columns)
    df = sentiment_comments.drop(columns=list(overlap)).merge(
        videos_clean[["video_id", "title", "channel_id", "channel_name", "category"]],
        on="video_id", how="left", validate="many_to_one",
    )

    rows = []
    for video_id, g in df.groupby("video_id"):
        title = str(g["title"].iloc[0])
        rows.append(_group_sentiment_row("video", f"{video_id} - {title}", g["sentiment_label"]))
    for channel_name, g in df.groupby("channel_name", dropna=False):
        rows.append(_group_sentiment_row("channel", str(channel_name), g["sentiment_label"]))
    for category, g in df.groupby("category", dropna=False):
        rows.append(_group_sentiment_row("category", str(category), g["sentiment_label"]))

    if community_assignments is not None and not community_assignments.empty:
        with_community = df.merge(
            community_assignments[["video_id", "community_id"]],
            on="video_id", how="inner",
        )
        for community_id, g in with_community.groupby("community_id"):
            rows.append(_group_sentiment_row("community", f"comunidad {community_id}", g["sentiment_label"]))

    return pd.DataFrame(rows)


def save_sentiment_group_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    """
    Una figura de barras apiladas (% POS/NEU/NEG) por tipo de grupo agregado
    con volumen manejable para visualizar (categoria y comunidad); video y
    canal quedan solo en la tabla porque con 19+ grupos una figura de barras
    se vuelve ilegible y no aporta mas que el csv.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"POS": "#3f7d4d", "NEU": "#9a9a9a", "NEG": "#b5533f"}

    for group_type in ("category", "community"):
        subset = summary[summary["group_type"] == group_type].sort_values("group_key")
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = pd.Series(0.0, index=subset.index)
        for label in SENTIMENT_LABELS:
            values = subset[f"pct_{label}"].fillna(0.0)
            ax.bar(subset["group_key"].astype(str), values, bottom=bottom, label=label, color=colors[label])
            bottom = bottom + values
        ax.set(ylabel="% de comentarios", ylim=(0, 105),
               title=f"Sentimiento por {group_type} (n<{config.SENTIMENT_SMALL_GROUP_N} = muestra pequeña)")
        for i, (n, small) in enumerate(zip(subset["n_comments"], subset["small_sample"])):
            marker = f"n={n}" + ("*" if small else "")
            ax.text(i, 102, marker, ha="center", va="bottom", fontsize=7)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / f"fig_sentiment_by_{group_type}.png", dpi=170)
        plt.close(fig)


# ---------------------------------------------------------------------------
# orquestacion de 9.1 + 9.2 (llamada desde main.py, etapa 'communities')
# ---------------------------------------------------------------------------

def run_sentiment_comparison() -> pd.DataFrame:
    """
    9.1 + 9.2: reutiliza sentiment_comments.csv (lo genera con run_sentiment()
    si todavia no existe) y produce sentiment_group_summary.csv + figuras. Si
    community_assignments.csv (actividad 7) ya existe, incluye la comparacion
    por comunidad; si no, compara solo por video/canal/categoria y lo deja
    documentado en consola.
    """
    sentiment_path = config.PROCESSED_DIR / "sentiment_comments.csv"
    if sentiment_path.exists():
        sentiment_comments = pd.read_csv(sentiment_path, encoding=config.CSV_ENCODING)
    else:
        print("[nlp] sentiment_comments.csv no existe, corriendo run_sentiment() primero...")
        sentiment_comments = run_sentiment()

    videos_clean = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)

    community_path = config.TABLES_DIR / "community_assignments.csv"
    community_assignments = None
    if community_path.exists():
        community_assignments = pd.read_csv(community_path, encoding=config.CSV_ENCODING)
    else:
        print(
            "[nlp] community_assignments.csv no existe todavia; "
            "sentiment_group_summary.csv se genera sin comparacion por comunidad. "
            "Corre 'python main.py --stage communities' antes para incluirla."
        )

    summary = build_sentiment_group_summary(sentiment_comments, videos_clean, community_assignments)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES_DIR / "sentiment_group_summary.csv", index=False, encoding=config.CSV_ENCODING)
    save_sentiment_group_figures(summary, config.FIGURES_DIR)

    n_small = int(summary["small_sample"].sum())
    print(
        f"[nlp] sentiment_group_summary: {len(summary)} grupos "
        f"({', '.join(sorted(summary['group_type'].unique()))}), {n_small} con muestra pequeña (n<{config.SENTIMENT_SMALL_GROUP_N})"
    )
    return summary


if __name__ == "__main__":
    run_sentiment_comparison()
