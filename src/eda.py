"""Actividad 3: EDA reproducible sobre los artefactos limpios de H1."""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

TOKEN_RE = re.compile(r"[a-záéíóúüñ]{2,}", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#([\wáéíóúüñ]+)", re.IGNORECASE)


def _safe_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _rank_counts(series: pd.Series, value_name: str, n: int = 20) -> pd.DataFrame:
    out = series.dropna().astype(str).value_counts().head(n).rename_axis("item").reset_index(name=value_name)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def build_video_participation(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    agg = comments.groupby("video_id").agg(
        comment_count=("comment_id", "count"),
        unique_authors=("author_channel_id", "nunique"),
        replies=("reply_count", "sum"),
        comment_likes=("like_count", "sum"),
    ).reset_index()
    cols = ["video_id", "title", "channel_id", "channel_name", "category", "view_count", "query_hits"]
    return agg.merge(videos[cols], on="video_id", how="left", validate="one_to_one")


def build_eda_tables(videos: pd.DataFrame, comments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    vp = build_video_participation(videos, comments)
    summary = pd.DataFrame([
        {"metric": "videos_dataset", "value": len(videos), "denominator": "filas de videos"},
        {"metric": "channels_dataset", "value": videos.channel_id.nunique(), "denominator": "channel_id"},
        {"metric": "comments", "value": len(comments), "denominator": "filas de comentarios"},
        {"metric": "authors", "value": comments.author_channel_id.nunique(), "denominator": "author_channel_id"},
        {"metric": "commented_videos", "value": comments.video_id.nunique(), "denominator": "video_id en comentarios"},
        {"metric": "videos_without_collected_comments", "value": len(videos)-comments.video_id.nunique(), "denominator": "cobertura observada"},
        {"metric": "median_views_all_videos", "value": videos.view_count.median(), "denominator": "293 videos"},
        {"metric": "median_replies", "value": comments.reply_count.median(), "denominator": "406 comentarios"},
        {"metric": "median_comment_likes", "value": comments.like_count.median(), "denominator": "comentarios con conteo válido"},
    ])
    videos_by_channel = videos.groupby(["channel_id", "channel_name"], dropna=False).size().reset_index(name="video_count").sort_values("video_count", ascending=False)
    categories = videos.category.value_counts(dropna=False).rename_axis("category").reset_index(name="video_count")
    queries = _rank_counts(videos.source_query, "video_count")
    query_hits = _rank_counts(videos.query_hits.map(_safe_list).explode(), "video_count")
    hashtags = _rank_counts(videos.description.fillna("").map(lambda x: HASHTAG_RE.findall(str(x))).explode(), "frequency")
    tokens = comments.texto_limpio.fillna("").map(lambda x: TOKEN_RE.findall(str(x).lower()))
    words = pd.DataFrame(Counter(t for row in tokens for t in row).most_common(30), columns=["item", "frequency"])
    bigrams = pd.DataFrame(Counter(f"{a} {b}" for row in tokens for a, b in zip(row, row[1:])).most_common(30), columns=["item", "frequency"])
    for frame in (words, bigrams): frame.insert(0, "rank", range(1, len(frame)+1))
    count_descriptives = pd.DataFrame({
        "view_count_all_videos": videos.view_count,
        "reply_count_comments": comments.reply_count,
        "like_count_comments": comments.like_count,
    }).describe(percentiles=[.25, .5, .75, .9, .95]).T.reset_index(names="variable")
    channel_participation = vp.groupby(["channel_id", "channel_name"], dropna=False).agg(
        comment_count=("comment_count", "sum"), unique_videos=("video_id", "nunique"),
        unique_authors=("unique_authors", "sum")
    ).reset_index().sort_values("comment_count", ascending=False)
    return {"eda_summary": summary, "count_descriptives": count_descriptives,
            "videos_by_channel": videos_by_channel, "video_participation": vp,
            "channel_participation": channel_participation,
            "category_frequencies": categories, "query_frequencies": queries, "query_hits_frequencies": query_hits,
            "hashtag_frequencies": hashtags,
            "word_frequencies": words, "bigram_frequencies": bigrams}


def build_concentration(video_participation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = int(video_participation.comment_count.sum())
    for level, keys, label in [("video", ["video_id", "title"], "title"), ("channel", ["channel_id", "channel_name"], "channel_name")]:
        ranked = video_participation.groupby(keys, dropna=False).comment_count.sum().sort_values(ascending=False)
        for top_n in (5, 10):
            used = min(top_n, len(ranked)); count = int(ranked.head(used).sum())
            rows.append({"unit": level, "top_n_requested": top_n, "units_available": len(ranked), "comments_in_top": count,
                         "total_comments": total, "share_pct": 100*count/total})
    return pd.DataFrame(rows)


def build_views_relation(vp: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    out = vp[["video_id", "title", "view_count", "comment_count", "unique_authors"]].dropna(subset=["view_count"]).copy()
    rho, p = spearmanr(out.view_count, out.comment_count)
    out["spearman_rho"] = rho; out["spearman_p_value"] = p; out["n_videos"] = len(out)
    return out, float(rho), float(p)


def build_additional_questions(vp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = vp.groupby("category").filter(lambda g: len(g) >= config.EDA_MIN_CATEGORY_N)
    med = eligible.groupby("category").comment_count.agg(["median", "count"]).sort_values("median", ascending=False)
    answer = "Sin categorías con n>=3" if med.empty else f"{med.index[0]} (mediana={med.iloc[0]['median']:.1f}, n={int(med.iloc[0]['count'])})"
    rows.append({"question_id": "3.6.1", "question": "¿Qué categoría tiene mayor mediana de comentarios por video (n>=3)?", "answer": answer})
    rho, p = spearmanr(vp.unique_authors, vp.comment_count)
    rows.append({"question_id": "3.6.2", "question": "¿Qué relación hay entre autores únicos y comentarios por video?", "answer": f"Spearman rho={rho:.3f}, p={p:.4g}, n={len(vp)}"})
    tmp = vp.assign(query_count=vp.query_hits.map(lambda x: len(_safe_list(x))), query_group=lambda d: np.where(d.query_count>1, "multiple", "single"))
    stats = tmp.groupby("query_group").view_count.agg(["count", "median", "mean"])
    rows.append({"question_id": "3.6.3", "question": "¿Difieren descriptivamente las vistas entre videos con múltiples query_hits y uno solo?", "answer": stats.round(2).to_dict(orient="index")})
    return pd.DataFrame(rows)


def save_figures(vp: pd.DataFrame, concentration: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5)); top = vp.nlargest(10, "comment_count").sort_values("comment_count")
    labels = top.title.astype(str).map(lambda s: s if len(s) <= 43 else s[:40] + "...")
    ax.barh(labels, top.comment_count, color="#2d6a9f"); ax.tick_params(axis="y", labelsize=9); ax.set(xlabel="Comentarios observados", title="Videos con mayor participación"); fig.tight_layout(); fig.savefig(figure_dir/"eda_top_videos.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(vp.view_count, vp.comment_count, alpha=.75, color="#b55239"); ax.set_xscale("log"); ax.set(xlabel="Visualizaciones (escala log)", ylabel="Comentarios observados", title="Visibilidad y participación (19 videos)"); fig.tight_layout(); fig.savefig(figure_dir/"eda_views_comments.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(vp.unique_authors, vp.comment_count, alpha=.75, color="#3f7d4d"); ax.set(xlabel="Autores únicos", ylabel="Comentarios", title="Autores únicos y volumen de comentarios"); fig.tight_layout(); fig.savefig(figure_dir/"eda_authors_comments.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); c = concentration.copy(); labels = c.unit+" top "+c.top_n_requested.astype(str); ax.bar(labels, c.share_pct, color=["#567" if x=="video" else "#986" for x in c.unit]); ax.set(ylabel="% de comentarios", ylim=(0, 105), title="Concentración de la participación"); fig.tight_layout(); fig.savefig(figure_dir/"eda_concentration.png", dpi=180); plt.close(fig)


def run_eda(videos_path: Path | None = None, comments_path: Path | None = None) -> dict[str, pd.DataFrame]:
    videos = pd.read_csv(videos_path or config.PROCESSED_DIR/"videos_clean.csv", encoding=config.CSV_ENCODING)
    comments = pd.read_csv(comments_path or config.PROCESSED_DIR/"comments_clean.csv", encoding=config.CSV_ENCODING)
    tables = build_eda_tables(videos, comments); vp = tables["video_participation"]
    tables["concentration"] = build_concentration(vp)
    tables["views_comments_relation"], _, _ = build_views_relation(vp)
    tables["additional_questions"] = build_additional_questions(vp)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items(): frame.to_csv(config.TABLES_DIR/f"{name}.csv", index=False, encoding=config.CSV_ENCODING)
    save_figures(vp, tables["concentration"], config.FIGURES_DIR)
    return tables
