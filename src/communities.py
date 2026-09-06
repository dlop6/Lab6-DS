"""
Actividad 7 (entrega final, Persona 2): deteccion de comunidades sobre la
proyeccion video-video que ya entrega src/networks.py.

Este modulo no reconstruye la bipartita ni las proyecciones: siempre las pide
a src/networks.py (mismo principio DRY que ya usa src/metrics.py para la
actividad 6). Tampoco recalcula pesos a mano: Louvain corre directamente
sobre los pesos de video_projection (numero de autores compartidos).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, eda, networks

# ---------------------------------------------------------------------------
# 7.1 - eleccion de red y justificacion (texto reutilizable en el informe)
# ---------------------------------------------------------------------------

NETWORK_CHOICE_TEXT = (
    "Se eligio la proyeccion video-video (ponderada) para detectar comunidades, "
    "no la bipartita ni la proyeccion autor-autor. La bipartita mezcla dos tipos "
    "de nodo con roles distintos (autor/video) y Louvain esta pensado para redes "
    "de un solo tipo de nodo, asi que aplicarlo directo sobre la bipartita "
    "confundiria comunidades de contenido con el propio patron autor-video. La "
    "proyeccion autor-autor agruparia personas, no videos, y con 332 autores "
    "donde el 97% comento en un unico video (ver peripheral_nodes.csv de la "
    "actividad 6), la mayoria de esas comunidades solo reflejarian pares "
    "aislados sin aportar mucho al analisis de contenido. La proyeccion "
    "video-video conecta videos que comparten autores: sus comunidades se "
    "interpretan como agrupaciones de videos con audiencias solapadas, que es "
    "exactamente lo que se necesita para caracterizar 'comunidades de "
    "participacion' por tema/canal/sentimiento en 7.5. El sesgo a tener en "
    "cuenta: esta proyeccion parte de incidencia binaria (weighted_projected_graph "
    "cuenta autores compartidos, no volumen de comentarios), y solo cubre los 19 "
    "videos con comentarios recolectados -- no dice nada sobre los otros 274 "
    "videos del dataset sin cobertura de comentarios."
)


def choose_projection_for_communities() -> str:
    return NETWORK_CHOICE_TEXT


# ---------------------------------------------------------------------------
# 7.2 - algoritmo (Louvain, pesos = autores compartidos, seed=42)
# ---------------------------------------------------------------------------

def detect_louvain_communities(
    G: nx.Graph,
    seed: int = config.SEED,
    weight: str = "weight",
    resolution: float = 1.0,
) -> list[set]:
    """
    Louvain sobre G usando el peso de las aristas (weight="weight") y
    resolution=1 (sin favorecer comunidades mas grandes ni mas chicas que la
    resolucion estandar). Louvain es heuristico: optimiza modularidad de forma
    aproximada (no exhaustiva), por lo que el resultado puede variar levemente
    entre corridas si no se fija una semilla -- aqui seed=42 asegura que el
    mismo grafo produzca siempre la misma particion.

    Devuelve la particion cruda de networkx (lista de sets de node_id). Los
    nodos sin ninguna arista (videos que no comparten autor con ningun otro
    video de los 19) quedan cada uno en su propia comunidad de tamano 1;
    Louvain no los descarta ni los fusiona artificialmente.
    """
    if G.number_of_nodes() == 0:
        return []
    return louvain_communities(G, weight=weight, resolution=resolution, seed=seed)


def _order_communities(communities: list[set]) -> list[tuple[int, set]]:
    """
    Asigna community_id de forma deterministica: ordena por tamano descendente
    y, en empate, por el node_id minimo (orden alfabetico), para que la misma
    particion siempre produzca los mismos ids sin importar el orden interno
    con el que Louvain devolvio los sets.
    """
    ordered = sorted(communities, key=lambda s: (-len(s), min(s)))
    return list(enumerate(ordered, start=1))


# ---------------------------------------------------------------------------
# 7.3 - resultado: tamanos, modularidad, tabla de asignacion
# ---------------------------------------------------------------------------

def build_community_assignments(G: nx.Graph, communities: list[set]) -> pd.DataFrame:
    """Una fila por nodo de video_projection con su community_id asignado."""
    ordered = _order_communities(communities)
    rows = []
    for community_id, members in ordered:
        for node_id in members:
            attrs = G.nodes[node_id]
            rows.append({
                "node_id": node_id,
                "video_id": attrs.get("video_id", node_id.split(":", 1)[-1]),
                "title": attrs.get("title"),
                "channel_id": attrs.get("channel_id"),
                "channel_name": attrs.get("channel_name"),
                "category": attrs.get("category"),
                "community_id": community_id,
                "community_size": len(members),
                "degree": G.degree(node_id),
            })
    return pd.DataFrame(rows).sort_values(["community_id", "node_id"]).reset_index(drop=True)


def build_community_metrics(G: nx.Graph, communities: list[set]) -> pd.DataFrame:
    """
    Una fila por comunidad: tamano, % de los 19 videos, peso interno promedio
    y si es una comunidad "trivial" (un solo nodo sin vecinos compartidos). La
    modularidad es una propiedad de la particion completa (no de una comunidad
    individual), asi que se reporta como la misma columna en todas las filas,
    junto con el numero total de comunidades -- asi el csv queda autocontenido
    sin necesitar un segundo archivo solo para dos numeros globales.
    """
    ordered = _order_communities(communities)
    n_total_nodes = G.number_of_nodes()
    q = modularity(G, communities, weight="weight", resolution=1.0) if communities else float("nan")

    rows = []
    for community_id, members in ordered:
        subgraph = G.subgraph(members)
        internal_edges = subgraph.number_of_edges()
        internal_weight = sum(d.get("weight", 0) for _, _, d in subgraph.edges(data=True))
        rows.append({
            "community_id": community_id,
            "n_videos": len(members),
            "pct_of_total_videos": round(100 * len(members) / n_total_nodes, 2) if n_total_nodes else 0.0,
            "internal_edges": internal_edges,
            "internal_weight_sum": internal_weight,
            "avg_internal_weight": round(internal_weight / internal_edges, 3) if internal_edges else 0.0,
            "is_trivial_singleton": len(members) == 1,
            "n_communities_total": len(communities),
            "modularity": round(q, 4),
            "algorithm": "louvain",
            "weight_used": "weight",
            "resolution": 1.0,
            "seed": config.SEED,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7.4 - visualizacion (TODAS las comunidades, ningun nodo/arista se elimina)
# ---------------------------------------------------------------------------

def draw_communities(G: nx.Graph, communities: list[set], output_path: Path) -> None:
    """
    Dibuja la proyeccion video-video completa (los 19 videos, todas las
    aristas), coloreando cada nodo segun su comunidad. Los nodos aislados
    (sin aristas, comunidades de tamano 1) se dibujan igual que el resto, no
    se ocultan ni se eliminan para "limpiar" la figura.
    """
    ordered = _order_communities(communities)
    community_of = {node: cid for cid, members in ordered for node in members}
    n_communities = len(ordered)

    cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_communities, 1))
    node_colors = [cmap(community_of[n] - 1) for n in G.nodes()]

    pos = nx.spring_layout(G, weight="weight", seed=config.SEED)

    fig, ax = plt.subplots(figsize=(10, 8))
    edge_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=0.35,
        width=[0.4 + 2.2 * (w / max_w) for w in edge_weights],
        edge_color="#999999",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=420,
        edgecolors="black", linewidths=0.6,
    )
    labels = {}
    for n in G.nodes():
        title = str(G.nodes[n].get("title") or G.nodes[n].get("video_id"))
        labels[n] = title if len(title) <= 22 else title[:19] + "..."
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(cid - 1),
                   markeredgecolor="black", markersize=9,
                   label=f"Comunidad {cid} (n={len(members)})")
        for cid, members in ordered
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=min(4, n_communities), frameon=False, fontsize=8)
    ax.set_title(
        f"Comunidades en la proyeccion video-video "
        f"({n_communities} comunidades, {G.number_of_nodes()} videos, "
        f"{G.number_of_edges()} aristas)",
        pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.5 - caracterizacion de hasta 3 comunidades principales
# ---------------------------------------------------------------------------

def _top_communities(communities: list[set], top_n: int = 3) -> list[tuple[int, set]]:
    """
    Hasta `top_n` comunidades por tamano. Si hay menos de `top_n` comunidades
    en total, se devuelven todas las disponibles (nunca se inventan
    comunidades ni se rellena con vacio).
    """
    ordered = _order_communities(communities)
    return ordered[: min(top_n, len(ordered))]


def characterize_communities(
    G: nx.Graph,
    communities: list[set],
    comments_clean: pd.DataFrame,
    sentiment_comments: pd.DataFrame | None = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Para cada una de las hasta `top_n` comunidades principales: videos,
    canales, autores distintos, intensidad de participacion (comentarios),
    terminos/bigramas/hashtags mas frecuentes, y sentimiento si esta
    disponible. Si hay menos de `top_n` comunidades, se caracterizan las que
    existan (nunca se rellena con comunidades ficticias).
    """
    top = _top_communities(communities, top_n=top_n)
    rows = []
    for community_id, members in top:
        video_ids = [G.nodes[n]["video_id"] for n in members]
        titles = [str(G.nodes[n].get("title")) for n in members]
        channels = sorted({str(G.nodes[n].get("channel_name")) for n in members})
        categories = sorted({str(G.nodes[n].get("category")) for n in members})

        subset_comments = comments_clean[comments_clean["video_id"].isin(video_ids)]
        n_comments = len(subset_comments)
        n_authors = subset_comments["author_channel_id"].nunique()

        tokens = subset_comments["texto_limpio"].fillna("").map(
            lambda x: eda.TOKEN_RE.findall(str(x).lower())
        )
        word_counts = Counter(t for row in tokens for t in row)
        bigram_counts = Counter(
            f"{a} {b}" for row in tokens for a, b in zip(row, row[1:])
        )
        hashtag_source = subset_comments["text"].fillna("") if "text" in subset_comments else pd.Series([], dtype=object)
        hashtag_counts = Counter(
            h.lower() for text in hashtag_source for h in eda.HASHTAG_RE.findall(str(text))
        )

        row = {
            "community_id": community_id,
            "n_videos": len(members),
            "video_titles": "; ".join(titles),
            "channels": "; ".join(channels),
            "categories": "; ".join(categories),
            "n_authors": int(n_authors),
            "n_comments_total": int(n_comments),
            "top_words": "; ".join(w for w, _ in word_counts.most_common(5)),
            "top_bigrams": "; ".join(b for b, _ in bigram_counts.most_common(5)),
            "top_hashtags": "; ".join(h for h, _ in hashtag_counts.most_common(5)) or "(sin hashtags)",
        }

        if sentiment_comments is not None and not sentiment_comments.empty:
            sent_subset = sentiment_comments[sentiment_comments["video_id"].isin(video_ids)]
            n_sent = len(sent_subset)
            if n_sent:
                dist = sent_subset["sentiment_label"].value_counts(normalize=True) * 100
                row.update({
                    "n_comments_with_sentiment": int(n_sent),
                    "sentiment_pos_pct": round(float(dist.get("POS", 0.0)), 2),
                    "sentiment_neu_pct": round(float(dist.get("NEU", 0.0)), 2),
                    "sentiment_neg_pct": round(float(dist.get("NEG", 0.0)), 2),
                    "sentiment_small_sample": n_sent < config.SENTIMENT_SMALL_GROUP_N,
                })
            else:
                row.update({
                    "n_comments_with_sentiment": 0,
                    "sentiment_pos_pct": None, "sentiment_neu_pct": None, "sentiment_neg_pct": None,
                    "sentiment_small_sample": True,
                })
        else:
            row.update({
                "n_comments_with_sentiment": None,
                "sentiment_pos_pct": None, "sentiment_neu_pct": None, "sentiment_neg_pct": None,
                "sentiment_small_sample": None,
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orquestacion de la etapa "communities" (llamada desde main.py)
# ---------------------------------------------------------------------------

def run_communities_stage(top_n: int = config.COMMUNITY_TOP_N) -> dict:
    """
    Punto de entrada de la etapa 'communities' (actividad 7). Reconstruye la
    bipartita y la proyeccion video-video llamando a src/networks.py (nunca a
    mano), corre Louvain con seed fija, guarda las tablas y la figura, y
    caracteriza hasta `top_n` comunidades principales. Si ya existe
    sentiment_comments.csv (generado por la etapa 'eda --sentiment' o por
    src/nlp.run_sentiment()) lo reutiliza para enriquecer la caracterizacion;
    si no existe, caracteriza sin sentimiento y lo deja documentado en el csv.
    """
    videos_clean = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)
    comments_clean = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv", encoding=config.CSV_ENCODING)

    bipartite_G = networks.build_bipartite_graph(comments_clean, videos_clean)
    video_proj = networks.build_video_projection(bipartite_G)

    communities = detect_louvain_communities(video_proj)

    assignments = build_community_assignments(video_proj, communities)
    metrics_table = build_community_metrics(video_proj, communities)

    sentiment_path = config.PROCESSED_DIR / "sentiment_comments.csv"
    sentiment_comments = None
    if sentiment_path.exists():
        sentiment_comments = pd.read_csv(sentiment_path, encoding=config.CSV_ENCODING)
    else:
        print(
            "[communities] sentiment_comments.csv no existe todavia; "
            "community_content_summary.csv se genera sin columnas de sentimiento. "
            "Corre 'python main.py --stage eda --sentiment' o src/nlp.run_sentiment() antes "
            "para incluir sentimiento por comunidad."
        )

    content_summary = characterize_communities(
        video_proj, communities, comments_clean, sentiment_comments, top_n=top_n,
    )

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(config.TABLES_DIR / "community_assignments.csv", index=False, encoding=config.CSV_ENCODING)
    metrics_table.to_csv(config.TABLES_DIR / "community_metrics.csv", index=False, encoding=config.CSV_ENCODING)
    content_summary.to_csv(config.TABLES_DIR / "community_content_summary.csv", index=False, encoding=config.CSV_ENCODING)
    draw_communities(video_proj, communities, config.FIGURES_DIR / "fig_video_communities.png")

    q = metrics_table["modularity"].iloc[0] if not metrics_table.empty else float("nan")
    print(
        f"[communities] video_projection: {len(communities)} comunidades, "
        f"modularidad={q:.4f}, caracterizadas {len(content_summary)} principales"
    )

    return {
        "graph": video_proj,
        "communities": communities,
        "assignments": assignments,
        "metrics": metrics_table,
        "content_summary": content_summary,
    }


if __name__ == "__main__":
    run_communities_stage()
