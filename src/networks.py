"""
Actividad 4 y soporte de actividad 5: red bipartita autor-video, tablas de
nodos/aristas, y funciones reutilizables de proyeccion (incidencia binaria).

No se reconstruye a mano en otros modulos:
quien necesite el grafo o las proyecciones importa estas funciones.

Regla de oro heredada de la guia y del plan: reply_count NUNCA participa en la
construccion de una arista. Una arista autor-video significa unicamente "el
autor publico al menos un comentario en ese video"; el peso es la cantidad de
comentarios de ese autor en ese video. No se interpreta como amistad,
conversacion directa ni aprobacion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

AUTHOR_PREFIX = "author:"
VIDEO_PREFIX = "video:"

EDGE_MEANING = (
    "Una arista autor-video indica que ese autor publico al menos un comentario "
    "principal en ese video. El peso (weight_comments) es la cantidad de "
    "comentarios que ese autor publico en ese video especifico. La arista NO "
    "debe interpretarse como amistad, conversacion directa, respuesta o "
    "aprobacion entre personas: los datos no permiten identificar quien "
    "respondio a quien (reply_count cuenta respuestas recibidas, pero nunca "
    "identifica a sus autores), asi que reply_count nunca se usa para construir "
    "aristas en esta red ni en sus proyecciones."
)


def _author_node_id(author_channel_id: str) -> str:
    return f"{AUTHOR_PREFIX}{author_channel_id}"


def _video_node_id(video_id: str) -> str:
    return f"{VIDEO_PREFIX}{video_id}"


# ---------------------------------------------------------------------------
# 4.1 / 4.2 / 4.3 - construccion de la red bipartita
# ---------------------------------------------------------------------------

def build_bipartite_graph(comments_clean: pd.DataFrame, videos_clean: pd.DataFrame) -> nx.Graph:
    """
    Red bipartita no dirigida: un conjunto de nodos = autores, el otro = videos.

    Solo se incluyen los videos que SI tienen comentarios recolectados (19 segun
    el contrato de datos actual) y los autores que efectivamente comentaron
    (332). Los otros 274 videos del dataset no tienen actividad observada en
    comments_clean y por lo tanto no son parte de esta red de participacion; no
    se agregan como nodos aislados de grado 0, porque eso confundiria "sin
    cobertura de comentarios" con "aislamiento estructural real" (ver
    build_comment_coverage, que si documenta a esos 274 por separado).

    Arista = par (autor, video) observado en comments_clean.
    Peso (weight_comments) = numero de comment_id de ese autor en ese video.
    reply_count nunca se toca aqui.
    """
    required_cols = {"author_channel_id", "video_id", "comment_id"}
    missing = required_cols - set(comments_clean.columns)
    if missing:
        raise ValueError(f"comments_clean no tiene las columnas requeridas: {missing}")

    G = nx.Graph()

    # --- nodos de video: solo los que tienen comentarios recolectados ---
    commented_video_ids = sorted(comments_clean["video_id"].dropna().unique().tolist())
    comment_counts_by_video = comments_clean.groupby("video_id")["comment_id"].count()

    videos_indexed = videos_clean.set_index("video_id")
    for video_id in commented_video_ids:
        node_id = _video_node_id(video_id)
        attrs = {
            "node_type": "video",
            "video_id": video_id,
            "label": None,
            "title": None,
            "channel_id": None,
            "channel_name": None,
            "category": None,
            "view_count": None,
            "comment_count": int(comment_counts_by_video.get(video_id, 0)),
        }
        if video_id in videos_indexed.index:
            row = videos_indexed.loc[video_id]
            attrs.update({
                "label": row.get("title"),
                "title": row.get("title"),
                "channel_id": row.get("channel_id"),
                "channel_name": row.get("channel_name"),
                "category": row.get("category"),
                "view_count": row.get("view_count"),
            })
        else:
            # no deberia pasar con el contrato actual (406/406 join), pero si un
            # video_id de comments no aparece en videos_clean, fallo visible en
            # vez de nodo silenciosamente incompleto.
            raise ValueError(
                f"video_id '{video_id}' aparece en comments_clean pero no en "
                f"videos_clean; revisar integracion antes de construir la red."
            )
        G.add_node(node_id, **attrs)

    # --- nodos de autor ---
    author_ids = sorted(comments_clean["author_channel_id"].dropna().unique().tolist())
    comments_by_author = comments_clean.groupby("author_channel_id")["comment_id"].count()
    videos_by_author = comments_clean.groupby("author_channel_id")["video_id"].nunique()
    # etiqueta: primer author_name observado, ordenando por comment_id para que
    # sea deterministico y reproducible sin importar el orden de lectura del csv
    label_lookup = (
        comments_clean.sort_values("comment_id")
        .drop_duplicates("author_channel_id", keep="first")
        .set_index("author_channel_id")["author_name"]
    )
    for author_id in author_ids:
        node_id = _author_node_id(author_id)
        G.add_node(
            node_id,
            node_type="author",
            author_channel_id=author_id,
            label=label_lookup.get(author_id),
            comments=int(comments_by_author.get(author_id, 0)),
            videos_distinct=int(videos_by_author.get(author_id, 0)),
        )

    # --- aristas: una fila por par (autor, video), peso = num. comentarios ---
    edge_weights = (
        comments_clean.groupby(["author_channel_id", "video_id"])["comment_id"]
        .count()
        .reset_index(name="weight_comments")
    )
    for row in edge_weights.itertuples(index=False):
        G.add_edge(
            _author_node_id(row.author_channel_id),
            _video_node_id(row.video_id),
            weight_comments=int(row.weight_comments),
        )

    return G


def get_author_nodes(G: nx.Graph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("node_type") == "author"]


def get_video_nodes(G: nx.Graph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("node_type") == "video"]


def build_bipartite_nodes_table(G: nx.Graph) -> pd.DataFrame:
    """
    Tabla de nodos con tipo y atributos relevantes.
    Videos: title/channel/category/views/comments. Autores: label/comments/videos_distinct.
    Las columnas que no aplican a un tipo de nodo quedan NaN para ese tipo.
    """
    rows = []
    for node_id, attrs in G.nodes(data=True):
        rows.append({
            "node_id": node_id,
            "node_type": attrs.get("node_type"),
            "label": attrs.get("label"),
            "video_id": attrs.get("video_id"),
            "title": attrs.get("title"),
            "channel_id": attrs.get("channel_id"),
            "channel_name": attrs.get("channel_name"),
            "category": attrs.get("category"),
            "view_count": attrs.get("view_count"),
            "author_channel_id": attrs.get("author_channel_id"),
            "comments": attrs.get("comments", attrs.get("comment_count")),
            "videos_distinct": attrs.get("videos_distinct"),
            "degree": G.degree(node_id),
        })
    return pd.DataFrame(rows).sort_values(["node_type", "node_id"]).reset_index(drop=True)


def build_bipartite_edges_table(G: nx.Graph) -> pd.DataFrame:
    """Tabla de aristas: un renglon por par (autor, video) con su peso en comentarios."""
    rows = []
    for u, v, attrs in G.edges(data=True):
        if G.nodes[u]["node_type"] == "author":
            author_node, video_node = u, v
        else:
            author_node, video_node = v, u
        rows.append({
            "author_node_id": author_node,
            "video_node_id": video_node,
            "author_channel_id": G.nodes[author_node]["author_channel_id"],
            "video_id": G.nodes[video_node]["video_id"],
            "weight_comments": attrs["weight_comments"],
        })
    return pd.DataFrame(rows).sort_values(["video_node_id", "author_node_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.5 - significado de la arista (texto reutilizable en el informe)
# ---------------------------------------------------------------------------

def edge_meaning_text() -> str:
    return EDGE_MEANING


# ---------------------------------------------------------------------------
# 4.4 - visualizacion de la red completa (no se filtran nodos/aristas)
# ---------------------------------------------------------------------------

def draw_full_bipartite(G: nx.Graph, output_path: Path) -> None:
    """
    Dibuja TODOS los nodos y TODAS las aristas de la red bipartita observada
    (332 autores + 19 videos segun el contrato actual). Solo se limitan
    etiquetas (las de autor, por volumen), nunca se eliminan nodos ni aristas
    para mejorar la estetica.
    """
    video_nodes = get_video_nodes(G)
    author_nodes = get_author_nodes(G)

    # posiciones manuales: autores en una fila superior muy ancha, videos en
    # una fila inferior con mas espacio horizontal entre si (bipartite_layout
    # los deja pegados porque hay 332 autores contra solo 19 videos; con
    # posiciones parejas las 19 etiquetas de video quedan encimadas e
    # ilegibles, asi que se espacian manualmente sin mover la topologia).
    n_authors, n_videos = len(author_nodes), len(video_nodes)
    pos = {}
    for i, n in enumerate(sorted(author_nodes)):
        pos[n] = ((i + 0.5) / n_authors, 1.0)
    for i, n in enumerate(sorted(video_nodes)):
        pos[n] = ((i + 0.5) / n_videos, 0.0)

    fig, ax = plt.subplots(figsize=(20, 11))
    nx.draw_networkx_edges(G, pos, alpha=0.12, width=0.6, edge_color="#888888", ax=ax)
    nx.draw_networkx_nodes(
        G, pos, nodelist=author_nodes, node_size=22, node_color="#3f7d9f",
        alpha=0.85, label=f"Autores ({len(author_nodes)})", ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=video_nodes, node_size=280, node_color="#b5533f",
        alpha=0.95, label=f"Videos ({len(video_nodes)})", ax=ax,
    )
    # etiquetas solo para videos (19), nunca para los 332 autores: ilegible y
    # no aporta -- pero ningun nodo/arista de autor se elimina del dibujo.
    # se alternan dos alturas para que las 19 etiquetas no se encimen.
    for i, n in enumerate(sorted(video_nodes)):
        title = str(G.nodes[n].get("title") or G.nodes[n].get("video_id"))
        label = title if len(title) <= 34 else title[:31] + "..."
        y_offset = -0.09 if i % 2 == 0 else -0.16
        x, y = pos[n]
        ax.annotate(
            label, xy=(x, y), xytext=(x, y + y_offset),
            ha="center", va="top", fontsize=7.2, rotation=20,
            annotation_clip=False,
        )

    ax.set_title(
        f"Red bipartita autor-video completa "
        f"({len(author_nodes)} autores, {len(video_nodes)} videos, {G.number_of_edges()} aristas)",
        pad=15,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    ax.set_ylim(-0.35, 1.1)
    ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# funciones de proyeccion reutilizables (soporte interno del avance para 3.5;
# la entrega/analisis formal es la actividad 5 del domingo)
# ---------------------------------------------------------------------------

def build_author_projection(G: nx.Graph) -> nx.Graph:
    """
    Proyeccion autor-autor: dos autores se conectan si comentaron en el mismo
    video. Peso = numero de videos compartidos.

    Se construye desde incidencia binaria (weighted_projected_graph cuenta
    vecinos compartidos en la estructura del grafo, sin mirar weight_comments),
    tal como exige el plan: comentar 5 veces en el mismo video sigue contando
    como 1 video compartido, no como 5.
    """
    author_nodes = get_author_nodes(G)
    return bipartite.weighted_projected_graph(G, author_nodes)


def build_video_projection(G: nx.Graph) -> nx.Graph:
    """
    Proyeccion video-video: dos videos se conectan si comparten al menos un
    autor. Peso = numero de autores compartidos. Misma logica de incidencia
    binaria que build_author_projection.
    """
    video_nodes = get_video_nodes(G)
    return bipartite.weighted_projected_graph(G, video_nodes)


def build_projection_edges_table(P: nx.Graph, id_attr: str) -> pd.DataFrame:
    """
    Tabla de aristas de una proyeccion (autor-autor o video-video).
    id_attr: 'author_channel_id' o 'video_id', segun cual proyeccion sea, para
    exponer el identificador de negocio ademas del node_id prefijado.
    """
    rows = []
    for u, v, attrs in P.edges(data=True):
        rows.append({
            "node_id_1": u,
            "node_id_2": v,
            id_attr + "_1": P.nodes[u].get(id_attr, u.split(":", 1)[-1]),
            id_attr + "_2": P.nodes[v].get(id_attr, v.split(":", 1)[-1]),
            "weight": attrs["weight"],
        })
    return pd.DataFrame(rows).sort_values(["node_id_1", "node_id_2"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# cobertura de comentarios: distingue "sin cobertura" de "aislado"
# ---------------------------------------------------------------------------

def build_comment_coverage(videos_clean: pd.DataFrame, comments_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por cada uno de los 293 videos del dataset. has_comments_collected
    indica si aparece en comments_clean. Los 274 en False son videos "sin
    comentarios recolectados" (limitacion de cobertura de la recoleccion), NO
    "nodos aislados" de la red -- esos 274 ni siquiera son parte del grafo
    bipartita (ver build_bipartite_graph). Solo los 19 con comments_collected
    son parte de la red y podrian, en principio, resultar perifericos o no
    dentro de esa red observada (eso se analiza en la actividad 6, no aqui).
    """
    comment_counts = comments_clean.groupby("video_id")["comment_id"].count()
    commented_ids = set(comment_counts.index)

    rows = []
    for row in videos_clean.itertuples(index=False):
        video_id = row.video_id
        has_comments = video_id in commented_ids
        rows.append({
            "video_id": video_id,
            "title": row.title,
            "channel_id": row.channel_id,
            "channel_name": row.channel_name,
            "category": row.category,
            "view_count": row.view_count,
            "comment_count_observed": int(comment_counts.get(video_id, 0)),
            "has_comments_collected": has_comments,
            "coverage_status": "participacion_observada" if has_comments else "sin_comentarios_recolectados",
        })
    out = pd.DataFrame(rows)
    assert out["has_comments_collected"].sum() == len(commented_ids)
    return out


# ---------------------------------------------------------------------------
# orquestacion de la etapa "bipartite" (llamada desde main.py)
# ---------------------------------------------------------------------------

def run_bipartite_stage(save_projections: bool = True) -> dict:
    """
    Punto de entrada de la etapa 'bipartite' del pipeline. Lee los datasets
    limpios de H1, construye la red bipartita, las tablas de nodos/aristas, la
    cobertura de comentarios, la figura de la red completa y, opcionalmente,
    las proyecciones preliminares (soporte para 3.5 de Persona 2; el analisis
    formal de proyecciones es la actividad 5 del domingo).
    """
    videos_clean = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)
    comments_clean = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv", encoding=config.CSV_ENCODING)

    G = build_bipartite_graph(comments_clean, videos_clean)
    nodes_table = build_bipartite_nodes_table(G)
    edges_table = build_bipartite_edges_table(G)
    coverage = build_comment_coverage(videos_clean, comments_clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    nodes_table.to_csv(config.TABLES_DIR / "bipartite_nodes.csv", index=False, encoding=config.CSV_ENCODING)
    edges_table.to_csv(config.TABLES_DIR / "bipartite_edges.csv", index=False, encoding=config.CSV_ENCODING)
    coverage.to_csv(config.TABLES_DIR / "comment_coverage.csv", index=False, encoding=config.CSV_ENCODING)
    draw_full_bipartite(G, config.FIGURES_DIR / "fig_bipartite_full.png")

    result = {
        "graph": G,
        "nodes_table": nodes_table,
        "edges_table": edges_table,
        "coverage": coverage,
    }

    if save_projections:
        author_proj = build_author_projection(G)
        video_proj = build_video_projection(G)
        author_edges = build_projection_edges_table(author_proj, "author_channel_id")
        video_edges = build_projection_edges_table(video_proj, "video_id")
        # preliminares: soporte interno del avance (3.5). la entrega formal
        # (actividad 5, con figuras y discusion) se produce el domingo.
        author_edges.to_csv(
            config.TABLES_DIR / "author_projection_edges_preliminary.csv",
            index=False, encoding=config.CSV_ENCODING,
        )
        video_edges.to_csv(
            config.TABLES_DIR / "video_projection_edges_preliminary.csv",
            index=False, encoding=config.CSV_ENCODING,
        )
        result["author_projection"] = author_proj
        result["video_projection"] = video_proj

    print(
        f"[networks] bipartita: {G.number_of_nodes()} nodos "
        f"({len(get_author_nodes(G))} autores, {len(get_video_nodes(G))} videos), "
        f"{G.number_of_edges()} aristas"
    )
    return result


if __name__ == "__main__":
    run_bipartite_stage()