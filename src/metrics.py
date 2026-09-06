"""
Actividad 6: topologia y fragmentacion. Analiza la bipartita y las
proyecciones que ya entrega src/networks.py -- este modulo no reconstruye
proyecciones ni recalcula pesos, solo las consume para medir estructura.

Tres redes se analizan siempre en el mismo orden: bipartite, author_projection,
video_projection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, networks

NETWORK_NAMES = ("bipartite", "author_projection", "video_projection")


# ---------------------------------------------------------------------------
# helpers de red
# ---------------------------------------------------------------------------

def _largest_connected_component_subgraph(G: nx.Graph) -> nx.Graph:
    """subgrafo de la componente conexa mas grande (LCC). falla si G no tiene nodos."""
    if G.number_of_nodes() == 0:
        raise ValueError("no se puede sacar la LCC de un grafo sin nodos")
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def _degree_series(G: nx.Graph, nodes: list[str] | None = None) -> pd.Series:
    """grados de los nodos indicados (o todos) como Series indexada por node_id."""
    degrees = dict(G.degree(nodes)) if nodes is not None else dict(G.degree())
    return pd.Series(degrees, name="degree")


# ---------------------------------------------------------------------------
# 6.1 - metricas base por red
# ---------------------------------------------------------------------------

def build_network_metrics_row(G: nx.Graph, network_name: str, is_bipartite: bool = False) -> dict:
    """
    nodos, aristas, densidad, grado medio, componentes y tamano de la LCC para
    una red. si es_bipartita, agrega grado medio separado por tipo de nodo
    (autor vs video), porque mezclar ambos lados en un solo promedio no dice
    nada util quiando los dos conjuntos tienen tamanos tan distintos (332 vs 19).
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2 * n_edges / n_nodes) if n_nodes else np.nan
    n_components = nx.number_connected_components(G) if n_nodes else 0
    lcc_size = len(max(nx.connected_components(G), key=len)) if n_nodes else 0

    row = {
        "network": network_name,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(G),
        "avg_degree": avg_degree,
        "avg_degree_authors": np.nan,
        "avg_degree_videos": np.nan,
        "n_components": n_components,
        "lcc_size": lcc_size,
        "lcc_pct": round(100 * lcc_size / n_nodes, 2) if n_nodes else np.nan,
    }
    if is_bipartite:
        author_nodes = networks.get_author_nodes(G)
        video_nodes = networks.get_video_nodes(G)
        row["avg_degree_authors"] = _degree_series(G, author_nodes).mean() if author_nodes else np.nan
        row["avg_degree_videos"] = _degree_series(G, video_nodes).mean() if video_nodes else np.nan
    return row


def build_network_metrics(bipartite_G: nx.Graph, author_proj: nx.Graph, video_proj: nx.Graph) -> pd.DataFrame:
    """una fila por red: bipartite, author_projection, video_projection."""
    rows = [
        build_network_metrics_row(bipartite_G, "bipartite", is_bipartite=True),
        build_network_metrics_row(author_proj, "author_projection"),
        build_network_metrics_row(video_proj, "video_projection"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6.2 - cohesion (node/edge connectivity en la LCC) y transitividad
# ---------------------------------------------------------------------------

def build_cohesion_row(G: nx.Graph, network_name: str) -> dict:
    """
    node_connectivity / edge_connectivity solo tienen sentido en un grafo
    conexo (networkx lanza NetworkXError si hay mas de una componente), asi
    que siempre se calculan sobre la LCC, nunca sobre la red completa
    desconectada. la fila deja constancia de cuantas componentes tenia la red
    original para que quede claro por que se restringio a la LCC.
    """
    n_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
    lcc = _largest_connected_component_subgraph(G)
    lcc_size = lcc.number_of_nodes()

    node_conn = nx.node_connectivity(lcc) if lcc_size > 1 else np.nan
    edge_conn = nx.edge_connectivity(lcc) if lcc_size > 1 else np.nan

    notes = []
    if n_components > 1:
        notes.append(
            f"la red completa tiene {n_components} componentes; node/edge "
            f"connectivity se calcularon solo sobre la LCC ({lcc_size} nodos), "
            f"porque estas metricas requieren un grafo conexo."
        )
    else:
        notes.append("la red completa ya es conexa (1 sola componente); la LCC es la red entera.")

    return {
        "network": network_name,
        "n_components": n_components,
        "lcc_size": lcc_size,
        "node_connectivity_lcc": node_conn,
        "edge_connectivity_lcc": edge_conn,
        "notes": " ".join(notes),
    }


def _bipartite_transitivity_row(G: nx.Graph) -> dict:
    """
    la transitividad estandar (por triangulos) es estructuralmente 0/no
    interpretable en un grafo bipartito puro, porque no puede haber triangulos
    entre dos conjuntos disjuntos de nodos. se usa el clustering bipartito
    (bipartite.average_clustering) como complemento, calculado por separado
    para el lado autor y el lado video.
    """
    author_nodes = networks.get_author_nodes(G)
    video_nodes = networks.get_video_nodes(G)
    clustering_authors = bipartite.average_clustering(G, nodes=author_nodes, mode="dot") if author_nodes else np.nan
    clustering_videos = bipartite.average_clustering(G, nodes=video_nodes, mode="dot") if video_nodes else np.nan
    return {
        "transitivity": np.nan,
        "bipartite_clustering_authors": clustering_authors,
        "bipartite_clustering_videos": clustering_videos,
        "transitivity_notes": (
            "transitividad por triangulos no es interpretable en una red bipartita "
            "pura (no hay triangulos posibles entre dos conjuntos disjuntos); se "
            "reporta bipartite.average_clustering (mode='dot') por lado como complemento."
        ),
    }


def _unipartite_transitivity_row(G: nx.Graph) -> dict:
    return {
        "transitivity": nx.transitivity(G) if G.number_of_nodes() else np.nan,
        "bipartite_clustering_authors": np.nan,
        "bipartite_clustering_videos": np.nan,
        "transitivity_notes": "transitividad estandar por triangulos, red unipartita.",
    }


def build_cohesion_transitivity(bipartite_G: nx.Graph, author_proj: nx.Graph, video_proj: nx.Graph) -> pd.DataFrame:
    """una fila por red con cohesion (LCC) + transitividad/clustering bipartito."""
    rows = []
    for name, G, is_bip in (
        ("bipartite", bipartite_G, True),
        ("author_projection", author_proj, False),
        ("video_projection", video_proj, False),
    ):
        cohesion = build_cohesion_row(G, name)
        transitivity = _bipartite_transitivity_row(G) if is_bip else _unipartite_transitivity_row(G)
        # las notas de cohesion y de transitividad se combinan en una sola columna
        # de texto para no fragmentar la interpretacion en demasiadas columnas sueltas
        combined_notes = f"{cohesion['notes']} {transitivity.pop('transitivity_notes')}"
        cohesion.pop("notes")
        rows.append({**cohesion, **transitivity, "notes": combined_notes})
    cols = [
        "network", "n_components", "lcc_size", "node_connectivity_lcc",
        "edge_connectivity_lcc", "transitivity", "bipartite_clustering_authors",
        "bipartite_clustering_videos", "notes",
    ]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# 6.3 - periferia y aislados
# ---------------------------------------------------------------------------

def identify_peripheral_nodes(
    G: nx.Graph, network_name: str, node_type_filter: str | None = None
) -> pd.DataFrame:
    """
    periferico = grado <= percentil 10 de su tipo de nodo en su red, o el nodo
    esta fuera de la LCC (componente chica). aislado = grado 0 (caso extremo,
    subconjunto de periferico). node_type_filter permite separar autores de
    videos dentro de la misma red bipartita (percentiles distintos por lado,
    porque mezclar 332 autores con 19 videos en un solo percentil no tiene
    sentido: las escalas de grado son muy distintas entre los dos conjuntos).
    """
    if node_type_filter is not None:
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type_filter]
    else:
        nodes = list(G.nodes())

    if not nodes:
        return pd.DataFrame(columns=[
            "network", "node_id", "node_type", "degree",
            "percentile_10_threshold", "is_peripheral", "is_isolated", "is_outside_lcc",
        ])

    degree = _degree_series(G, nodes)
    threshold = float(np.percentile(degree.values, 10))

    lcc_nodes = set(max(nx.connected_components(G), key=len)) if G.number_of_nodes() else set()

    rows = []
    for node_id, deg in degree.items():
        rows.append({
            "network": network_name,
            "node_id": node_id,
            "node_type": G.nodes[node_id].get("node_type", node_type_filter),
            "degree": int(deg),
            "percentile_10_threshold": threshold,
            "is_peripheral": bool(deg <= threshold),
            "is_isolated": bool(deg == 0),
            "is_outside_lcc": bool(node_id not in lcc_nodes),
        })
    return pd.DataFrame(rows)


def build_peripheral_nodes(bipartite_G: nx.Graph, author_proj: nx.Graph, video_proj: nx.Graph) -> pd.DataFrame:
    """
    concatena periferia/aislados de: autores bipartita, videos bipartita,
    author_projection, video_projection. los 274 videos sin comentarios
    recolectados nunca aparecen aqui, porque ni siquiera son nodos del grafo
    bipartita (build_bipartite_graph solo incluye los 19 videos con
    comentarios observados) -- confundirlos con "aislados" seria un error de
    interpretacion que el plan prohibe explicitamente.
    """
    parts = [
        identify_peripheral_nodes(bipartite_G, "bipartite", node_type_filter="author"),
        identify_peripheral_nodes(bipartite_G, "bipartite", node_type_filter="video"),
        identify_peripheral_nodes(author_proj, "author_projection"),
        identify_peripheral_nodes(video_proj, "video_projection"),
    ]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# orquestacion de la etapa "metrics" (llamada desde main.py)
# ---------------------------------------------------------------------------

def run_metrics_stage() -> dict:
    """
    punto de entrada de la etapa 'metrics' (actividad 6). lee los datasets
    limpios de H1, reconstruye la bipartita y las dos proyecciones llamando a
    src/networks.py (nunca recalcula pesos a mano), y escribe las 3 tablas de
    topologia/fragmentacion.
    """
    videos_clean = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)
    comments_clean = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv", encoding=config.CSV_ENCODING)

    bipartite_G = networks.build_bipartite_graph(comments_clean, videos_clean)
    author_proj = networks.build_author_projection(bipartite_G)
    video_proj = networks.build_video_projection(bipartite_G)

    network_metrics = build_network_metrics(bipartite_G, author_proj, video_proj)
    cohesion_transitivity = build_cohesion_transitivity(bipartite_G, author_proj, video_proj)
    peripheral_nodes = build_peripheral_nodes(bipartite_G, author_proj, video_proj)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    network_metrics.to_csv(config.TABLES_DIR / "network_metrics.csv", index=False, encoding=config.CSV_ENCODING)
    cohesion_transitivity.to_csv(config.TABLES_DIR / "cohesion_transitivity.csv", index=False, encoding=config.CSV_ENCODING)
    peripheral_nodes.to_csv(config.TABLES_DIR / "peripheral_nodes.csv", index=False, encoding=config.CSV_ENCODING)

    print(
        f"[metrics] bipartite: {bipartite_G.number_of_nodes()} nodos, "
        f"{nx.number_connected_components(bipartite_G)} componentes | "
        f"author_projection: {author_proj.number_of_nodes()} nodos, "
        f"{nx.number_connected_components(author_proj)} componentes | "
        f"video_projection: {video_proj.number_of_nodes()} nodos, "
        f"{nx.number_connected_components(video_proj)} componentes"
    )

    return {
        "network_metrics": network_metrics,
        "cohesion_transitivity": cohesion_transitivity,
        "peripheral_nodes": peripheral_nodes,
        "bipartite_graph": bipartite_G,
        "author_projection": author_proj,
        "video_projection": video_proj,
    }


if __name__ == "__main__":
    run_metrics_stage()
