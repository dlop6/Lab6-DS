"""
pruebas minimas de contrato para persona 1. cada test carga los datos crudos
por su cuenta, no depende de que main.py ya haya corrido antes.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, io_data, networks
from src import eda


def test_raw_videos_shape():
    videos = io_data.load_videos()
    assert videos.shape == config.EXPECTED_VIDEOS_SHAPE, (
        f"shape actual {videos.shape} no coincide con el contrato {config.EXPECTED_VIDEOS_SHAPE}"
    )


def test_raw_comments_shape():
    comments = io_data.load_comments()
    assert comments.shape == config.EXPECTED_COMMENTS_SHAPE, (
        f"shape actual {comments.shape} no coincide con el contrato {config.EXPECTED_COMMENTS_SHAPE}"
    )


def test_video_id_unique():
    videos = io_data.load_videos()
    n_unique = videos["video_id"].nunique()
    assert n_unique == len(videos), (
        f"video_id deberia ser unico, hay {len(videos)} filas pero {n_unique} valores unicos"
    )


def test_comment_id_unique():
    comments = io_data.load_comments()
    n_unique = comments["comment_id"].nunique()
    assert n_unique == len(comments), (
        f"comment_id deberia ser unico, hay {len(comments)} filas pero {n_unique} valores unicos"
    )


def test_critical_ids_not_null():
    videos, comments = io_data.load_all()
    checks = {
        "videos.video_id": videos["video_id"],
        "videos.channel_id": videos["channel_id"],
        "comments.video_id": comments["video_id"],
        "comments.comment_id": comments["comment_id"],
        "comments.channel_id": comments["channel_id"],
        "comments.author_channel_id": comments["author_channel_id"],
    }
    for name, series in checks.items():
        n_nulls = int(series.isna().sum())
        assert n_nulls == 0, f"{name} tiene {n_nulls} nulos, deberia tener 0"


def test_join_coverage_406_406():
    videos, comments = io_data.load_all()
    _, join_audit = io_data.integrate_comments_videos(videos, comments)
    row = join_audit.iloc[0]
    assert row["passed"], (
        f"join incompleto: {row['matched_both']}/{row['total_comments']} "
        f"(coverage {row['coverage_pct']}%)"
    )
    assert row["coverage_pct"] == 100.0, f"coverage_pct actual {row['coverage_pct']}, se esperaba 100.0"


def test_commented_videos_count():
    comments = io_data.load_comments()
    n_unique_videos = comments["video_id"].nunique()
    assert n_unique_videos == config.EXPECTED_COMMENTED_VIDEOS, (
        f"hay {n_unique_videos} video_id unicos en comments, "
        f"se esperaban {config.EXPECTED_COMMENTED_VIDEOS}"
    )


def test_persona2_video_participation_contract():
    import pandas as pd
    videos = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)
    comments = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv", encoding=config.CSV_ENCODING)
    result = eda.build_video_participation(videos, comments)
    assert len(result) == config.EXPECTED_COMMENTED_VIDEOS
    assert result["comment_count"].sum() == config.EXPECTED_COMMENTS_SHAPE[0]
    assert result["video_id"].is_unique


def test_persona2_concentration_denominator():
    import pandas as pd
    videos = pd.read_csv(config.PROCESSED_DIR / "videos_clean.csv", encoding=config.CSV_ENCODING)
    comments = pd.read_csv(config.PROCESSED_DIR / "comments_clean.csv", encoding=config.CSV_ENCODING)
    concentration = eda.build_concentration(eda.build_video_participation(videos, comments))
    assert set(concentration["top_n_requested"]) == {5, 10}
    assert (concentration["total_comments"] == config.EXPECTED_COMMENTS_SHAPE[0]).all()
    assert concentration["share_pct"].between(0, 100).all()



# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_videos() -> pd.DataFrame:
    return pd.read_csv(config.RAW_VIDEOS_PATH, encoding=config.CSV_ENCODING)


@pytest.fixture(scope="module")
def raw_comments() -> pd.DataFrame:
    return pd.read_csv(config.RAW_COMMENTS_PATH, encoding=config.CSV_ENCODING)


@pytest.fixture(scope="module")
def videos_clean() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "videos_clean.csv"
    if not path.exists():
        pytest.skip("videos_clean.csv no existe todavia; corre la etapa 'data' primero.")
    return pd.read_csv(path, encoding=config.CSV_ENCODING)


@pytest.fixture(scope="module")
def comments_clean() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "comments_clean.csv"
    if not path.exists():
        pytest.skip("comments_clean.csv no existe todavia; corre la etapa 'data' primero.")
    return pd.read_csv(path, encoding=config.CSV_ENCODING)


@pytest.fixture(scope="module")
def bipartite_graph(videos_clean, comments_clean):
    return networks.build_bipartite_graph(comments_clean, videos_clean)


@pytest.fixture(scope="module")
def bipartite_edges_table(bipartite_graph):
    return networks.build_bipartite_edges_table(bipartite_graph)


@pytest.fixture(scope="module")
def author_projection(bipartite_graph):
    return networks.build_author_projection(bipartite_graph)


@pytest.fixture(scope="module")
def video_projection(bipartite_graph):
    return networks.build_video_projection(bipartite_graph)


# ---------------------------------------------------------------------------
# RAW_VIDEOS / RAW_COMMENTS
# ---------------------------------------------------------------------------

def test_raw_videos_shape(raw_videos):
    assert raw_videos.shape == config.EXPECTED_VIDEOS_SHAPE, (
        f"youtube_videos.csv tiene shape {raw_videos.shape}, se esperaba "
        f"{config.EXPECTED_VIDEOS_SHAPE}. Si el dataset cambio de verdad, "
        f"auditar antes de actualizar este numero."
    )


def test_raw_comments_shape(raw_comments):
    assert raw_comments.shape == config.EXPECTED_COMMENTS_SHAPE, (
        f"youtube_comments.csv tiene shape {raw_comments.shape}, se esperaba "
        f"{config.EXPECTED_COMMENTS_SHAPE}."
    )


# ---------------------------------------------------------------------------
# PK_UNIQUE
# ---------------------------------------------------------------------------

def test_video_id_is_primary_key(raw_videos):
    assert raw_videos["video_id"].isna().sum() == 0
    assert raw_videos["video_id"].nunique() == len(raw_videos)


def test_comment_id_is_primary_key(raw_comments):
    assert raw_comments["comment_id"].isna().sum() == 0
    assert raw_comments["comment_id"].nunique() == len(raw_comments)


# ---------------------------------------------------------------------------
# CRITICAL_IDS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", ["video_id", "channel_id"])
def test_critical_ids_not_null_videos(raw_videos, col):
    assert raw_videos[col].isna().sum() == 0, f"videos.{col} tiene nulos"


@pytest.mark.parametrize("col", ["video_id", "comment_id", "channel_id", "author_channel_id"])
def test_critical_ids_not_null_comments(raw_comments, col):
    assert raw_comments[col].isna().sum() == 0, f"comments.{col} tiene nulos"


# ---------------------------------------------------------------------------
# JOIN_COVERAGE
# ---------------------------------------------------------------------------

def test_join_coverage_406_of_406(raw_videos, raw_comments):
    merged = pd.merge(
        raw_comments, raw_videos, on="video_id", how="left",
        validate="many_to_one", indicator=True,
    )
    matched = int((merged["_merge"] == "both").sum())
    assert matched == len(raw_comments) == 406, (
        f"join incompleto: {matched}/{len(raw_comments)} comentarios asociados a un video valido."
    )


# ---------------------------------------------------------------------------
# COMMENTED_VIDEOS
# ---------------------------------------------------------------------------

def test_commented_videos_count(raw_comments):
    assert raw_comments["video_id"].nunique() == config.EXPECTED_COMMENTED_VIDEOS


def test_authors_observed_count(raw_comments):
    # no es una prueba obligatoria por nombre en la seccion 10, pero protege
    # el numero de autores (332) que usan H3/H4/H5/H6 como referencia comun.
    assert raw_comments["author_channel_id"].nunique() == 332


# ---------------------------------------------------------------------------
# BIPARTITE_EDGE_UNIQUE
# ---------------------------------------------------------------------------

def test_bipartite_edge_unique(bipartite_edges_table):
    pairs = bipartite_edges_table[["author_channel_id", "video_id"]]
    assert pairs.duplicated().sum() == 0, (
        "hay pares (autor, video) duplicados en bipartite_edges; debe haber "
        "un solo renglon por par, con el conteo acumulado en weight_comments."
    )


def test_bipartite_node_counts(bipartite_graph):
    authors = networks.get_author_nodes(bipartite_graph)
    videos = networks.get_video_nodes(bipartite_graph)
    assert len(authors) == 332
    assert len(videos) == config.EXPECTED_COMMENTED_VIDEOS
    # los 274 videos sin comentarios recolectados nunca deben aparecer como
    # nodos aislados de esta red (ver build_comment_coverage para esos 274).
    assert bipartite_graph.number_of_nodes() == 332 + config.EXPECTED_COMMENTED_VIDEOS


# ---------------------------------------------------------------------------
# BIPARTITE_WEIGHT
# ---------------------------------------------------------------------------

def test_bipartite_weight_matches_comment_count(bipartite_edges_table, comments_clean):
    expected = (
        comments_clean.groupby(["author_channel_id", "video_id"])["comment_id"]
        .count().reset_index(name="expected_weight")
    )
    check = bipartite_edges_table.merge(
        expected, on=["author_channel_id", "video_id"], how="outer", indicator=True,
    )
    assert (check["_merge"] == "both").all(), "faltan pares o sobran pares vs. comments_clean"
    assert (check["weight_comments"] == check["expected_weight"]).all(), (
        "weight_comments no coincide con el conteo real de comment_id para ese par autor-video"
    )
    assert bipartite_edges_table["weight_comments"].sum() == len(comments_clean), (
        "la suma de todos los pesos debe ser igual al total de comentarios (406), "
        "porque cada comentario pertenece a exactamente un par autor-video."
    )


# ---------------------------------------------------------------------------
# PROJ_AUTHOR_WEIGHT / PROJ_VIDEO_WEIGHT
# ---------------------------------------------------------------------------

def test_author_projection_weight_is_shared_videos(bipartite_graph, author_projection, comments_clean):
    """Peso autor-autor = numero de videos compartidos (incidencia binaria,
    NO suma de comentarios: comentar 5 veces en el mismo video sigue
    contando como 1 video compartido)."""
    videos_by_author = comments_clean.groupby("author_channel_id")["video_id"].apply(set)
    sample_edges = list(author_projection.edges(data=True))[:25]
    assert sample_edges, "la proyeccion autor-autor no tiene aristas para verificar"
    for u, v, attrs in sample_edges:
        author_u = bipartite_graph.nodes[u]["author_channel_id"]
        author_v = bipartite_graph.nodes[v]["author_channel_id"]
        expected = len(videos_by_author[author_u] & videos_by_author[author_v])
        assert attrs["weight"] == expected, (
            f"peso autor-autor incorrecto para ({author_u}, {author_v}): "
            f"esperado {expected} videos compartidos, obtenido {attrs['weight']}"
        )


def test_video_projection_weight_is_shared_authors(bipartite_graph, video_projection, comments_clean):
    """Peso video-video = numero de autores compartidos."""
    authors_by_video = comments_clean.groupby("video_id")["author_channel_id"].apply(set)
    edges = list(video_projection.edges(data=True))
    assert edges, "la proyeccion video-video no tiene aristas para verificar"
    for u, v, attrs in edges:
        video_u = bipartite_graph.nodes[u]["video_id"]
        video_v = bipartite_graph.nodes[v]["video_id"]
        expected = len(authors_by_video[video_u] & authors_by_video[video_v])
        assert attrs["weight"] == expected, (
            f"peso video-video incorrecto para ({video_u}, {video_v}): "
            f"esperado {expected} autores compartidos, obtenido {attrs['weight']}"
        )


# ---------------------------------------------------------------------------
# NO_REPLY_EDGES
# ---------------------------------------------------------------------------

def test_reply_count_never_used_for_edges():
    """reply_count no debe aparecer como codigo ejecutable en
    src/networks.py (chequeo estatico). El nombre SI puede aparecer en
    docstrings/comentarios explicando la regla (ese es justamente el
    proposito de documentarla), asi que se ignoran ambos antes de revisar."""
    import ast

    source = Path(config.ROOT_DIR / "src" / "networks.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # se neutralizan docstrings (module/funcion/clase) y la constante de
    # prosa EDGE_MEANING, que documenta la regla en lenguaje natural para el
    # informe. Cualquier OTRO uso de "reply_count" (columna de dataframe,
    # atributo, clave de dict) si debe seguir siendo detectado.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0] = ast.Pass()
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EDGE_MEANING" for t in node.targets
        ):
            node.value = ast.Constant(value="")
    code_without_docstrings = ast.unparse(tree)

    assert "reply_count" not in code_without_docstrings, (
        "src/networks.py usa reply_count en codigo ejecutable; esta variable "
        "nunca debe participar en la construccion de aristas ni proyecciones "
        "(solo puede mencionarse en comentarios/docstrings explicando la regla)."
    )


def test_reply_count_not_in_edge_weight(bipartite_edges_table):
    assert "reply_count" not in bipartite_edges_table.columns


# ---------------------------------------------------------------------------
# REPRODUCIBLE_COMMUNITY (formal el domingo; se deja lista para cuando
# Persona 2 entregue src/communities.py y community_assignments.csv)
# ---------------------------------------------------------------------------

def test_community_detection_uses_fixed_seed():
    communities_module = config.ROOT_DIR / "src" / "communities.py"
    if not communities_module.exists():
        pytest.skip("src/communities.py aun no existe (actividad 7, entrega del domingo).")
    source = communities_module.read_text(encoding="utf-8")
    assert "seed=42" in source or "seed = 42" in source, (
        "la deteccion de comunidades debe fijar seed=42 para ser reproducible."
    )
    assert 'weight="weight"' in source or "weight='weight'" in source, (
        "la deteccion de comunidades debe usar weight=\"weight\" explicitamente."
    )


# ---------------------------------------------------------------------------
# RAW_IMMUTABLE
# ---------------------------------------------------------------------------

def test_raw_directory_only_has_original_files():
    files = {p.name for p in config.RAW_DIR.iterdir() if p.is_file()}
    assert files == {"youtube_videos.csv", "youtube_comments.csv"}, (
        f"data/raw/ deberia contener solo los 2 csv originales, tiene: {files}. "
        f"Ningun script debe escribir en data/raw/."
    )
