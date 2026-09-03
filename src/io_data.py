"""
carga cruda, integracion (join) y diccionario de datos.
este modulo NUNCA transforma el raw ni escribe nada de vuelta a data/raw.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def load_videos(path: Path = config.RAW_VIDEOS_PATH) -> pd.DataFrame:
    """carga youtube_videos.csv tal cual viene, sin tocar nada."""
    return _load_csv(path)


def load_comments(path: Path = config.RAW_COMMENTS_PATH) -> pd.DataFrame:
    """carga youtube_comments.csv tal cual viene, sin tocar nada."""
    return _load_csv(path)


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=config.CSV_ENCODING)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no se encontro {path}. revisa que el csv este en data/raw/ "
            f"y no se haya movido o renombrado."
        )
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"error de encoding leyendo {path} con {config.CSV_ENCODING}: {e.reason}"
        )


def report_shape(df: pd.DataFrame, name: str, expected_shape: tuple | None = None) -> dict:
    """arma un resumen de shape/columnas y avisa si no coincide con el contrato esperado.
    no lanza excepcion aca, eso lo decide quien valida el contrato (los tests)."""
    info = {"name": name, "shape": df.shape, "columns": list(df.columns)}
    print(f"[io_data] {name}: shape={df.shape}, columnas={len(df.columns)}")
    if expected_shape is not None and df.shape != expected_shape:
        print(
            f"[io_data] AVISO: {name} tiene shape {df.shape}, "
            f"se esperaba {expected_shape}. revisar antes de continuar."
        )
    return info


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """punto de entrada unico para cargar ambos datasets crudos."""
    videos = load_videos()
    comments = load_comments()
    report_shape(videos, "videos", config.EXPECTED_VIDEOS_SHAPE)
    report_shape(comments, "comments", config.EXPECTED_COMMENTS_SHAPE)
    return videos, comments


def integrate_comments_videos(
    videos: pd.DataFrame, comments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    left join comments -> videos por video_id, validate=many_to_one.
    si la cobertura no es total, el pipeline falla aca mismo (fallo visible,
    nunca se sigue con datos incompletos silenciosamente).
    """
    videos_nulls = videos["video_id"].isna().sum()
    comments_nulls = comments["video_id"].isna().sum()
    if videos_nulls or comments_nulls:
        raise ValueError(
            f"no se puede integrar: video_id tiene {videos_nulls} nulos en videos "
            f"y {comments_nulls} nulos en comments. limpia esto antes del join."
        )

    merged = pd.merge(
        comments, videos,
        on="video_id", how="left",
        validate="many_to_one", indicator=True,
    )

    total_comments = len(comments)
    matched_both = int((merged["_merge"] == "both").sum())
    unmatched = total_comments - matched_both

    if matched_both != total_comments:
        raise ValueError(
            f"join incompleto: {matched_both}/{total_comments} comentarios "
            f"asociados a un video. se esperaba {total_comments}/{total_comments}. "
            f"hay {unmatched} comentarios sin video_id valido en videos.csv."
        )

    comments_enriched = merged.rename(columns={"_merge": "join_status"})

    join_audit = pd.DataFrame([{
        "total_comments": total_comments,
        "matched_both": matched_both,
        "unmatched_left_only": unmatched,
        "coverage_pct": round(matched_both / total_comments * 100, 4),
        "unique_videos_in_videos": int(videos["video_id"].nunique()),
        "unique_video_ids_in_comments": int(comments["video_id"].nunique()),
        "videos_with_comments_count": int(comments["video_id"].nunique()),
        "videos_without_comments_count": int(
            videos["video_id"].nunique() - comments["video_id"].nunique()
        ),
        "validate_mode": "many_to_one",
        "join_type": "left",
        "passed": matched_both == total_comments,
    }])

    return comments_enriched, join_audit


# rol semantico curado a mano: el dtype no dice si algo es PK, FK o etiqueta,
# eso viene del enunciado del laboratorio, no se puede inferir del dataframe
_DATA_DICTIONARY_SPEC = [
    # dataset, variable, tipo_observado, rol, unidad_de_observacion, descripcion, usar_en_analisis
    ("videos", "video_id", "texto/identificador", "PK", "video", "identificador unico del video, llave primaria", "si"),
    ("videos", "title", "texto", "variable_texto_principal", "video", "titulo del video, fuente para topicos/palabras", "si"),
    ("videos", "channel_name", "texto/categorica", "etiqueta_visible", "video", "nombre visible del canal, puede repetirse o cambiar", "con_precaucion"),
    ("videos", "channel_id", "texto/identificador", "identificador_secundario", "video", "id unico del canal, preferible a channel_name", "si"),
    ("videos", "source_query", "texto/categorica", "metadato_muestreo", "video", "describe como se busco el video, no el tema definitivo", "con_precaucion"),
    ("videos", "source_group", "categorica", "atributo", "video", "estrategia de busqueda: topic/official_gov/channel", "si"),
    ("videos", "dataset_sources", "texto/lista", "metadato_muestreo", "video", "archivos origen antes de integrar, separados por |", "con_precaucion"),
    ("videos", "channel_handle", "texto/identificador_visible", "etiqueta_visible", "video", "handle del canal, util para mostrar, no como id", "con_precaucion"),
    ("videos", "published_time", "texto/temporal_relativo", "atributo", "video", "tiempo relativo desde publicacion, depende del momento de recoleccion", "con_precaucion"),
    ("videos", "view_count_text", "texto_numerico", "atributo", "video", "vistas en formato texto de youtube, usar view_count para calculos", "con_precaucion"),
    ("videos", "description_snippet", "texto", "atributo", "video", "fragmento de descripcion, puede estar incompleto", "con_precaucion"),
    ("videos", "video_url", "texto/url", "atributo", "video", "url completa, sirve para verificar, no como id", "no"),
    ("videos", "query_hits", "texto/lista", "metadato_muestreo", "video", "consultas que recuperaron el video, requiere parseo a lista", "con_precaucion"),
    ("videos", "keywords", "texto/lista", "atributo", "video", "palabras clave del video, puede venir vacia []", "con_precaucion"),
    ("videos", "description", "texto", "variable_texto_principal", "video", "descripcion completa, fuente para topicos", "si"),
    ("videos", "view_count", "numerica_entera", "atributo", "video", "vistas observadas al recolectar, fuente primaria para popularidad", "si"),
    ("videos", "publish_date", "fecha_hora", "atributo", "video", "fecha/hora de publicacion iso 8601", "si"),
    ("videos", "upload_date", "fecha_hora", "atributo", "video", "coincide 100% con publish_date en este dataset", "no"),
    ("videos", "category", "categorica", "atributo", "video", "categoria asignada por youtube", "si"),
    ("videos", "owner_handle", "texto/identificador_visible", "etiqueta_visible", "video", "coincide 100% con channel_handle en este dataset", "no"),

    ("comments", "video_id", "texto/identificador", "FK", "comentario_principal", "id del video comentado, llave foranea hacia videos", "si"),
    ("comments", "comment_id", "texto/identificador", "PK", "comentario_principal", "identificador unico del comentario, llave primaria", "si"),
    ("comments", "video_title", "texto", "atributo_redundante", "comentario_principal", "redundante, se obtiene via join por video_id", "no"),
    ("comments", "channel_name", "texto/categorica", "etiqueta_visible", "comentario_principal", "canal dueno del video, no del autor del comentario", "con_precaucion"),
    ("comments", "channel_id", "texto/identificador", "identificador_secundario", "comentario_principal", "id del canal dueno del video comentado, distinto de author_channel_id", "si"),
    ("comments", "author_name", "texto", "etiqueta_visible", "comentario_principal", "nombre visible del autor, puede cambiar", "con_precaucion"),
    ("comments", "author_channel_id", "texto/identificador", "identificador_secundario", "comentario_principal", "id del autor del comentario, usar para nodos de red", "si"),
    ("comments", "text", "texto", "variable_texto_principal", "comentario_principal", "contenido del comentario, base para nlp y sentimiento", "si"),
    ("comments", "source_query", "texto/categorica", "metadato_muestreo", "comentario_principal", "consulta que llevo a recolectar este comentario", "con_precaucion"),
    ("comments", "source_group", "categorica", "atributo", "comentario_principal", "tipo de fuente: topic/channel", "si"),
    ("comments", "dataset_sources", "texto/lista", "metadato_muestreo", "comentario_principal", "archivos origen antes de integrar, separados por |", "con_precaucion"),
    ("comments", "author_handle", "texto/identificador_visible", "etiqueta_visible", "comentario_principal", "handle del autor, no usar como id", "con_precaucion"),
    ("comments", "published_text", "texto/temporal_relativo", "atributo", "comentario_principal", "tiempo relativo desde publicacion del comentario", "con_precaucion"),
    ("comments", "like_count_text", "texto_numerico", "atributo", "comentario_principal", "likes en texto, puede venir en blanco, requiere parseo", "con_precaucion"),
    ("comments", "reply_count", "numerica_entera", "atributo", "comentario_principal", "cantidad de respuestas, nunca crear aristas con esto", "con_precaucion"),
    ("comments", "is_pinned", "logica/booleana", "no_usable", "comentario_principal", "constante False en los 406 registros, sin variabilidad", "no"),
    ("comments", "viewer_rating", "numerica_vacia", "no_usable", "comentario_principal", "vacia en el 100% de los registros", "no"),
]


def build_data_dictionary(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """tabla curada a mano: pk/fk, unidad de observacion y rol de cada variable.
    el rol semantico no se puede inferir solo del dtype, viene del enunciado del lab."""
    df = pd.DataFrame(
        _DATA_DICTIONARY_SPEC,
        columns=[
            "dataset", "variable", "tipo_observado", "rol",
            "unidad_de_observacion", "descripcion", "usar_en_analisis",
        ],
    )
    # validacion de que no se nos olvido documentar ninguna columna real
    videos_documented = set(df.loc[df["dataset"] == "videos", "variable"])
    comments_documented = set(df.loc[df["dataset"] == "comments", "variable"])
    missing_videos = set(videos.columns) - videos_documented
    missing_comments = set(comments.columns) - comments_documented
    if missing_videos or missing_comments:
        raise ValueError(
            f"el diccionario de datos no cubre todas las columnas reales. "
            f"faltan en videos: {missing_videos}. faltan en comments: {missing_comments}."
        )
    return df
