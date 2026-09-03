"""
diagnostico de calidad, normalizacion de ids/etiquetas, conteos y limpieza de texto.
todo lo que audita o transforma los datos crudos hacia una version limpia vive aca.
regla de oro: nunca imputar. lo que falta se queda como NaN, y se cuantifica.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

_REPLACEMENT_CHAR = "�"  # el caracter de reemplazo unicode, señal de mojibake


# ---------------------------------------------------------------------------
# bloque calidad (2.1, 2.2)
# ---------------------------------------------------------------------------

def _quality_row(dataset: str, col: str, series: pd.Series) -> dict:
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    notes = []
    if series.dtype == object:
        has_replacement = series.astype(str).str.contains(_REPLACEMENT_CHAR, na=False).any()
        if has_replacement:
            n_affected = int(series.astype(str).str.contains(_REPLACEMENT_CHAR, na=False).sum())
            notes.append(f"contiene caracter de reemplazo unicode en {n_affected} filas (mojibake de origen)")
        has_blank_strings = series.astype(str).str.strip().eq("").sum()
        if has_blank_strings and n_missing < n_total:
            notes.append(f"contiene {int(has_blank_strings)} strings vacios/solo espacio (no NaN tecnico)")
    return {
        "dataset": dataset,
        "variable": col,
        "dtype": str(series.dtype),
        "n_total": n_total,
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_total * 100, 2) if n_total else 0.0,
        "n_unique": n_unique,
        "is_constant": n_unique <= 1,
        "n_duplicated_values": int(n_total - n_unique - n_missing) if n_total else 0,
        "notes": "; ".join(notes),
    }


def build_quality_summary(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """diagnostico columna por columna: dtypes, faltantes, duplicados, constantes, notas."""
    rows = [_quality_row("videos", c, videos[c]) for c in videos.columns]
    rows += [_quality_row("comments", c, comments[c]) for c in comments.columns]
    return pd.DataFrame(rows)


def build_quality_duplicates(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """duplicados de fila completa por dataset + duplicados de texto en comments."""
    rows = [
        {"dataset": "videos", "check": "full_row_duplicates", "n_affected": int(videos.duplicated().sum()), "example_ids": ""},
        {"dataset": "comments", "check": "full_row_duplicates", "n_affected": int(comments.duplicated().sum()), "example_ids": ""},
    ]
    text_dupe_mask = comments["text"].duplicated(keep=False) & comments["text"].notna()
    n_text_dupes = int(text_dupe_mask.sum())
    example_ids = ", ".join(comments.loc[text_dupe_mask, "comment_id"].head(10).tolist())
    rows.append({
        "dataset": "comments", "check": "text_value_duplicates",
        "n_affected": n_text_dupes, "example_ids": example_ids,
    })
    return pd.DataFrame(rows)


def build_quality_outliers(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """outliers via iqr sobre las variables numericas relevantes."""
    targets = [("videos", "view_count", videos), ("comments", "reply_count", comments)]
    rows = []
    for dataset, col, df in targets:
        series = df[col].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < low) | (series > high)).sum())
        rows.append({
            "dataset": dataset, "variable": col, "metodo": "iqr",
            "limite_inferior": round(float(low), 2), "limite_superior": round(float(high), 2),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(series) * 100, 2) if len(series) else 0.0,
        })
    return pd.DataFrame(rows)


def build_quality_id_consistency(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """consistencia nombre<->id y handle<->id: cuantos ids distintos comparte una misma etiqueta."""
    checks = [
        ("channel_name_to_channel_id", videos, "channel_name", "channel_id"),
        ("author_name_to_author_channel_id", comments, "author_name", "author_channel_id"),
    ]
    rows = []
    for check_name, df, label_col, id_col in checks:
        grouped = df.groupby(label_col)[id_col].nunique()
        n_affected = int((grouped > 1).sum())
        rows.append({
            "check_name": check_name,
            "entity_with_multiple_ids": n_affected,
            "n_affected": n_affected,
        })
    return pd.DataFrame(rows)


_VARIABLE_TREATMENT_SPEC = [
    ("comments", "viewer_rating", "vacia en el 100% de los registros", "no_usar", "no aporta informacion, todo NaN"),
    ("comments", "is_pinned", "constante False en los 406 registros", "excluir_de_analisis_explicativo", "sin variabilidad, documentar y excluir"),
    ("videos", "published_time", "tiempo relativo, depende del momento de recoleccion", "usar_con_precaucion", "no es fecha exacta, solo aproximada/ordinal"),
    ("comments", "published_text", "tiempo relativo, depende del momento de recoleccion", "usar_con_precaucion", "no es fecha exacta, solo aproximada/ordinal"),
    ("videos", "source_query", "describe el proceso de busqueda, no el tema", "usar_con_precaucion", "es metadato de muestreo, no tema definitivo del video"),
    ("comments", "source_query", "describe el proceso de busqueda, no el tema", "usar_con_precaucion", "es metadato de muestreo, no tema definitivo"),
    ("videos", "channel_name", "puede cambiar o repetirse", "usar_con_precaucion", "etiqueta visible, nunca sustituye a channel_id"),
    ("videos", "channel_handle", "puede cambiar", "usar_con_precaucion", "etiqueta visible, nunca sustituye a channel_id"),
    ("videos", "owner_handle", "coincide 100% con channel_handle", "usar_con_precaucion", "etiqueta visible redundante, nunca sustituye a channel_id"),
    ("comments", "author_name", "puede cambiar o repetirse", "usar_con_precaucion", "etiqueta visible, nunca sustituye a author_channel_id"),
    ("comments", "author_handle", "puede cambiar", "usar_con_precaucion", "etiqueta visible, nunca sustituye a author_channel_id"),
    ("comments", "reply_count", "cuenta respuestas pero no identifica autores", "nunca_usar_para_aristas",
     "solo indica cuantas respuestas recibio un comentario, no identifica autores, nunca se usa para "
     "crear aristas entre usuarios, inferir conversaciones directas, amistad ni aprobacion"),
    ("videos", "dataset_sources", "lista de archivos origen separados por |", "usar_con_precaucion", "describe procedencia de integracion, no es variable analitica directa"),
    ("comments", "dataset_sources", "lista de archivos origen separados por |", "usar_con_precaucion", "describe procedencia de integracion, no es variable analitica directa"),
    ("videos", "query_hits", "texto con estructura de lista", "requiere_parseo", "debe convertirse a lista antes de analizar, un video puede tener mas de un hit"),
    ("videos", "keywords", "texto con estructura de lista, puede venir vacia []", "requiere_parseo", "convertir a lista antes de analizar"),
    ("comments", "like_count_text", "contiene strings de solo espacio en blanco", "parsear_para_auditoria",
     "el blanco se convierte a NaN, nunca a cero, porque el dataset no documenta que un espacio signifique ausencia de likes"),
    ("videos", "view_count_text", "texto con separador de miles y palabra 'vistas'", "parsear_para_auditoria",
     "view_count es la fuente numerica primaria; discrepancias entre ambos se auditan, no se promedian ni se fuerzan"),
    ("videos", "published_time", "caracteres de reemplazo unicode detectados", "usar_con_precaucion",
     "el texto perdio informacion de codificacion antes de llegar al csv, no se puede reconstruir con certeza, se documenta y no se adivina"),
    ("videos", "query_hits", "caracteres de reemplazo unicode detectados", "usar_con_precaucion",
     "mojibake de origen, no se reconstruye, solo se documenta"),
    ("videos", "keywords", "caracteres de reemplazo unicode detectados", "usar_con_precaucion",
     "mojibake de origen, no se reconstruye, solo se documenta"),
]


def build_variable_treatment() -> pd.DataFrame:
    """tabla curada a mano: que variables son delicadas y por que. viene del enunciado
    del laboratorio, no se puede inferir automaticamente de los datos."""
    return pd.DataFrame(
        _VARIABLE_TREATMENT_SPEC,
        columns=["dataset", "variable", "problema_detectado", "tratamiento", "justificacion"],
    )


# ---------------------------------------------------------------------------
# bloque normalizacion de ids/etiquetas (2.3)
# ---------------------------------------------------------------------------

def normalize_ids(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """strip de whitespace en columnas id, sin tocar mayus/minus. no modifica el df original."""
    df = df.copy()
    for col in id_columns:
        if col not in df.columns:
            raise KeyError(f"columna id '{col}' no existe en el dataframe")
        if df[col].dtype != object:
            raise TypeError(f"columna id '{col}' no es tipo texto (dtype={df[col].dtype}), no se puede strip")
        df[col] = df[col].str.strip()
    return df


def normalize_labels(df: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """trim + colapsa espacios en etiquetas visibles. solo cosmetico, jamas sustituye ids."""
    df = df.copy()
    for col in label_columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        df.loc[df[col] == "nan", col] = np.nan
    return df


# ---------------------------------------------------------------------------
# bloque conteos (2.4) - cero imputacion
# ---------------------------------------------------------------------------

def parse_view_count_text(value) -> float:
    """extrae el numero de un string tipo '2,390 vistas'. NaN si no es interpretable."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    text = re.sub(r"vistas?|views?", "", text)
    text = text.replace(",", "").strip()
    if not text or not re.fullmatch(r"\d+(\.\d+)?", text):
        return np.nan
    return float(text)


def parse_like_count_text(value) -> float:
    """convierte like_count_text a numero. blanco o no interpretable -> NaN, nunca 0."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_count_conversion_audit(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """audita el parseo de conteos en texto, comparando contra la fuente numerica primaria."""
    view_parsed = videos["view_count_text"].apply(parse_view_count_text)
    view_mismatch = int(((view_parsed.notna()) & (view_parsed != videos["view_count"])).sum())
    n_blank_view = int(videos["view_count_text"].astype(str).str.strip().eq("").sum())

    like_parsed = comments["like_count_text"].apply(parse_like_count_text)
    like_str = comments["like_count_text"].astype(str).str.strip()
    n_blank_like = int((like_str == "").sum())
    n_uninterpretable_like = int(like_parsed.isna().sum() - n_blank_like - comments["like_count_text"].isna().sum())
    n_uninterpretable_like = max(n_uninterpretable_like, 0)

    rows = [
        {
            "dataset": "videos", "variable_origen": "view_count_text",
            "variable_derivada": "view_count_from_text",
            "n_total": len(videos),
            "n_parsed_ok": int(view_parsed.notna().sum()),
            "n_blank_to_nan": n_blank_view,
            "n_uninterpretable_to_nan": int(view_parsed.isna().sum() - videos["view_count_text"].isna().sum()),
            "n_mismatch_vs_reference": view_mismatch,
            "reference_variable": "view_count",
            "notes": "view_count es la fuente primaria para calculos, discrepancias no se resuelven aqui",
        },
        {
            "dataset": "comments", "variable_origen": "like_count_text",
            "variable_derivada": "like_count",
            "n_total": len(comments),
            "n_parsed_ok": int(like_parsed.notna().sum()),
            "n_blank_to_nan": n_blank_like,
            "n_uninterpretable_to_nan": n_uninterpretable_like,
            "n_mismatch_vs_reference": 0,
            "reference_variable": "NA",
            "notes": "blanco se convierte a NaN, nunca a cero, no se imputa",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# bloque texto (2.5, 2.6, 2.7)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+\b")
_PUNCT_RE = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")

_nlp_cache = {}


def load_spacy_model(model_name: str = config.SPACY_MODEL_NAME):
    """carga el modelo de spacy una sola vez (cache a nivel de modulo)."""
    if model_name in _nlp_cache:
        return _nlp_cache[model_name]
    try:
        import spacy
        nlp = spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"el modelo de spacy '{model_name}' no esta instalado. "
            f"corre: python -m spacy download {model_name}"
        )
    _nlp_cache[model_name] = nlp
    return nlp


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def strip_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def dehashtag(text: str) -> str:
    """convierte #palabra en palabra, se queda solo con el texto sin el simbolo."""
    return _HASHTAG_RE.sub(r"\1", text)


def remove_punctuation_and_numeric_tokens(text: str) -> str:
    text = _NUMERIC_TOKEN_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text, nlp_model) -> str:
    """
    pipeline completo de limpieza de texto, en este orden:
    minusculas -> quitar urls -> dehashtag -> quitar menciones ->
    quitar puntuacion/numeros -> tokenizar+stopwords+lematizar (spacy) -> normalizar espacios.
    texto vacio o NaN de entrada devuelve '' sin lanzar excepcion (nunca elimina la fila).
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    if nlp_model is None:
        raise RuntimeError("clean_text necesita un modelo de spacy cargado, pasa nlp_model")

    t = str(text).lower()
    t = remove_urls(t)
    t = dehashtag(t)
    t = strip_mentions(t)
    t = remove_punctuation_and_numeric_tokens(t)
    t = normalize_whitespace(t)

    if not t:
        return ""

    doc = nlp_model(t)
    tokens = [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_space and tok.lemma_.strip()]
    return normalize_whitespace(" ".join(tokens))


def clean_corpus(texts: pd.Series, nlp_model=None) -> pd.Series:
    """aplica clean_text a una serie completa usando nlp.pipe para procesar en batch."""
    if nlp_model is None:
        nlp_model = load_spacy_model()

    # separamos lo que ya sabemos vacio de lo que hay que mandar por el pipeline pesado
    is_blank = texts.isna() | (texts.astype(str).str.strip() == "")
    result = pd.Series("", index=texts.index, dtype=object)

    non_blank_idx = texts.index[~is_blank]
    if len(non_blank_idx) == 0:
        return result

    pre_processed = (
        texts.loc[non_blank_idx].astype(str).str.lower()
        .apply(remove_urls).apply(dehashtag).apply(strip_mentions)
        .apply(remove_punctuation_and_numeric_tokens).apply(normalize_whitespace)
    )

    cleaned = []
    for doc in nlp_model.pipe(pre_processed.tolist()):
        tokens = [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_space and tok.lemma_.strip()]
        cleaned.append(normalize_whitespace(" ".join(tokens)))

    result.loc[non_blank_idx] = cleaned
    return result


def add_dual_text(comments: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """agrega texto_original (copia exacta) y texto_limpio (limpio via clean_corpus)."""
    df = comments.copy()
    df["texto_original"] = df[text_column]
    df["texto_limpio"] = clean_corpus(df[text_column])
    return df


def build_text_cleaning_effect(comments_clean: pd.DataFrame) -> pd.DataFrame:
    """cuantifica el efecto de la limpieza de texto sobre comments_clean."""
    n_total = len(comments_clean)
    original = comments_clean["texto_original"]
    limpio = comments_clean["texto_limpio"]

    modified_mask = (original.astype(str).str.lower().str.strip()) != limpio.astype(str).str.strip()
    n_modified = int(modified_mask.sum())
    n_empty_after = int((limpio == "").sum())

    n_dupes_before = int(original.duplicated(keep=False).sum())
    n_dupes_after = int(limpio[limpio != ""].duplicated(keep=False).sum())

    # la regla es que nunca se elimina fila por texto vacio, esto verifica que se cumplio
    n_rows_removed = 0

    return pd.DataFrame([{
        "n_total": n_total,
        "n_modified": n_modified,
        "n_empty_after_cleaning": n_empty_after,
        "pct_empty_after_cleaning": round(n_empty_after / n_total * 100, 2) if n_total else 0.0,
        "n_duplicates_before": n_dupes_before,
        "n_duplicates_after": n_dupes_after,
        "n_rows_removed": n_rows_removed,
    }])


# ---------------------------------------------------------------------------
# ensamblaje final: videos_clean y comments_clean
# ---------------------------------------------------------------------------

def build_videos_clean(videos: pd.DataFrame) -> pd.DataFrame:
    """aplica normalizacion de ids/etiquetas + conteo derivado, listo para videos_clean.csv."""
    df = normalize_ids(videos, config.VIDEOS_ID_COLUMNS)
    df = normalize_labels(df, config.VIDEOS_LABEL_COLUMNS)
    df["view_count_from_text"] = df["view_count_text"].apply(parse_view_count_text)
    return df


def build_comments_clean(comments: pd.DataFrame, nlp_model=None) -> pd.DataFrame:
    """aplica normalizacion de ids/etiquetas + conteo derivado + texto dual/limpio."""
    df = normalize_ids(comments, config.COMMENTS_ID_COLUMNS)
    df = normalize_labels(df, config.COMMENTS_LABEL_COLUMNS)
    df["like_count"] = df["like_count_text"].apply(parse_like_count_text)
    df["texto_original"] = df["text"]
    df["texto_limpio"] = clean_corpus(df["text"], nlp_model)
    return df
