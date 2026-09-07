"""
config central del proyecto. rutas, semilla y contratos de datos esperados.
nada de logica de negocio aca, solo constantes que el resto de modulos consume.
"""
from pathlib import Path

# raiz del repo, calculada desde este archivo para no depender del cwd desde
# donde se ejecute el script (evita romper rutas si alguien corre desde otra carpeta)
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
EDA_MIN_CATEGORY_N = 3
SENTIMENT_SMALL_GROUP_N = 5

RAW_VIDEOS_PATH = RAW_DIR / "youtube_videos.csv"
RAW_COMMENTS_PATH = RAW_DIR / "youtube_comments.csv"

SEED = 42
SPACY_MODEL_NAME = "es_core_news_sm"
CSV_ENCODING = "utf-8-sig"  # ambos csv traen bom utf-8, hay que leerlo explicito

# contrato de shapes con los datos actuales. si esto cambia algun dia hay que
# auditar por que, no simplemente actualizar el numero a lo loco
EXPECTED_VIDEOS_SHAPE = (293, 20)
EXPECTED_COMMENTS_SHAPE = (406, 17)
EXPECTED_COMMENTED_VIDEOS = 19  # video_id unicos que aparecen en comments

# columnas identificadoras criticas por dataset. nunca se reemplazan por
# nombres/handles y siempre deben venir sin nulos
VIDEOS_ID_COLUMNS = ["video_id", "channel_id"]
COMMENTS_ID_COLUMNS = ["video_id", "comment_id", "channel_id", "author_channel_id"]

# etiquetas visibles, solo para presentacion, jamas se usan como identificador
VIDEOS_LABEL_COLUMNS = ["channel_name", "channel_handle", "owner_handle"]
COMMENTS_LABEL_COLUMNS = ["author_name", "author_handle"]
COMMUNITY_TOP_N = 3