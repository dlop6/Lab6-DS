# Laboratorio 6 - Análisis de redes sociales en YouTube

CC3084 Data Science, UVG, Semestre II 2026. Equipo de 3 personas.
Avance (actividades 1-4): jueves 3/09/2026. Entrega final (1-10): domingo 6/09/2026, 23:59.

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Estructura

```
lab6/
  data/raw/            youtube_videos.csv, youtube_comments.csv (INMUTABLES)
  data/processed/      videos_clean.csv, comments_clean.csv, comments_enriched.csv
  src/                 modulos por responsabilidad (config, io_data, cleaning, eda, nlp, networks, ...)
  outputs/tables/      csv de resultados (auditorias, EDA, red, proyecciones, ...)
  outputs/figures/     figuras generadas
  tests/               pytest de contratos criticos
  main.py              orquestador por etapas
  schema_contract.md   contrato de datos entregado por Persona 1 (H1)
```

## Cómo ejecutar

Pruebas de contrato primero, siempre:

```bash
pytest tests/test_contracts.py -v
```

Pipeline completo (equivalente a correr las 3 etapas del avance en orden):

```bash
python main.py --stage all
```

O por etapa individual:

```bash
python main.py --stage data        # Persona 1: actividades 1-2 (carga, calidad, limpieza)
python main.py --stage eda         # Persona 2: actividad 3 (EDA). Agregar --sentiment para
                                    #   precomputar sentimiento (descarga el modelo de pysentimiento)
python main.py --stage bipartite   # Persona 3: actividad 4 (red bipartita autor-video)
```

Cada etapa lee lo que la etapa anterior dejó en `data/processed/`, así que para
un run limpio de principio a fin se corre `data` antes que `eda` o `bipartite`.

## Estado del avance (actividades 1-4)

- **Datos**: `youtube_videos.csv` (293×20) y `youtube_comments.csv` (406×17).
  Join comments→videos: **406/406** (100% cobertura). Solo **19** `video_id`
  únicos tienen comentarios recolectados; los otros **274** no tienen
  cobertura de comentarios — nunca se les llama "aislados" (ver
  `outputs/tables/comment_coverage.csv`).
- **Red bipartita autor-video** (actividad 4): **332 autores + 19 videos =
  351 nodos**, **343 aristas**. Cada arista significa "el autor publicó al
  menos un comentario en ese video"; el peso es la cantidad de comentarios de
  ese autor en ese video. `reply_count` nunca participa en la construcción de
  aristas (verificado en `tests/test_contracts.py`). Ver
  `outputs/tables/bipartite_nodes.csv`, `bipartite_edges.csv`,
  `outputs/figures/fig_bipartite_full.png`.
- **Proyecciones preliminares** (soporte interno de la pregunta 3.5, no la
  entrega formal de la actividad 5 que es del domingo):
  `outputs/tables/author_projection_edges_preliminary.csv` y
  `video_projection_edges_preliminary.csv`, generadas con
  `src/networks.build_author_projection()` / `build_video_projection()`.

## Notas de compatibilidad (hallazgo de Persona 3, pendiente de que Persona 1 lo corrija)

Este repo se probó con **pandas 3.0.2**. Con esa versión (y con el backend de
string nativo de pandas 2.x en general), `pandas.read_csv` puede asignar a las
columnas de texto el dtype `str` (`pandas.StringDtype`) en vez de `object`.
Dos funciones de `src/cleaning.py` asumen `dtype == object` y por eso se
comportan distinto de lo esperado en este entorno:

1. **`normalize_ids()` lanza `TypeError`** ("no es tipo texto") aunque la
   columna sí sea de texto y el `.str.strip()` en sí funcione perfectamente.
   Mientras no se corrija, este repo fija `pandas<3.0` en `requirements.txt`
   como mitigación (bajo esa restricción el dtype vuelve a ser `object` y la
   función corre sin tocarla).
2. **`_quality_row()` dentro de `build_quality_summary()` deja de detectar
   mojibake y strings en blanco silenciosamente** (sin excepción) cuando la
   columna no es `object`, porque el chequeo está guardado detrás de
   `if series.dtype == object`. Verificamos manualmente que sí existe 1
   carácter de reemplazo Unicode (mojibake) en `videos.description` — no en
   `published_time`/`query_hits`/`keywords` como indica
   `outputs/tables/variable_treatment.csv` actualmente, así que además de la
   compatibilidad de dtype, vale la pena que Persona 1 re-verifique esa
   atribución de columnas.

Arreglo sugerido (no aplicado aquí, es código de Persona 1):
reemplazar `df[col].dtype != object` por
`not pandas.api.types.is_string_dtype(df[col])`.

## Enlaces del equipo

- Espacio colaborativo: _pendiente, pegar cuando exista_.
- Repositorio (versionado): _pendiente, pegar cuando exista_.