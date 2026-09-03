# schema contract - H1 (persona 1 -> persona 2 y 3)

pruebas de contrato (`pytest tests/test_contracts.py`) pasando en 7/7 antes de este documento.

## 1.3 relaciones entre entidades

canal publica video; autor publica comentario en un video; categoria pertenece al video;
source_query describe el muestreo con el que se recolecto el video/comentario, no es el tema
definitivo del contenido.

```
canal (channel_id) --publica--> video (video_id) --pertenece_a--> categoria (category)
video (video_id) <--comentado_en-- comentario (comment_id) --publicado_por--> autor (author_channel_id)
video (video_id) --recuperado_via--> source_query (describe muestreo, no tema definitivo)
```

## columnas criticas y tipos

| dataset | columna | rol | tipo |
|---|---|---|---|
| videos | video_id | PK | texto/identificador |
| videos | channel_id | identificador secundario | texto/identificador |
| comments | comment_id | PK | texto/identificador |
| comments | video_id | FK -> videos.video_id | texto/identificador |
| comments | channel_id | identificador secundario | texto/identificador |
| comments | author_channel_id | identificador secundario (nodo autor) | texto/identificador |

## estado del join (datos actuales)

- `youtube_videos.csv`: 293 filas x 20 columnas, `video_id` unico (293/293).
- `youtube_comments.csv`: 406 filas x 17 columnas, `comment_id` unico (406/406).
- LEFT JOIN comments -> videos por `video_id`, `validate="many_to_one"`: **406/406** comentarios
  asociados a un video valido. cobertura 100%.
- solo **19** `video_id` unicos aparecen en `youtube_comments.csv`. los otros **274** videos no
  tienen comentarios recolectados en este dataset — nunca se les llama "aislados", son
  "videos sin comentarios recolectados" (falta de cobertura, no aislamiento real de la red).
- `author_channel_id`: 332 valores unicos observados en comments.

## csv que produce H1

- `data/processed/videos_clean.csv`
- `data/processed/comments_clean.csv`
- `data/processed/comments_enriched.csv`
- `outputs/tables/data_dictionary.csv`
- `outputs/tables/join_audit.csv`
- `outputs/tables/quality_summary.csv` (+ `quality_duplicates.csv`, `quality_outliers.csv`, `quality_id_consistency.csv`)
- `outputs/tables/variable_treatment.csv`
- `outputs/tables/count_conversion_audit.csv`
- `outputs/tables/text_cleaning_effect.csv`

## regla metodologica critica (aplica a todo el equipo)

`reply_count` solo indica cuantas respuestas recibio un comentario. no identifica autores.
nunca se usa para crear aristas entre usuarios, inferir conversaciones directas, amistad ni
aprobacion. persona 1 no construye aristas, pero deja esta regla documentada para persona 2 y 3.
