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

## 6. topologia y fragmentacion (entrega final, H5)

analiza la bipartita autor-video y las dos proyecciones (autor-autor, video-video) que ya
entrega `src/networks.py`. no se reconstruye ninguna proyeccion aca, solo se consumen para
medir estructura. codigo en `src/metrics.py`, etapa `python main.py --stage metrics`.

### 6.1 metricas base (`network_metrics.csv`)

las tres redes tienen **10 componentes conexos** cada una — el mismo numero en las tres,
porque las proyecciones se derivan de la misma bipartita y heredan su fragmentacion. la
componente mas grande (LCC) cubre el 81% de los nodos en la bipartita (286/351) y el 83% en
author_projection (276/332), pero solo el 53% en video_projection (10/19): casi la mitad de
los videos con comentarios recolectados no comparte ningun autor con el resto de la red.

la bipartita es muy dispersa (densidad 0.0056, esperable: son solo 343 aristas contra 351
nodos posibles). la proyeccion autor-autor es mucho mas densa (0.195) porque agrupa a
cualquier par de autores que haya comentado en el mismo video, incluso si esa coincidencia es
minima. el grado medio separado por tipo de nodo en la bipartita confirma la asimetria: los
autores tienen grado medio ~1.03 (la mayoria comento en un solo video), los videos tienen
grado medio ~18.05 (cada video con comentarios recolectados concentra a muchos autores).

### 6.2 cohesion y transitividad (`cohesion_transitivity.csv`)

como las tres redes estan fragmentadas en 10 componentes, `node_connectivity`/`edge_connectivity`
se calcularon siempre sobre la componente mas grande (LCC), nunca sobre la red completa
desconectada (networkx lo exige: estas metricas fallan con un error si el grafo no es conexo).
en las tres LCC, la conectividad de nodos es 1 — basta con remover un solo nodo puente para
fragmentar aun mas esa componente, asi que ninguna de las tres redes es estructuralmente
robusta a la perdida de nodos clave. la conectividad de aristas es mas alta en author_projection
(5) que en las otras dos (1), es decir, la componente principal de autores tiene mas caminos
alternativos entre sus miembros.

la transitividad estandar (por triangulos) no es interpretable en la bipartita porque una red
bipartita pura no puede tener triangulos entre dos conjuntos disjuntos; ahi se reporta
`bipartite.average_clustering` por lado en su lugar: 0.94 del lado autor (muy alto, casi
cualquier par de autores conectados al mismo video termina con vecinos en comun) y 0.014 del
lado video (casi nulo, los videos rara vez comparten "vecinos de vecinos" mas alla de sus
autores directos). en las proyecciones si aplica transitividad estandar: 0.98 en
author_projection (grupos de autores muy entrelazados, coherente con la alta densidad) y 0.32
en video_projection (menos consistente, hay menos triangulos de videos que comparten autores
entre si).

### 6.3 periferia y aislados (`peripheral_nodes.csv`)

**aclaracion obligatoria:** los 274 videos sin comentarios recolectados (de los 293 totales)
**nunca aparecen en esta tabla**, porque ni siquiera son nodos del grafo bipartita —
`build_bipartite_graph` solo incluye los 19 videos que si tienen comentarios observados. la
tabla de periferia solo describe aislamiento *dentro* de la red que si se pudo construir, no
ausencia de datos de recoleccion (esa distincion vive en `comment_coverage.csv`, generado por
persona 3 en la etapa `bipartite`).

el criterio usado es cuantitativo: periferico = grado <= percentil 10 de su tipo de nodo en su
red, mas cualquier nodo fuera de la LCC; aislado = grado 0 (subconjunto de periferico). un
hallazgo notable: en el lado autor de la bipartita, **323 de 332 autores (97%) quedan
marcados como "perifericos"**, porque el 97% de los autores comento en un solo video (grado 1)
y el percentil 10 de esa distribucion es exactamente 1. esto no es un error de metodo: con este
criterio, "periferico" termina describiendo el patron dominante de participacion (comentar una
sola vez), no una minoria marginal. es un hallazgo legitimo sobre la estructura de la red — la
gran mayoria de la interaccion observada es de autores que participan una unica vez, y solo un
grupo pequeno (9 autores con grado 2-3) tiene participacion repetida entre videos.

en video_projection, 9 de 19 videos (47%) tienen grado 0 (aislados: no comparten ningun autor
con ningun otro video de los 19), coherente con el 53% de LCC visto en 6.1. en la bipartita, del
lado video ningun nodo tiene grado 0 (por construccion, todo video en la red tiene al menos un
comentario/autor), pero 4 de 19 quedan marcados como perifericos por grado bajo relativo.

### 6.4 limite de interpretacion

estas metricas describen la estructura de la muestra observada (332 autores, 19 videos con
comentarios recolectados), no la poblacion completa de comentaristas de youtube ni de los 293
videos del dataset. la alta fragmentacion (10 componentes en las tres redes) y la fuerte
asimetria autor-video son un rasgo de como se recolectaron los datos (pocos videos concentran
la mayor parte de la participacion, muchos autores comentan una sola vez) y no deben
generalizarse mas alla de esta muestra.
