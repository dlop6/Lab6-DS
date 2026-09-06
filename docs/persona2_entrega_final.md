# Persona 2 — entrega final (actividades 7 y 9)

Contrato H6 (comunidades + contenido/sentimiento). Codigo en `src/communities.py`
y `src/nlp.py`, etapa `python main.py --stage communities`. Consume la bipartita
y las proyecciones de `src/networks.py` (H3/H4), nunca las reconstruye a mano.

## 7. Comunidades

### 7.1 Red elegida y justificación

Se usó la **proyección video-video** (ponderada por autores compartidos), no
la bipartita ni la proyección autor-autor. La bipartita mezcla dos tipos de
nodo y Louvain está pensado para un solo tipo; la proyección autor-autor
agruparía personas, y con el 97% de los 332 autores comentando en un único
video (ver `peripheral_nodes.csv`, actividad 6) la mayoría de esas
"comunidades" serían pares triviales sin valor de contenido. La proyección
video-video conecta videos con audiencias solapadas, que es justo lo que se
necesita para caracterizar comunidades de contenido/audiencia en 7.5.

**Sesgo a documentar:** la proyección parte de incidencia binaria (cuenta
autores compartidos, no volumen de comentarios) y solo cubre los 19 videos con
comentarios recolectados — no dice nada de los otros 274 videos del dataset.

### 7.2 Algoritmo

Louvain (`networkx.algorithms.community.louvain_communities`) sobre
`video_projection`, con `weight="weight"`, `resolution=1` y `seed=42` para
reproducibilidad. Louvain es heurístico: optimiza modularidad de forma
aproximada (no exhaustiva), moviendo nodos a la comunidad vecina que más
mejora la modularidad hasta que ningún movimiento mejora más. El peso usado
es el de la proyección (autores compartidos); `reply_count` no participa en
ningún punto de esta etapa (verificado en `tests/test_contracts.py`).

### 7.3 Resultado

**12 comunidades**, modularidad **Q = 0.4053** (`community_metrics.csv`).
Tamaños: 2 comunidades de 3–4 videos concentran a los videos con más autores
compartidos (21% y 16% de los 19 videos respectivamente), y **9 de las 12
comunidades son singletons** (un solo video, sin ningún autor en común con el
resto) — coherente con el hallazgo de la actividad 6 de que el 47% de los
videos (9/19) quedan aislados en `video_projection`.

### 7.4 Visualización

`outputs/figures/fig_video_communities.png` dibuja los 19 nodos y las 11
aristas de la proyección completa, coloreados por comunidad; los 9 nodos
aislados se dibujan igual que el resto (no se ocultan).

### 7.5 Caracterización de las 3 comunidades principales

| Comunidad | Videos | Canales | Categoría | Autores | Comentarios | Términos frecuentes |
|---|---|---|---|---|---|---|
| 1 (n=4) | Bloqueos por alza de combustibles; cooptación de Walter Mazariegos en la USAC; captura de ladrón en Retalhuleu; "Qué rico come tu diputado" | PrensaLibreOficial, TN23 Guatemala, Quorum | News & Politics | 187 | 225 | pueblo, diputado, pagar, dinero; bigramas "pueblo pagar", "pagar sueldo", "pacto corrupto" |
| 2 (n=3) | Conferencia de prensa del Gobierno; recuperación del Puente Belice II; captura de delincuentes disfrazados | Gobierno de la República de Guatemala, Noti7 | Entertainment / News & Politics | 62 | 84 | presidente, guatemala, país, arevalo; bigramas "presidente bernardo", "apoyo presidente" |
| 3 (n=3) | "Arroz con pollo a la MONOPOLIO"; "Caminar en una ciudad hecha para carros"; "Internet: escoger el menos malo" | Quorum | News & Politics | 29 | 34 | excelente, empresa, internet, ley; bigramas "ley competencia", "libre mercado" |

Lectura: la comunidad 1 agrupa contenido de indignación/crítica política
(corrupción, sueldos de diputados); la comunidad 2 gira en torno a
figuras/instituciones de gobierno (presidente, conferencias oficiales); la
comunidad 3 es contenido de un solo canal (Quorum) sobre economía/regulación
(monopolios, competencia). Las tres siguen la misma categoría dominante del
dataset (News & Politics), esperable dado que 335/406 comentarios del dataset
caen en esa categoría.

**Sentimiento por comunidad (9.2):** columnas `sentiment_*` de
`community_content_summary.csv`. Quedan en blanco en este commit porque
`sentiment_comments.csv` requiere pysentimiento con descarga del modelo desde
Hugging Face (`pysentimiento/robertuito-sentiment-analysis`), que no estuvo
disponible en el entorno donde se verificó este código. Al correr
`python main.py --stage eda --sentiment` (o `python main.py --stage all
--sentiment`) seguido de `python main.py --stage communities` con acceso a
internet, las columnas se completan automáticamente — la lógica de
`characterize_communities()` y `build_sentiment_group_summary()` ya está
probada con datos sintéticos en `tests/test_contracts.py`
(`test_sentiment_group_summary_*`).

## 9. Análisis de contenido y sentimiento

### 9.1 Modelo

`pysentimiento`, `analyzer(task="sentiment", lang="es")`, sobre
`texto_original` (no `texto_limpio`): el toolkit trae su propio
preprocesamiento social para texto de redes (URLs, menciones, hashtags,
emojis) y aplicarle además la limpieza agresiva de 2.6 (minúsculas forzadas,
sin puntuación) le quitaría señales útiles para el modelo (signos de
exclamación, mayúsculas). Se guardan etiqueta (`POS`/`NEU`/`NEG`) y las tres
probabilidades por comentario en `sentiment_comments.csv`. El mismo archivo
se reutiliza sin recálculo en la pregunta 3.5 del avance y en esta sección.

### 9.2 Comparaciones

`sentiment_group_summary.csv` agrega el sentimiento por **video** (19
grupos), **canal**, **categoría** y **comunidad** (los 12 grupos de la
sección 7). Regla aplicada tal como la exige el plan: **ningún grupo con
n<5 se oculta**, solo se marca `small_sample=True` para tratarlo con cautela
en la interpretación (no se sobre-interpreta un porcentaje calculado sobre 1
o 2 comentarios). Figuras: `fig_sentiment_by_category.png` y
`fig_sentiment_by_community.png` (video y canal se dejan solo en la tabla:
con 19+ grupos una figura de barras se vuelve ilegible).

### 9.3 Hallazgos

Pendiente de completar con los porcentajes reales una vez corrido
`pysentimiento` con acceso a internet (ver nota en 7.5). El pipeline ya
soporta la lectura de esos resultados agregados por video/canal/categoría/
comunidad; la interpretación final debe describir patrones (p. ej. si la
comunidad 1, de tono más crítico/político por sus términos frecuentes,
concentra más `NEG` que la comunidad 3, de contenido más neutro/informativo)
sin atribuir causalidad ni generalizar más allá de esta muestra de 406
comentarios en 19 videos.
