"""
orquestador definitivo del proyecto (arquitectura de repo, owner: Persona 3).

ejecuta el pipeline por etapas: data, eda, bipartite y metrics. el orquestador
no contiene logica de negocio propia: cada etapa delega a las funciones que
ya viven en su modulo/owner correspondiente.

uso:
    python main.py --stage data
    python main.py --stage eda [--sentiment]
    python main.py --stage bipartite
    python main.py --stage metrics
    python main.py --stage communities   # persona 2, entrega final: actividades 7 y 9
    python main.py --stage all [--sentiment]     # corre las 5 etapas en orden
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, io_data, cleaning


def run_persona1_pipeline() -> None:
    """Etapa 'data': actividades 1 y 2 (carga, integracion, calidad, limpieza).
    Logica sin modificar, solo re-expuesta bajo el nombre de
    etapa que usa el orquestador definitivo."""
    print("=== persona 1: carga, integracion, calidad y limpieza ===")

    # 1. carga cruda
    videos, comments = io_data.load_all()

    # 2. normalizar ids antes del join (defensivo, aunque hoy no haya whitespace)
    videos = cleaning.normalize_ids(videos, config.VIDEOS_ID_COLUMNS)
    comments = cleaning.normalize_ids(comments, config.COMMENTS_ID_COLUMNS)

    # 3. integracion, puede abortar todo el pipeline si el join no da 100%
    comments_enriched, join_audit = io_data.integrate_comments_videos(videos, comments)
    print(f"[main] join ok: {join_audit.iloc[0]['matched_both']}/{join_audit.iloc[0]['total_comments']}")

    # 4. diagnostico de calidad
    quality_summary = cleaning.build_quality_summary(videos, comments)
    quality_duplicates = cleaning.build_quality_duplicates(videos, comments)
    quality_outliers = cleaning.build_quality_outliers(videos, comments)
    quality_id_consistency = cleaning.build_quality_id_consistency(videos, comments)
    variable_treatment = cleaning.build_variable_treatment()

    # 5. auditoria de conteos en texto
    count_conversion_audit = cleaning.build_count_conversion_audit(videos, comments)

    # 6. datasets limpios finales
    videos_clean = cleaning.build_videos_clean(videos)

    print("[main] cargando modelo de spacy (puede tardar unos segundos)...")
    nlp_model = cleaning.load_spacy_model()
    comments_clean = cleaning.build_comments_clean(comments, nlp_model)

    # 7. efecto de la limpieza de texto
    text_cleaning_effect = cleaning.build_text_cleaning_effect(comments_clean)

    # 8. diccionario de datos
    data_dictionary = io_data.build_data_dictionary(videos, comments)

    # 9. escribir todo a disco
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    videos_clean.to_csv(config.PROCESSED_DIR / "videos_clean.csv", index=False, encoding="utf-8-sig")
    comments_clean.to_csv(config.PROCESSED_DIR / "comments_clean.csv", index=False, encoding="utf-8-sig")
    comments_enriched.to_csv(config.PROCESSED_DIR / "comments_enriched.csv", index=False, encoding="utf-8-sig")

    data_dictionary.to_csv(config.TABLES_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    join_audit.to_csv(config.TABLES_DIR / "join_audit.csv", index=False, encoding="utf-8-sig")
    quality_summary.to_csv(config.TABLES_DIR / "quality_summary.csv", index=False, encoding="utf-8-sig")
    quality_duplicates.to_csv(config.TABLES_DIR / "quality_duplicates.csv", index=False, encoding="utf-8-sig")
    quality_outliers.to_csv(config.TABLES_DIR / "quality_outliers.csv", index=False, encoding="utf-8-sig")
    quality_id_consistency.to_csv(config.TABLES_DIR / "quality_id_consistency.csv", index=False, encoding="utf-8-sig")
    variable_treatment.to_csv(config.TABLES_DIR / "variable_treatment.csv", index=False, encoding="utf-8-sig")
    count_conversion_audit.to_csv(config.TABLES_DIR / "count_conversion_audit.csv", index=False, encoding="utf-8-sig")
    text_cleaning_effect.to_csv(config.TABLES_DIR / "text_cleaning_effect.csv", index=False, encoding="utf-8-sig")

    print("=== etapa 'data' completa, revisa data/processed/ y outputs/tables/ ===")


def run_persona2_advance(include_sentiment: bool = False) -> None:
    """Ejecuta solo los entregables del avance de Persona 2 (actividad 3)."""
    from src import eda
    print("=== persona 2: analisis exploratorio (actividad 3) ===")
    tables = eda.run_eda()
    print(f"[main] EDA generado: {len(tables)} tablas y 4 figuras")
    if include_sentiment:
        from src import nlp
        nlp.run_sentiment()
        print("[main] sentimiento preliminar generado")
    else:
        print("[main] sentimiento omitido; use --sentiment para descargar/usar el modelo de pysentimiento")
    print("[main] comunidad preliminar pendiente de H3 (src/networks.py de Persona 3)")


def run_persona3_bipartite() -> None:
    """Etapa 'bipartite': actividad 4 y soporte de la actividad 5 (red bipartita
    autor-video, tablas, cobertura, figura y proyecciones preliminares).
    """
    from src import networks
    print("=== persona 3: red bipartita autor-video (actividad 4) ===")
    networks.run_bipartite_stage()
    print("=== etapa 'bipartite' completa, revisa outputs/tables/ y outputs/figures/ ===")


def run_persona1_metrics() -> None:
    """Etapa 'metrics': actividad 6 (topologia y fragmentacion). Depende de que
    la etapa 'data' y 'bipartite' ya hayan corrido; no reconstruye proyecciones,
    solo las consume desde src/networks.py para medir estructura."""
    from src import metrics
    print("=== persona 1: topologia y fragmentacion (actividad 6) ===")
    metrics.run_metrics_stage()
    print("=== etapa 'metrics' completa, revisa outputs/tables/ ===")


def run_persona2_final() -> None:
    """Etapa 'communities': entrega final de Persona 2, actividades 7 y 9.

    Depende de que 'data' y 'bipartite' ya hayan corrido (necesita
    videos_clean.csv, comments_clean.csv y la bipartita/proyecciones que
    entrega src/networks.py). No reconstruye la red a mano.

    7 (comunidades): Louvain sobre la proyeccion video-video, seed=42,
    weight="weight". Genera community_assignments.csv, community_metrics.csv,
    fig_video_communities.png y community_content_summary.csv (hasta 3
    comunidades principales).

    9 (contenido y sentimiento): reutiliza sentiment_comments.csv (lo genera
    si aun no existe) y produce sentiment_group_summary.csv + figuras,
    comparando por video, canal, categoria y comunidad (esta ultima usa el
    community_assignments.csv que la propia etapa acaba de generar).
    """
    from src import communities, nlp
    print("=== persona 2: comunidades (actividad 7) ===")
    communities.run_communities_stage()
    print("=== persona 2: contenido y sentimiento (actividad 9) ===")
    nlp.run_sentiment_comparison()
    print("=== etapa 'communities' completa, revisa outputs/tables/ y outputs/figures/ ===")


STAGES = {
    "data": lambda args: run_persona1_pipeline(),
    "eda": lambda args: run_persona2_advance(args.sentiment),
    "bipartite": lambda args: run_persona3_bipartite(),
    "metrics": lambda args: run_persona1_metrics(),
    "communities": lambda args: run_persona2_final(),
}


def run_all(args) -> None:
    for stage_fn in STAGES.values():
        stage_fn(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Orquestador del pipeline de Laboratorio 6.")
    parser.add_argument("--stage", choices=[*STAGES.keys(), "all"], default="all")
    parser.add_argument("--sentiment", action="store_true", help="incluye el precompute de sentimiento en la etapa eda")
    args = parser.parse_args()

    if args.stage == "all":
        run_all(args)
    else:
        STAGES[args.stage](args)
