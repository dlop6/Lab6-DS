"""
punto de entrada del pipeline de persona 1 (actividades 1 y 2 del laboratorio).
esto NO es el orquestador final del proyecto (eso le toca a persona 3 con las
etapas de eda/red bipartita), pero run_persona1_pipeline() se puede importar
tal cual desde el main.py definitivo cuando se integre.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, io_data, cleaning


def run_persona1_pipeline() -> None:
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

    print("=== pipeline de persona 1 completo, revisa data/processed/ y outputs/tables/ ===")


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["persona1", "persona2"], default="persona2")
    parser.add_argument("--sentiment", action="store_true")
    args = parser.parse_args()
    run_persona1_pipeline() if args.stage == "persona1" else run_persona2_advance(args.sentiment)
