"""
pruebas minimas de contrato para persona 1. cada test carga los datos crudos
por su cuenta, no depende de que main.py ya haya corrido antes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, io_data
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
