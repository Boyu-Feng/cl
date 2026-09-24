from pathlib import Path


VIEWERS = [
    Path("viewers/single_task_viewer.html"),
    Path("viewers/run_all_viewer.html"),
    Path("viewers/compare_traces.html"),
]


def test_viewers_accept_gzipped_artifact_files():
    for viewer in VIEWERS:
        html = viewer.read_text(encoding="utf-8")
        assert ".json.gz" in html
        assert "application/gzip" in html


def test_viewers_parse_gzipped_fetch_responses():
    for viewer in VIEWERS:
        html = viewer.read_text(encoding="utf-8")
        assert "DecompressionStream" in html
        assert "parseJsonBytes" in html
        assert "isGzipBytes" in html


def test_viewers_explain_when_gzip_decompression_is_unavailable():
    for viewer in VIEWERS:
        html = viewer.read_text(encoding="utf-8")
        assert 'typeof DecompressionStream === "undefined"' in html
        assert "This browser cannot read .json.gz artifacts" in html
