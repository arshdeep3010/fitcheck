# fitcheck: no-index (sample requirements below are test data, not evidence)
from app.rag.chunker import chunk_repo, chunk_text


def test_chunks_have_1_based_inclusive_line_ranges():
    text = "\n".join(f"line {i}" for i in range(1, 131))
    chunks = chunk_text("repo", "a.py", text, window=60, overlap=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert chunks[0].text.splitlines()[0] == "line 1"
    assert chunks[0].id == "repo:a.py:1-60"


def test_skips_env_files_and_dependency_folders(sample_repo):
    paths = {c.path for c in chunk_repo("demo-repo", sample_repo)}
    assert paths == {"src/ClaimController.java", "src/rag.py"}


def test_files_marked_no_index_are_skipped(sample_repo):
    (sample_repo / "src" / "sample_jd.py").write_text(
        '# fitcheck: no-index\nJD = """\n- Spring Boot\n- Kafka\n"""\nprint(JD)\n', encoding="utf-8")
    assert "src/sample_jd.py" not in {c.path for c in chunk_repo("demo-repo", sample_repo)}


def test_chunk_with_a_secret_is_never_indexed():
    fake_key = "nvapi-" + "abcdefghijklmnopqrstuvwxyz123456"  # split so secret scanners skip it
    text = f'API_KEY = "{fake_key}"\nprint(1)\nprint(2)\n'
    assert chunk_text("repo", "config.py", text) == []


def test_near_empty_chunks_are_dropped():
    assert chunk_text("repo", "a.py", "\n\nx = 1\n\n") == []
