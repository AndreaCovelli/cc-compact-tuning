from conftest import T0, assistant, write_jsonl

from cc_compact_tuning import cli, replay, store


def test_collect_reads_its_paths_from_the_environment(tmp_path, monkeypatch, capsys):
    write_jsonl(tmp_path / "projects" / "p" / "s.jsonl", [assistant("m1", T0, out=7)])
    monkeypatch.setenv("CC_TUNING_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("CC_TUNING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CC_TUNING_RATE_LOG", str(tmp_path / "missing.tsv"))

    cli.main(["collect"])

    assert "requests: 1 (+1)" in capsys.readouterr().out
    assert store.load(tmp_path / "data" / "store.json.gz")["requests"]["m1"]["out"] == 7


def test_replay_takes_the_model_by_its_short_name(monkeypatch):
    seen = {}
    monkeypatch.setattr(replay, "report", lambda st, d, w, model: seen.update(model=model))
    monkeypatch.setenv("CC_TUNING_DATA_DIR", "/nonexistent")
    cli.main(["replay", "--model", "fable"])
    assert seen["model"] == "claude-fable-5-2"
