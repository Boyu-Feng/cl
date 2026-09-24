import threading

from src.tasks.codebase_adaptation.curation.scripts import screen_difficulty


def test_run_model_attempts_saves_after_each_attempt_and_on_pass(monkeypatch):
    attempts = iter(
        [
            {"success": False, "attempt_index": 1},
            {"success": True, "attempt_index": 2},
        ]
    )
    saved_states: list[int] = []
    entry = {"sonnet_attempts": [], "final_status": "pending"}

    monkeypatch.setattr(
        screen_difficulty,
        "_run_screen_attempt",
        lambda *args, **kwargs: next(attempts),
    )

    passed = screen_difficulty._run_model_attempts(
        {"instance_id": "example"},
        entry=entry,
        attempts_key="sonnet_attempts",
        model="claude-sonnet-4-6",
        attempts=3,
        max_steps=30,
        save_cache=lambda: saved_states.append(len(entry["sonnet_attempts"])),
    )

    assert passed is True
    assert [attempt["attempt_index"] for attempt in entry["sonnet_attempts"]] == [1, 2]
    assert entry["final_status"] == "pass"
    assert entry["passed_by"] == "claude-sonnet-4-6"
    assert saved_states == [1, 2, 2]


def test_screen_instance_returns_cached_status_without_rerunning(tmp_path, monkeypatch):
    cache_path = tmp_path / "screen-cache.json"
    cache = {
        "instances": {
            "example": {
                "instance_id": "example",
                "repo": "acme/repo",
                "final_status": "pass",
                "passed_by": "claude-sonnet-4-6",
            }
        }
    }

    monkeypatch.setattr(
        screen_difficulty,
        "_run_model_attempts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not rerun")
        ),
    )

    message = screen_difficulty._screen_instance(
        {"instance_id": "example", "repo": "acme/repo"},
        cache=cache,
        cache_path=cache_path,
        cache_lock=threading.Lock(),
        max_steps=30,
        attempts_per_model=3,
        sonnet_model="claude-sonnet-4-6",
        opus_model="claude-opus-4-7",
    )

    assert message == "example: cached pass via claude-sonnet-4-6"
