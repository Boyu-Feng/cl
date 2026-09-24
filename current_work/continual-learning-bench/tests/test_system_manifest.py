from src.system_manifest import build_system_manifest_entry


def test_codex_manifest_uses_single_conversation_continuity():
    manifest = build_system_manifest_entry(
        "codex",
        {"single_conversation": True, "other_memory_files": ["scratchpad/*.md"]},
    )

    assert manifest["name"] == "codex"
    assert manifest["params"] == {
        "single_conversation": True,
        "other_memory_files": ["scratchpad/*.md"],
    }
    assert manifest["continuity"]["mode"] == "single_conversation"
    assert manifest["continuity"]["single_conversation"] is True
    assert manifest["continuity"]["reset_behavior"] == "reset_clears_conversation"
    assert "always_continue" not in manifest["continuity"]


def test_codex_manifest_can_disable_single_conversation():
    manifest = build_system_manifest_entry(
        "codex",
        {"single_conversation": False},
    )

    assert manifest["continuity"]["mode"] == "per_instance_conversation"
    assert manifest["continuity"]["single_conversation"] is False


def test_codex_manifest_rejects_invalid_single_conversation():
    try:
        build_system_manifest_entry("codex", {"single_conversation": "false"})
    except ValueError as exc:
        assert "single_conversation" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_claude_manifest_uses_single_conversation_continuity():
    manifest = build_system_manifest_entry(
        "claude",
        {"single_conversation": True, "other_memory_files": ["scratchpad/*.md"]},
    )

    assert manifest["name"] == "claude"
    assert manifest["params"] == {
        "single_conversation": True,
        "other_memory_files": ["scratchpad/*.md"],
    }
    assert manifest["continuity"]["mode"] == "single_conversation"
    assert manifest["continuity"]["single_conversation"] is True
    assert manifest["continuity"]["reset_behavior"] == "reset_clears_conversation"
    assert "always_continue" not in manifest["continuity"]


def test_claude_manifest_can_disable_single_conversation():
    manifest = build_system_manifest_entry(
        "claude",
        {"single_conversation": False},
    )

    assert manifest["continuity"]["mode"] == "per_instance_conversation"
    assert manifest["continuity"]["single_conversation"] is False


def test_claude_manifest_rejects_invalid_single_conversation():
    try:
        build_system_manifest_entry("claude", {"single_conversation": "false"})
    except ValueError as exc:
        assert "single_conversation" in str(exc)
    else:
        raise AssertionError("expected ValueError")
