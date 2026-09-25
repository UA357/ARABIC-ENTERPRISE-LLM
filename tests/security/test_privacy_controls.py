import pytest

from src.security.privacy_controls import (
    MAX_PROMPT_LENGTH,
    prepare_prompt,
    sanitize_for_logging,
)


def test_public_prompt():
    result = prepare_prompt("What are the main benefits of enterprise AI?")
    assert result["classification"] == "public"
    assert result["contains_pii"] is False
    assert result["redacted"] is False
    assert result["processed_text"] == "What are the main benefits of enterprise AI?"


def test_pii_prompt_is_redacted():
    result = prepare_prompt(
        "Customer email test@example.com and phone 0501234567"
    )
    assert result["contains_pii"] is True
    assert result["pii_count"] == 2
    assert result["redacted"] is True
    assert "test@example.com" not in result["processed_text"]
    assert "0501234567" not in result["processed_text"]


def test_log_sanitization():
    text = "Contact test@example.com or call 0501234567"
    safe_text = sanitize_for_logging(text)
    assert "test@example.com" not in safe_text
    assert "0501234567" not in safe_text
    assert "[REDACTED_EMAIL]" in safe_text
    assert "[REDACTED_PHONE]" in safe_text


def test_prompt_length_limit():
    with pytest.raises(
        ValueError,
        match=f"maximum length of {MAX_PROMPT_LENGTH}",
    ):
        prepare_prompt("x" * (MAX_PROMPT_LENGTH + 1))


def test_empty_prompt_rejected():
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        prepare_prompt("")


def test_non_string_prompt_rejected():
    with pytest.raises(TypeError, match="Prompt must be a string"):
        prepare_prompt(12345)


def test_redaction_can_be_disabled():
    result = prepare_prompt(
        "Customer email test@example.com",
        redact=False,
    )
    assert result["contains_pii"] is True
    assert result["redacted"] is False
    assert result["redaction_count"] == 0
    assert "test@example.com" in result["processed_text"]
    assert "test@example.com" not in result["log_safe_text"]
