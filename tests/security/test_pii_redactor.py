from src.security.pii_redactor import redact_and_report, redact_pii


def test_email_redaction():
    result = redact_and_report("Contact test@example.com")
    assert result["redacted"] is True
    assert result["pii_count"] == 1
    assert "[REDACTED_EMAIL]" in result["text"]
    assert "test@example.com" not in result["text"]


def test_phone_redaction():
    result = redact_and_report("Call 0501234567")
    assert result["redacted"] is True
    assert "[REDACTED_PHONE]" in result["text"]
    assert "0501234567" not in result["text"]


def test_national_id_redaction():
    result = redact_and_report("Saudi ID: 1234567890")
    assert "[REDACTED_NATIONAL_ID]" in result["text"]
    assert "1234567890" not in result["text"]


def test_multiple_pii_redaction():
    result = redact_and_report(
        "Email test@example.com and phone 0501234567"
    )
    assert result["pii_count"] == 2
    assert "[REDACTED_EMAIL]" in result["text"]
    assert "[REDACTED_PHONE]" in result["text"]
    assert "test@example.com" not in result["text"]
    assert "0501234567" not in result["text"]


def test_public_text_unchanged():
    text = "Saudi tourism information for visitors"
    redacted, count = redact_pii(text)
    assert redacted == text
    assert count == 0


def test_api_key_redaction():
    result = redact_and_report(
        "API key: sk-abcdefghijklmnopqrstuvwxyz123456"
    )
    assert "[REDACTED_API_KEY]" in result["text"]
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result["text"]
