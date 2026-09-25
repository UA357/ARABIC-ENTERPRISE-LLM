from src.security.data_classifier import classify_text
from src.security.pii_detector import pii_summary
from src.security.pii_redactor import redact_pii


MAX_PROMPT_LENGTH = 8000


def sanitize_for_logging(text: str) -> str:
    if not text:
        return ""

    redacted_text, _ = redact_pii(text)
    return redacted_text


def prepare_prompt(
    text: str,
    redact: bool = True,
) -> dict:
    if not isinstance(text, str):
        raise TypeError("Prompt must be a string.")

    if not text.strip():
        raise ValueError("Prompt cannot be empty.")

    if len(text) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters."
        )

    classification = classify_text(text)
    pii = pii_summary(text)

    processed_text = text
    redaction_count = 0

    if redact and pii["contains_pii"]:
        processed_text, redaction_count = redact_pii(text)

    return {
        "original_length": len(text),
        "processed_length": len(processed_text),
        "classification": classification.value,
        "contains_pii": pii["contains_pii"],
        "pii_types": pii["types"],
        "pii_count": pii["count"],
        "redacted": redaction_count > 0,
        "redaction_count": redaction_count,
        "processed_text": processed_text,
        "log_safe_text": sanitize_for_logging(text),
    }
