from typing import Tuple

from src.security.pii_detector import detect_pii


REDACTION_LABELS = {
    "email": "[REDACTED_EMAIL]",
    "saudi_national_id": "[REDACTED_NATIONAL_ID]",
    "saudi_phone": "[REDACTED_PHONE]",
    "saudi_iban": "[REDACTED_IBAN]",
    "passport": "[REDACTED_PASSPORT]",
    "credit_card": "[REDACTED_CARD]",
    "api_key": "[REDACTED_API_KEY]",
}


def redact_pii(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0

    matches = detect_pii(text)

    if not matches:
        return text, 0

    redacted = text

    for match in reversed(matches):
        replacement = REDACTION_LABELS.get(
            match.pii_type,
            "[REDACTED_PII]",
        )
        redacted = (
            redacted[:match.start]
            + replacement
            + redacted[match.end:]
        )

    return redacted, len(matches)


def redact_and_report(text: str) -> dict:
    redacted_text, count = redact_pii(text)

    return {
        "original_contains_pii": count > 0,
        "redacted": count > 0,
        "pii_count": count,
        "text": redacted_text,
    }
