import re
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int


PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "saudi_national_id": re.compile(r"(?i)(?:(?:saudi|national)\s+id|iqama|\u0631\u0642\u0645\s+(?:\u0627\u0644\u0647\u0648\u064a\u0629|\u0627\u0644\u0625\u0642\u0627\u0645\u0629)|\u0627\u0644\u0647\u0648\u064a\u0629|\u0627\u0644\u0625\u0642\u0627\u0645\u0629)\s*[:#-]?\s*[12]\d{9}(?!\d)"),
    "saudi_phone": re.compile(r"(?<!\d)(?:\+966[- ]?5\d{8}|00966[- ]?5\d{8}|05\d{8})(?!\d)"),
    "saudi_iban": re.compile(r"\bSA\d{2}[A-Z0-9]{20}\b", re.IGNORECASE),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "passport": re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d{6,8}(?![A-Za-z0-9])", re.IGNORECASE),
    "api_key": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"),
}


def detect_pii(text: str) -> List[PIIMatch]:
    if not text:
        return []

    matches: List[PIIMatch] = []

    for pii_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append(
                PIIMatch(
                    pii_type=pii_type,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return sorted(matches, key=lambda item: item.start)


def contains_pii(text: str) -> bool:
    return bool(detect_pii(text))


def pii_summary(text: str) -> dict:
    matches = detect_pii(text)
    return {
        "contains_pii": bool(matches),
        "count": len(matches),
        "types": sorted(set(match.pii_type for match in matches)),
        "matches": [asdict(match) for match in matches],
    }
