import re


ARABIC_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)


def normalize_arabic(text: str) -> str:
    text = text.replace("\u0640", "")  # Tatweel

    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    text = ARABIC_DIACRITICS.sub("", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_arabic(text)

    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F"
        r"\u08A0-\u08FF"
        r"A-Za-z0-9"
        r"\s.,!?،؛؟:()\-/]",
        "",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    sample = "إِنَّ المملكةَ العربيةَ السُّعُودِيَّةَ"

    print("Original:")
    print(sample)

    print("\nNormalized:")
    print(normalize_arabic(sample))

    print("\nCleaned:")
    print(clean_text(sample))