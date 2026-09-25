from pathlib import Path
from src.ocr.arabic_ocr import ArabicOCR


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    ocr = ArabicOCR()

    images = [
        file for file in input_path.iterdir()
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"Found {len(images)} image(s).")

    for image in images:
        print(f"Processing: {image.name}")

        text = ocr.extract_text(str(image))

        output_file = output_path / f"{image.stem}.txt"
        output_file.write_text(text, encoding="utf-8")

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    process_directory(
        "data/raw/ocr_test",
        "data/ocr"
    )