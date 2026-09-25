from pathlib import Path
import easyocr


class ArabicOCR:
    def __init__(self):
        self.reader = easyocr.Reader(
            ["ar", "en"],
            gpu=True
        )

    def extract_text(self, image_path: str) -> str:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.reader.readtext(str(image_path))

        text = "\n".join(
            result[1]
            for result in results
            if result[1].strip()
        )

        return text


if __name__ == "__main__":
    print("Arabic OCR module loaded successfully.")