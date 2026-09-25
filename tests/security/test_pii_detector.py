from src.security.pii_detector import contains_pii, detect_pii, pii_summary


def test_email_detection():
    result = pii_summary("Contact test@example.com")
    assert result["contains_pii"]
    assert "email" in result["types"]


def test_saudi_national_id_detection():
    result = pii_summary("Saudi ID: 1234567890")
    assert "saudi_national_id" in result["types"]


def test_saudi_phone_detection():
    result = pii_summary("Phone: 0501234567")
    assert "saudi_phone" in result["types"]


def test_saudi_iban_detection():
    result = pii_summary("IBAN: SA0380000000608010167519")
    assert "saudi_iban" in result["types"]


def test_passport_detection():
    result = pii_summary("Passport: A1234567")
    assert "passport" in result["types"]


def test_credit_card_detection():
    result = pii_summary("Card: 4111 1111 1111 1111")
    assert "credit_card" in result["types"]


def test_api_key_detection():
    result = pii_summary("API key: sk-abcdefghijklmnopqrstuvwxyz123456")
    assert "api_key" in result["types"]


def test_no_pii():
    text = "Saudi tourism information for visitors"
    assert contains_pii(text) is False
    assert pii_summary(text)["count"] == 0


def test_multiple_pii_types():
    text = "Email test@example.com and phone 0501234567"
    result = detect_pii(text)
    types = {match.pii_type for match in result}
    assert "email" in types
    assert "saudi_phone" in types
    assert len(result) == 2


def test_bare_ten_digit_number_is_not_national_id():
    result = pii_summary("The model generated 1234567890 tokens during training.")
    assert "saudi_national_id" not in result["types"]
    assert result["contains_pii"] is False


def test_labeled_national_id_is_detected():
    result = pii_summary("Saudi ID: 1234567890")
    assert "saudi_national_id" in result["types"]
