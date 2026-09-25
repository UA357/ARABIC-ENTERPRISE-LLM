from src.security.data_classifier import DataClassification, classify_text


def test_public_data():
    assert classify_text("Saudi tourism information") == DataClassification.PUBLIC


def test_internal_data():
    assert classify_text("This is an internal company policy") == DataClassification.INTERNAL


def test_confidential_data():
    assert classify_text("Employee salary and contract information") == DataClassification.CONFIDENTIAL


def test_sensitive_data():
    assert classify_text("Customer passport and bank account information") == DataClassification.SENSITIVE


def test_arabic_internal_data():
    assert classify_text("\u0647\u0630\u0647 \u0633\u064a\u0627\u0633\u0629 \u062f\u0627\u062e\u0644\u064a\u0629") == DataClassification.INTERNAL


def test_arabic_confidential_data():
    assert classify_text("\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0631\u0627\u062a\u0628 \u0627\u0644\u0645\u0648\u0638\u0641") == DataClassification.CONFIDENTIAL


def test_arabic_sensitive_data():
    assert classify_text("\u0631\u0642\u0645 \u0627\u0644\u0647\u0648\u064a\u0629 \u0648\u062c\u0648\u0627\u0632 \u0627\u0644\u0633\u0641\u0631") == DataClassification.SENSITIVE
