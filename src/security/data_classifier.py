from enum import Enum

class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"


SENSITIVE_KEYWORDS = [
    "password","passwd","secret","api key","api_key","token","credit card","bank account","national id","iqama","passport","medical record","medical history",
    "كلمة المرور","كلمة السر","سر","مفتاح api","رمز الوصول","رقم الهوية","رقم الإقامة","الإقامة","جواز السفر","الحساب البنكي","البطاقة الائتمانية","سجل طبي","السجل الطبي",
    "password",
    "passwd",
    "secret",
    "api key",
    "api_key",
    "token",
    "credit card",
    "bank account",
    "national id",
    "iqama",
    "passport",
    "medical record",
    "medical history",
]

CONFIDENTIAL_KEYWORDS = [
    "confidential","private","internal only","employee record","salary","contract","customer record",
    "راتب","راتب الموظف","عقد","سجل العميل","بيانات العميل","معلومات العميل",
    "confidential",
    "private",
    "internal only",
    "employee record",
    "salary",
    "contract",
    "customer record",
]

INTERNAL_KEYWORDS = [
    "internal","company policy","internal document","employee","department",
    "داخلي","وثيقة داخلية","سياسة الشركة","موظف","قسم",
    "internal",
    "company policy",
    "internal document",
    "employee",
    "department",
]


def classify_text(text: str) -> DataClassification:
    if not text or not text.strip():
        return DataClassification.PUBLIC

    normalized = text.casefold()

    if any(keyword in normalized for keyword in SENSITIVE_KEYWORDS):
        return DataClassification.SENSITIVE

    if any(keyword in normalized for keyword in CONFIDENTIAL_KEYWORDS):
        return DataClassification.CONFIDENTIAL

    if any(keyword in normalized for keyword in INTERNAL_KEYWORDS):
        return DataClassification.INTERNAL

    return DataClassification.PUBLIC


def classification_reason(text: str) -> str:
    classification = classify_text(text)

    if classification == DataClassification.SENSITIVE:
        return "Sensitive-data indicator detected."

    if classification == DataClassification.CONFIDENTIAL:
        return "Confidential-data indicator detected."

    if classification == DataClassification.INTERNAL:
        return "Internal-data indicator detected."

    return "No restricted-data indicator detected."
