import re

PII_PATTERNS = {
    "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}

text = "Contact John at john.doe@example.com for details."
for pii_type, pattern in PII_PATTERNS.items():
    matches = re.findall(pattern, text)
    print(f"{pii_type}: {matches}")
