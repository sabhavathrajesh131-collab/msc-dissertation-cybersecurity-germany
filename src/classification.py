"""
Event classification models.
"""

def rule_based_classifier(text):
    if "restore" in text or "resumed" in text:
        return "Recovery"
    elif "investigation" in text or "response" in text:
        return "Response"
    else:
        return "Incident"
