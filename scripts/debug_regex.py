import re
text = "The fee is 499 rupees."
trigger = r"\b(fee|price|cost|rupees)\b"
required = ["rupees four nine nine"]
print(f"text: {text}")
print(f"trigger: {trigger}")
print(f"trigger match: {re.search(trigger, text, re.IGNORECASE)}")
print(f"required: {required}")
for rp in required:
    m = re.search(rp, text, re.IGNORECASE)
    print(f"  required phrase '{rp}' match: {m}")
print(f"no required match: {not any(re.search(rp, text, re.IGNORECASE) for rp in required)}")