import os
import requests

token = os.environ.get("HF_TOKEN")
if not token:
    print("No HF_TOKEN found")
    exit(1)

# Try various billing endpoints
endpoints = [
    "/api/billing",
    "/api/billing/history", 
    "/api/billing/credits",
    "/api/billing/invoices"
]

for ep in endpoints:
    try:
        r = requests.get(f"https://huggingface.co{ep}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        print(f"{ep}: {r.status_code}")
        if r.status_code == 200:
            print(f"  Response: {r.text[:500]}")
        elif r.status_code != 404:
            print(f"  Error: {r.text[:200]}")
    except Exception as e:
        print(f"{ep}: Error - {e}")