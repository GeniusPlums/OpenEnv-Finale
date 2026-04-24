from huggingface_hub import hf_hub_download
import json

path = hf_hub_download('GeniusPlums/role-drift-runs-diag', 'episode_log.jsonl')
with open(path) as f:
    lines = f.readlines()

print("Episode return distribution:")
for i, l in enumerate(lines):
    d = json.loads(l)
    print(f"Ep {i}: min={d['min_return']:.2f} max={d['max_return']:.2f}")