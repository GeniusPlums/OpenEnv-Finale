from huggingface_hub import hf_hub_download
import zipfile

path = hf_hub_download("GeniusPlums/role-drift-rollouts", "rollouts.zip")
with zipfile.ZipFile(path, "r") as zf:
    with zf.open("error.txt") as f:
        print(f.read().decode("utf-8", errors="replace"))