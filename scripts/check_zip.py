from huggingface_hub import hf_hub_download
import zipfile

path = hf_hub_download("GeniusPlums/role-drift-rollouts", "rollouts.zip")
print(f"Zip path: {path}")
with zipfile.ZipFile(path, "r") as zf:
    print("Files in archive:")
    for name in zf.namelist():
        print(f"  {name}")