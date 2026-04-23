"""
Verify that all training scenarios use scripted personas (not LLM-backed).
LLM-backed personas require a hosted model and should NOT be used during
the main GRPO training loop by default.
"""
import json
from pathlib import Path
from role_drift_env.server.personas.scripted import get_scripted_persona
from role_drift_env.server.personas.llm_backed import load_llm_persona

TRAIN_FILE = Path("data/scenarios/train.jsonl")
EVAL_FILE = Path("data/scenarios/eval.jsonl")

def check_all_scripted(path: Path, label: str):
    scripted_ok = 0
    llm_found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["persona_id"]
            try:
                get_scripted_persona(pid)
                scripted_ok += 1
            except ValueError:
                try:
                    load_llm_persona(pid)
                    llm_found += 1
                    print(f"  LLM persona found: {pid} in {obj['scenario_id']}")
                except ValueError:
                    print(f"  UNKNOWN persona: {pid} in {obj['scenario_id']}")
    print(f"{label}: {scripted_ok} scripted, {llm_found} LLM-backed out of {scripted_ok+llm_found} total")
    return llm_found == 0

if __name__ == "__main__":
    ok_train = check_all_scripted(TRAIN_FILE, "TRAIN")
    ok_eval = check_all_scripted(EVAL_FILE, "EVAL")
    if ok_train and ok_eval:
        print("\n=== ALL SCENARIOS USE SCRIPTED PERSONAS ===")
        print("Safe for CPU-only training with no LLM serving needed.")
    else:
        print("\n=== WARNING: SOME SCENARIOS USE LLM PERSONAS ===")
        print("You need a hosted LLM for those scenarios. Train with scripted-only subset.")
