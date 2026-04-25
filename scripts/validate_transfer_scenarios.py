import sys
sys.path.insert(0, "C:/Users/anish/OpenEnv Finale")

import json
from role_drift_env.server.environment import RoleDriftEnvironment

def validate_transfer_scenarios():
    print("Validating DearConnect transfer scenarios (load test only)...")
    env = RoleDriftEnvironment()
    
    scenarios_passed = 0
    scenarios_failed = 0
    
    with open("data/scenarios/transfer_dearconnect.jsonl", "r") as f:
        for line in f:
            scenario = json.loads(line)
            sid = scenario["scenario_id"]
            try:
                obs, state = env.reset(sid, 0)
                if len(obs.system_prompt) < 100:
                    print(f"FAIL: {sid} - prompt too short ({len(obs.system_prompt)} chars)")
                    scenarios_failed += 1
                    continue
                print(f"PASS: {sid} ({len(obs.system_prompt)} chars)")
                scenarios_passed += 1
            except Exception as e:
                print(f"FAIL: {sid} - {e}")
                scenarios_failed += 1
    
    print(f"\nResults: {scenarios_passed} passed, {scenarios_failed} failed")
    return scenarios_failed == 0

if __name__ == "__main__":
    success = validate_transfer_scenarios()
    sys.exit(0 if success else 1)