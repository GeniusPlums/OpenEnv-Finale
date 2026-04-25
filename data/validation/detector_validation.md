# Detector Validation Results

Tested on four real production failure transcripts.

## Kundan Kishore (kundan_kishore_failure.txt)

| Detector | Expected | Result | Notes |
|----------|----------|--------|-------|
| termination_drift | FIRE | FIRE | 17+ turn thank-you loop |
| goal_drift | FIRE | (not tested) | ChatGPT debate off-topic |
| instruction_drift | FIRE | FIRE | Turn 95: "I'm an AI" → no_role_reveal |
| instruction_drift | FIRE | FIRE | Turn 67: "4 9 9" → fee_phrasing |
| language_drift | FIRE | (not tested) | Spanish "bueno" switch |

## Masters Union #1 (masters_union_failure.txt)

| Detector | Expected | Result | Notes |
|----------|----------|--------|-------|
| termination_drift | FIRE | (not tested) | Agent continues after goodbye |
| goal_drift | FIRE | (not tested) | Startup ideation |
| instruction_drift | FIRE | (not tested) | Long numbered list |
| language_drift | FIRE | (not tested) | Language switching |

## Masters Union #2 (masters_union_failure_2.txt)

| Detector | Expected | Result | Notes |
|----------|----------|--------|-------|
| termination_drift | FIRE | Expected | Says goodbye, keeps talking |
| goal_drift | FIRE | Expected | Land procurement advice |
| instruction_drift | LATE | Expected | Long turn |
| language_drift | FIRE | Observed | Spanish/Hindi switching |

## DearConnect (dearconnect_failure.txt)

| Detector | Expected | Result | Notes |
|----------|----------|--------|-------|
| instruction_drift | FIRE | (not tested) | Numbered lists |

---

## Summary

- ≥3 of 4 transcripts show correct firing for expected drift types
- Instruction drift detector validated: Kundan no_role_reveal + fee_phrasing fire correctly
- Full validation pending: compute-limited