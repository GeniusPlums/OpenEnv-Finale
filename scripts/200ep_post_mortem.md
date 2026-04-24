# 200-EP Job Post-Mortem

**Job ID:** 69eb66666bbd7bff45bff018  
**Status:** CANCELED  
**Created:** 2026-04-24 12:47:34 UTC  
**Timeout:** 3h

---

## Step 1: Job Metadata

```
Status: CANCELED
Message: null
Docker image: python:3.12
Flavor: l40sx1
Secrets: HF_TOKEN
Environment: OUTPUT_REPO=GeniusPlums/role-drift-runs-200ep
```

No exit_reason, exit_code, or duration fields in the response.

---

## Step 2: Container Logs

`hf jobs logs 69eb66666bbd7bff45bff018` returned **empty/no output**.

This is unusual - even a killed/OOM container typically produces some log output before termination.

---

## Step 3: Billing State

**Attempted to check:**
- `hf auth whoami` - returned user: GeniusPlums
- Billing API via Python - failed because HF_TOKEN is not available in the local environment

**Subsequent attempt to re-run returned:**
```
402 Payment Required: Pre-paid credit balance is insufficient
```

This strongly suggests the 402 was returned when trying to START a NEW job after the first one cancelled - not during the first job's execution.

---

## Step 4: Diagnosis

### Evidence:
1. Job was RUNNING for approximately 40 minutes before showing CANCELED
2. No container logs available (hf jobs logs returned empty)
3. 402 error when attempting to launch a subsequent job
4. No artifacts (episode_log.jsonl, output.txt, best/) in the output repo

### Analysis:

**Hypothesis A: BILLING-KILL** - Partial support
- The 402 on the subsequent launch suggests billing issues
- However, the 402 came AFTER the first job was already canceled, when trying to launch a NEW job
- If billing killed the first job mid-run, we'd expect to see it in logs or job metadata

**Hypothesis B: OOM** - No supporting evidence
- No OOM signatures in container logs (logs are empty)
- Diag2 ran 20 eps successfully with same config
- 200 eps might accumulate more memory but gradient checkpointing is enabled

**Hypothesis C: MANUAL/PREEMPT** - Possible
- Could have been manually canceled
- Or HF's scheduler may have preempted due to resource constraints

### Most Likely: BILLING (Hypothesis A)

The sequence appears to be:
1. Job ran for ~40 minutes
2. Something caused it to cancel (possibly billing threshold hit mid-run)
3. When trying to launch a new job, 402 returned because remaining balance was too low
4. The empty logs could be because the final upload step never executed

---

## Step 5: Recommendations

**If billing is the issue:**
- Safe to retry once credits are added
- Consider adding more than the minimum to avoid mid-run kills

**If the job did run but produced no output:**
- The training loop may have crashed before producing any episodes
- Or the final upload step failed silently

**To recover any partial data:**
- The L40S instance may have saved data locally before termination
- HF may retain job working directories for a limited time
- Contact HF support if you need to recover any partial state

---

## What We Know Works

- Diag2 (20 episodes) completed successfully with same hyperparameters
- The reward signal is learnable (diag2 showed +0.54 improvement)
- The same config on same hardware (L40S) worked for shorter runs

## What Failed

- 200-ep run either didn't start training, crashed early, or was killed before producing artifacts
- No diagnostic data to determine exact failure point
