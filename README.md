# Role Drift Environment

**OpenEnv**-compatible text environment that scores production voice-agent pathologies: termination drift, goal drift, instruction drift, and language drift. This repo is the code for a GRPO (TRL) run on a **Qwen2.5-1.5B** policy with an LLM-backed **Qwen2.5-7B** customer simulator (served with vLLM, OpenAI-compatible API so the policy does not load a second 7B in process).

- **Code:** [github.com/GeniusPlums/OpenEnv-Finale](https://github.com/GeniusPlums/OpenEnv-Finale)  
- **Trained model:** [huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo](https://huggingface.co/GeniusPlums/role-drift-qwen-1-5b-grpo) (weights, `episode_log.jsonl`, `training.log`)  
- **Eval results (V10):** [huggingface.co/datasets/GeniusPlums/role-drift-eval-results](https://huggingface.co/datasets/GeniusPlums/role-drift-eval-results) (after the eval job)  
- **Live demo (Gradio):** (coming soon) `spaces/GeniusPlums/role-drift-demo` — update this line when Bundle C is deployed

---

## 1. Why this exists

A voice stack is only as good as the LLM. Frontier models are too slow for real-time voice; small models are fast enough but **drift**: they break persona, miss explicit rules, and cannot exit polite “thank you” loops. Prompts alone are brittle on behavior, not facts. This project turns drift into a **composable reward** and trains a deployable-size model to reduce it.

---

## 2. The result (V9 training + V10 eval)

GRPO on 100 episodes (group-relative advantages, vLLM customer-sim, 6 max turns in rollouts) produced a best **group-mean return of about 3.215** in the reported run. **Exact headline deltas between baseline and trained** on the held-out **eval** and **transfer** sets live in `BENCHMARK.md` and in the JSONs under `data/eval_results/` after you pull the dataset. Figures below are generated with `python scripts/make_plots.py` once `episode_log.jsonl` and the five eval JSONs are present locally.

![GRPO reward and rolling mean](plots/reward_curve.png)

![In-domain bar comparison by drift type](plots/eval_comparison.png)

---

## 3. How the environment works

An episode is a full agent–customer dialogue. The agent is your policy; the **customer** is a frozen LLM (or scripted persona) so only the policy learns. On each turn the **RewardComposer** adds weighted contributions from `term`, `goal`, `instr`, `lang`, and a small **task** bonus for clean, on-policy turns. Terminal success is an extra term from the terminal-success module. The detector implementations are under `role_drift_env/server/rewards/`; treat them as frozen when comparing V9 to V10 eval (do not “fix” them between training and final eval, or the comparison is muddled).

---

## 4. Artifacts you can reproduce

- **Scenarios:** `data/scenarios/train.jsonl` (training), `eval.jsonl` (in-domain test), `transfer_dearconnect.jsonl` (domain shift). The eval set is **disjoint** from training IDs.  
- **V10 eval job:** `scripts/run_v10_eval_job.sh` — persona gate, sequential baseline/trained eval (no two policies on the GPU at once), Hub upload with `trap` and per-scenario refresh.  
- **Local eval only:** `python scripts/run_eval.py in_domain --policy-checkpoint ...` (see `--help`).

---

## 5. Quick start (dev)

```bash
git clone https://github.com/GeniusPlums/OpenEnv-Finale.git
cd OpenEnv-Finale
pip install -e .
export ROLE_DRIFT_PERSONA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # vLLM for 7B customer
# Start vLLM for Qwen2.5-7B-Instruct, then:
python -c "from role_drift_env.server.environment import RoleDriftEnvironment as E; o,s=E().reset('term_kk_01'); print(len(o.system_prompt or ''))"
```

---

## 6. Plots and tables

- **Methodology and numbered results:** [BENCHMARK.md](BENCHMARK.md)  
- **Hypotheses (pre-registered):** [docs/hypotheses.md](docs/hypotheses.md) if present

---

## 7. Generalization (transfer)

The DearConnect **transfer** scenarios are defined in `data/scenarios/transfer_dearconnect.jsonl` (eight scenarios). The same eval harness runs baseline and trained checkpoints on that file; bar charts and CIs are in `transfer.png` and in the `transfer_*.json` results.

![Transfer (DearConnect) comparison](plots/transfer.png)

---

## 8. Failure modes (operator notes)

- **vLLM before eval:** always wait for `/v1/models`, then run a persona line that is **not** the scripted fallback string.  
- **Hub uploads:** any job that produces checkpoints or JSONs should **upload before exit**; use traps and `|| true` on non-fatal steps.  
- **OOM:** do not co-load two full policies; eval scripts are one process per policy. The Space should **not** load 7B; use scripted customers only.  
- **Token scope:** `HF_TOKEN` for Jobs must be **write**-capable for uploads and dataset commits.

This README is the public face of the submission: environment definition, where the weights and logs are, and how the eval was run honestly on held-out scenario IDs.
