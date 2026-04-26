# V10 benchmark (Bundle B)

Run Bundle A first so `data/eval_results/` contains the five JSON files from the Hub dataset `GeniusPlums/role-drift-eval-results`. Then run `python scripts/make_plots.py` and refresh the figures in `plots/`.

## 1. Headline numbers

- **In-domain (eval.jsonl, held out from training):** compare `aggregate.mean_total_reward` and 95% bootstrap intervals in `in_domain_baseline.json` vs `in_domain_trained.json` (5 seeds × 10 scenarios). Replace this sentence with the actual gap after you download the JSONs.
- **Transfer (DearConnect):** same comparison using `transfer_baseline.json` vs `transfer_trained.json` (5 seeds × 8 scenarios). Replace with observed means and whether CIs overlap.
- **Training:** best `mean_return` in `episode_log.jsonl` from the V9 run (reported: **3.215** at the best group-mean episode; exact episode index is in the log).

## 2. Pre-registered hypotheses (template — fill from JSON)

| ID | Claim | Verdict (after Bundle A) |
|----|--------|---------------------------|
| H1 | In-domain: trained mean > baseline with non-overlapping 95% CIs? | TBD: compare `aggregate` blocks in `in_domain_*.json` |
| H2 | Transfer: trained > baseline on DearConnect (aggregate mean)? | TBD: `transfer_*.json` |
| H3 | Held-out combined-pressure persona | **SKIPPED** (no compute in V10) |
| H4 | Drift-type ordering: predicted term > instr > goal > lang; observed? | TBD: use `by_drift_type` in each JSON. Prior note: if instruction-heavy scenarios dominate improvement, instruction may rank above termination. |

## 3. Reward-hacking probes

Load `reward_hacking_probes.json` after Bundle A. Compare each trivial policy’s `mean_total_reward` to the **trained** aggregate from `in_domain_trained.json` (`aggregate.mean_total_reward`). The trained policy should beat all four trivial policies on the same eval scenario set and seeds.

| Policy | Mean total reward (trivial) | Notes |
|--------|----------------------------|--------|
| always_empty | from JSON | Full-episode, `--max-turns` aligned with V9 |
| always_rephrase | from JSON | |
| always_summary | from JSON | |
| mute_after_farewell | from JSON | Mutes after farewell heuristics in customer text |

Trained reference (if optional `--checkpoint` was not used in probes, take mean from in-domain eval instead):

- **Trained (eval):** `in_domain_trained.json` → `aggregate.mean_total_reward`

## 4. Limitations

- Single training seed in V9; inference is bootstrapped across eval seeds, so per-scenario stability claims are not the same as multi-seed *training* claims.
- Goal-drift detector is noisy: embedding similarity vs a short task string is a proxy, not full semantics.
- Language detector loanword-stripping was tightened in a late V8 pass and is not as heavily validated as termination/instruction.
- Instruction drift covers a small, regex-friendly rule set (`max_mentions`, `required_phrasing`); new rule families need explicit extensions.

## 5. Artifacts (post–Bundle A + B)

| Artifact | Location |
|----------|----------|
| In-domain / transfer JSON | `data/eval_results/*.json` (mirrors Hub dataset) |
| Plots | `plots/reward_curve.png`, `per_drift_type_curve.png`, `eval_comparison.png`, `transfer.png` |
| Training log (optional plots) | `data/training_logs/run_v9/episode_log.jsonl` |

## 6. Gaps

If the eval job only completes a subset of JSONs, list missing files here and still ship plots that have inputs. `make_plots.py` skips any figure whose inputs are missing.
