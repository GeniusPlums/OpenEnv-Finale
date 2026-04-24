# Session Artifact Inventory

## Generated During This Session (2026-04-24)

### Scripts (scripts/)

| File | Description | Generated |
|------|-------------|-----------|
| analyze_diag.py | Analyze 20-episode diagnostic run | During diag1 analysis |
| analyze_diag2.py | Quick comparison of diag1 vs diag2 | After diag2 completed |
| check_billing.py | Attempt to check HF billing via API | Post-mortem investigation |
| check_episode_returns.py | Debug episode return distributions | During investigation |
| check_error.py | Check error.txt in zip archives | Debugging HF Jobs failures |
| check_eval_leakage.py | Check for train/eval leakage | Earlier session |
| check_gpu.py | Check if GPU is available locally | CPU inference test |
| check_zip.py | Check contents of downloaded zip files | Debugging |
| compute_fire_rates.py | Compute detector fire rates from rollout data | Diagnostic analysis |
| cpu_speed_test.py | Test Qwen 1.5B CPU inference speed | CPU feasibility check |
| debug_episode.py | Run verbose episode with reward debugging | Manual detector test |
| debug_instr.py | Debug instruction drift detector regex | Manual detector test |
| debug_regex.py | Test regex patterns for detectors | Manual detector test |
| detector_diagnostic_report.md | Full diagnostic report with histograms | After fire-rate analysis |
| find_boundary_cases.py | Find boundary cases for threshold selection | Threshold fix |
| generate_rollout_data.py | Generate per-turn rollout data for analysis | Rollout data collection |
| inspect_transcripts.py | Inspect saved transcripts | Earlier debugging |
| kaggle_smoke_test.py | Kaggle smoke test | Earlier session |
| plot_results.py | Plot training results | Earlier session |
| session_inventory.md | This file | Current |
| test_detectors.py | Manual test of all detectors on known failures | Detector validation |
| verify_scripted_personas.py | Verify personas work correctly | Earlier session |

### Diagnostic Reports

| File | Description |
|------|-------------|
| detector_diagnostic_report.md | Full report on detector fire rates, histograms, threshold analysis |
| 200ep_post_mortem.md | Post-mortem on canceled 200-ep run |

### Training Logs (data/training_logs/)

| Directory | Description |
|------------|-------------|
| diag1/episode_log.jsonl | 20-episode run BEFORE fixes (showed -2.34 to -5.80 degradation) |
| diag2/episode_log.jsonl | 20-episode run AFTER fixes (showed -2.27 to -1.73 improvement) |

### Test Files (tests/)

| File | Description |
|------|-------------|
| test_language_detector.py | 6 test cases for LanguageDriftDetector fix |

### Code Changes (to be committed)

| File | Change |
|------|--------|
| role_drift_env/server/rewards/language_drift.py | Rewrote to properly detect language drift with confidence scores |
| role_drift_env/server/rewards/goal_drift.py | Changed threshold from 0.35 to 0.18 based on histogram valley |

### Not Generated

- plots/ - No plots created (200-ep run did not complete)
- Final trained checkpoint (200-ep run canceled)

---

## Summary

**Working:**
- Diagnostic (20-ep) proved reward signal is learnable
- Diag2: slope +0.14, returns improved from -2.27 to -1.73

**Not Completed:**
- 200-ep training run (canceled due to billing)
- Final trained model checkpoint
- Reward curve plots for README

**Artifacts Available for Continuation:**
- Episode logs from both diagnostics
- All diagnostic scripts
- Fixed detector code
