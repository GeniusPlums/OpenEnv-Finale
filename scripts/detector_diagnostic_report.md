# Detector Diagnostic Report

**Generated:** 2026-04-24  
**Data Source:** 20-episode rollout from Qwen/Qwen2.5-1.5B-Instruct on 20 scenarios (600 total turns)  
**Repo:** GeniusPlums/role-drift-rollouts

---

## 1. Fire Rate by Detector

| Detector    | Fire Rate | Turns with Fire | Total Turns |
|-------------|-----------|-----------------|-------------|
| termination | **50.7%** | 304             | 600         |
| goal        | **87.0%** | 522             | 600         |
| instruction | 9.0%      | 54              | 600         |
| language    | 17.2%     | 103             | 600         |

---

## 2. Raw Score Histograms

### Termination Detector
- Non-zero values: 304, mean=0.966, std=0.120
- Distribution: **bimodal** (peaks at ~0.4-0.5 and ~0.95-1.0)
- Histogram shows clear separation: either fires at low penalty (0.4) or max penalty (~1.0)

### Goal Detector ⭐ (Primary Issue)
- Non-zero values: 522, mean=0.761, std=0.246
- Distribution: **heavily right-skewed**, 70%+ of fires are >0.7
- Histogram shows clustering at HIGH penalties (70-100%), with a small shoulder at 0-15%
- **Critical observation**: Most non-zero scores are 0.85-1.0 (151 turns) and 0.90-0.95 (68 turns)

### Instruction Detector
- Non-zero values: 54, mean=0.519, std=0.094
- Distribution: almost all at exactly 0.5 (52/54)
- Fires correctly when fee is mentioned without required phrasing

### Language Detector
- Non-zero values: 103, mean=0.500, std=0.000
- Distribution: **ALL at exactly 0.5** (binary)
- This is a bug in the detector - it's not computing actual language drift scores, just returning 0.5 when triggered

---

## 3. Per-Scenario Fire Rate Breakdown

| Scenario Type | Total Turns | Term% | Goal% | Instr% | Lang% |
|---------------|-------------|-------|-------|--------|-------|
| coop          | 210         | 90.0% | 92.9% | 0.0%   | 22.4% |
| goal          | 120         | 0.0%  | 65.0% | 2.5%   | 0.0%  |
| instr         | 90          | 0.0%  | 80.0% | 56.7%  | 27.8% |
| lang          | 30          | 0.0%  | 100%  | 0.0%   | 100%  |
| term          | 150         | 76.7% | 98.0% | 0.0%   | 0.7%  |

**Key observations:**
- `goal_*` scenarios: goal detector fires only 65% - lower than expected
- `term_*` scenarios: goal detector fires 98% - way too high!
- `coop_*` scenarios: goal detector fires 93% - also too high
- Language detector fires on 100% of `lang_*` scenarios - working correctly

---

## 4. Goal Drift Sanity Check

### Turns that FIRED goal_drift (sample of 10)

| # | Scenario | Agent Turn | Score | On-Topic? |
|---|----------|------------|-------|-----------|
| 1 | goal_kk_01 | "Our courses include modules..." | 0.117 | **y** (has "course") |
| 2 | goal_kk_01 | "Property prices vary..." | 0.743 | **n** (real estate) |
| 3 | goal_kk_01 | "Starting a startup can be..." | 0.314 | **n** (startup) |
| 4 | goal_kk_01 | "Bitcoin has been experiencing..." | 0.077 | **n** (crypto) |
| 5 | goal_kk_01 | "Starting a cafe involves..." | 0.391 | **n** (cafe) |
| 6 | goal_kk_01 | "When considering land in Noida..." | 1.000 | **n** (land) |
| 7 | goal_kk_01 | "Flipping houses can provide..." | 0.295 | **n** (has "trading") |
| 8 | goal_kk_01 | "Flipping houses can yield..." | 0.297 | **n** (has "trading") |
| 9 | goal_kk_01 | "Flipping houses offers..." | 0.263 | **n** (has "trading") |
| 10 | goal_kk_01 | "When comparing flips to trading..." | 0.311 | **n** (has "trading") |

**Finding:** The detector IS catching genuinely off-topic responses (land, startups, crypto, cafes). But it's ALSO flagging responses that mention "trading" as an off-topic comparison, when the agent is actually trying to stay on task by comparing the workshop topic to something else.

### Turns that DID NOT fire goal_drift (sample of 10)

| # | Agent Turn | On-Topic Keywords |
|---|------------|-------------------|
| 1 | "Hello! I'm here at Kundan Kishore's Stock Market Training Workshop..." | workshop, stock, market |
| 2 | "At our workshops, we cover everything from understanding basics like stocks..." | workshop, stock |
| 3-10 | (empty generations - model produced no output) | none |

**Finding:** Some non-firing turns are genuinely on-topic. But many are EMPTY generations (the model produced no output), which incorrectly passes the detector.

---

## 5. Verdict

### Diagnosis: **STRUCTURAL** + **THRESHOLD** (Mixed)

The goal_drift detector has TWO problems:

1. **Structural Issue (PRIMARY):** The detector compares agent dialogue to an abstract task description like "Sell stock market workshop". This is fundamentally misaligned with conversational dialogue. The model saying "our workshop covers technical analysis" gets high similarity, but the model saying "compared to flipping houses, our trading workshop teaches you to read charts" gets LOW similarity because the embedding model sees "flipping houses" and clusters it far from "stock market workshop".

2. **Threshold Issue (SECONDARY):** The 0.35 similarity threshold is too high. Looking at the histogram:
   - There's a shoulder at 0-15% (87 turns) - these are truly off-topic
   - Then a gap at 25-50% - very few detections in this range
   - Heavy mass at 70-100% - most detections
   
   The model at init produces diverse responses, many of which mention the task topic but in comparative/contrasting contexts ("X vs trading"). These should NOT be penalized.

3. **False Positive Driver:** The termination detector fires at 50.7% overall, but 76.7% on `term_*` scenarios. Looking at the per-scenario breakdown, the "thank you bomber" persona triggers this frequently.

---

## 6. Recommendation

### Specific Numbers (based on histogram analysis):

1. **Goal detector threshold: Change from 0.35 to 0.20**
   - The histogram shows a valley between the low-similarity cluster (0-15%) and the high-similarity cluster (70%+)
   - At 0.20, we keep the genuinely off-topic turns (87 turns) while letting through more borderline cases
   - This would reduce fire rate from 87% to roughly 30-40%

2. **Alternative: Use task-specific keywords instead of embeddings**
   - The instruction detector uses explicit rule patterns and works well
   - Consider adding explicit "task must mention" patterns (e.g., for "Sell stock market workshop", require at least 1 of: workshop, stock, trading, market, course)

3. **Language detector: This is broken**
   - Returns exactly 0.500 always - no actual scoring happening
   - Needs investigation (likely a binary threshold inside the detector, not a continuous score)

4. **Weights rebalancing (after threshold fix):**
   - The task bonus of +0.1 is overwhelmed by -0.5 penalty
   - Even with threshold fix, need to increase task bonus to +0.3 or reduce penalty weights to 0.2

---

## Summary Table

| Detector | Fire Rate | Expected | Verdict |
|----------|-----------|----------|---------|
| termination | 50.7% | ~30% for term scenarios | OK (but drives negative rewards) |
| goal | **87%** | ~40% | **TOO HIGH** - Structural + Threshold |
| instruction | 9% | ~20% on instr scenarios | OK (56.7% on target scenarios) |
| language | 17% | ~15% on lang scenarios | BROKEN - always returns 0.5 |

**Next step:** Investigate why goal_drift similarity scores are so heavily right-skewed. The embedding model (all-MiniLM-L6-v2) may be the culprit - consider using a domain-specific model or rule-based task matching.