# Video Script - Role Drift Env

## 0:00-0:25 — Hook
- **Visual**: Masters Union #2 transcript scrolling on screen
- **Voiceover**: "This is a real production voice agent. It was supposed to recover an incomplete college application. Watch what happens."
- **Clip shows**: User wants different program → Agent says goodbye → Agent continues with startup ideas → real estate → land procurement

## 0:25-0:50 — The Problem
- **Voiceover**: "I ran a services agency. 500 cold-callers. The LLM was the hard part. Frontier models add latency, fast models drift. Every voice-agent company hits this wall."

## 0:50-1:20 — The Environment  
- **Visual**: Four detector names + sample rule
- **Voiceover**: "Four programmatic rewards. Frozen customer. OpenEnv-compliant. Trained against the actual 3500-word production prompts. Validated on real failure transcripts."

## 1:20-1:45 — Results
- **Visual**: `plots/baseline_vs_trained.png` full screen
- **Voiceover**: "The diagnostic run shows positive slope (+0.14) with healthy KL (0.06). Full training gated on compute — recipe is validated."
- **Visual**: diag2 reward curve screenshot

## 1:45-2:00 — Close
- **Voiceover**: "Code in repo. Colab notebook proves pipeline works. Anyone with GPU can train against it. Link below."

---

## Shot List

1. screen_recording_terminal.md - Run env, detectors
2. plot_reward_curve.png - From diag2
3. plot_baseline_vs_trained.png - Comparison
4. transcript_masters_union_excerpt.txt - The money shot lines
5. code_snippet_env.py - Sample code