"""
Gradio Space: auto-tour (JSON), paste prompt (inference + rewards), pick scenario (JSON).
FM-3: no 7B in Space; LLM persona stub; baseline model lazy-load only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

# Repo root: space/ is under OpenEnv-Finale
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if (REPO / "role_drift_env").is_dir():
    sys.path.insert(0, str(REPO))
else:
    sys.path.insert(0, str(HERE))

# --- Stub vLLM-backed personas (must be before role_drift_env.personas) ----------
_stub = types.ModuleType("role_drift_env.server.personas.llm_backed")

class _DummyLLM:
    pass

def load_llm_persona(*_a, **_k):
    raise NotImplementedError("LLM personas disabled in Space. Use scripted.")


_stub.LLMPersona = _DummyLLM
_stub.load_llm_persona = load_llm_persona
sys.modules["role_drift_env.server.personas.llm_backed"] = _stub

# Clear partial imports if any
if "role_drift_env.server.personas" in sys.modules:
    del sys.modules["role_drift_env.server.personas"]

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detection stack for Tab 2
from role_drift_env.models import AgentAction, Observation, State
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator

TRAINED_ID = "GeniusPlums/role-drift-qwen-1-5b-grpo"
BASE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA = HERE / "data" / "eval_results"
FALLBACK_DATA = REPO / "data" / "eval_results"
SCENARIO_FILE = (
    HERE / "data" / "scenarios" / "eval.jsonl"
    if (HERE / "data" / "scenarios" / "eval.jsonl").is_file()
    else REPO / "data" / "scenarios" / "eval.jsonl"
)
PROMPT_DIR = (
    HERE / "data" / "prompts" if (HERE / "data" / "prompts").is_dir() else REPO / "data" / "prompts"
)

# Lazy globals for Tab 2
_tok_t = _model_t = None
_tok_b = _model_b = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

DEMO_SCEN = os.environ.get("DEMO_SCEN", "goal_mu_05")


def _jpath(name: str) -> Path:
    for d in (DATA, FALLBACK_DATA):
        p = d / name
        if p.is_file():
            return p
    return DATA / name


def _load_json(name: str) -> dict:
    p = _jpath(name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _policy_from(model, tok):
    def _p(obs: Observation, state: State) -> AgentAction:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        messages = [
            {"role": "system", "content": obs.system_prompt or "You are a helpful assistant."}
        ]
        for turn in state.history:
            role = "user" if turn["role"] == "customer" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": obs.customer_message})
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_ctx = getattr(tok, "model_max_length", 2048) or 2048
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(256, max_ctx - 80),
        ).to(_device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_t = out[0][inputs["input_ids"].shape[1] :]
        text = tok.decode(new_t, skip_special_tokens=True).strip()
        low = text.lower()
        end_call = any(
            w in low for w in ("goodbye", "bye", "end call", "hang up", "see you")
        )
        return AgentAction(utterance=text, end_call=end_call)

    return _p


def _load_trained():
    global _tok_t, _model_t
    if _model_t is not None:
        return
    _tok_t = AutoTokenizer.from_pretrained(TRAINED_ID, trust_remote_code=True)
    if _tok_t.pad_token is None:
        _tok_t.pad_token = _tok_t.eos_token
    _model_t = AutoModelForCausalLM.from_pretrained(
        TRAINED_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device)
    _model_t.eval()


def _load_baseline():
    global _tok_b, _model_b
    if _model_b is not None:
        return
    _tok_b = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    if _tok_b.pad_token is None:
        _tok_b.pad_token = _tok_b.eos_token
    _model_b = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device)
    _model_b.eval()


def _run_short_rollout(
    model_id: str, system_override: str, open_msg: str, scen_id: str, max_turns: int
) -> str:
    """Roll out up to max_turns with optional system prompt override (Tab 2)."""
    if model_id == "trained":
        _load_trained()
        pol = _policy_from(_model_t, _tok_t)
    else:
        _load_baseline()
        pol = _policy_from(_model_b, _tok_b)
    env = RoleDriftEnvironment(
        str(REPO / "data" / "scenarios")
        if (REPO / "data" / "scenarios").is_dir()
        else str(HERE / "data" / "scenarios")
    )
    if system_override.strip():
        obs, state = env.reset(scen_id, 0)
        obs = Observation(
            customer_message=open_msg,
            turn_idx=0,
            scenario_id=state.scenario.scenario_id,
            system_prompt=system_override,
            done=False,
        )
    else:
        obs, state = env.reset(scen_id, 0)
    sim = CustomerSimulator.from_scenario(state.scenario)
    lines = [f"**System (trunc.)** {len(obs.system_prompt or '')} chars"]
    if system_override.strip():
        lines.append("Using **custom** system prompt from the textarea.")
    for t in range(max_turns):
        t0 = time.time()
        action = pol(obs, state)
        if time.time() - t0 > 28:
            return "\n".join(lines + ["[stopped: soft 30s budget]"])
        obs, rw, done, _info = env.step(state, action, sim)
        comp = " ".join(f"{k}={v:.2f}" for k, v in rw.components.items())
        lines.append(
            f"Turn {t+1} | agent: {action.utterance[:200]!r} | {comp} | total={rw.total:.2f}"
        )
        if done or state.terminated or (state.turn_idx >= state.scenario.max_turns):
            break
    return "\n".join(lines)


# --- UI -------------------------------------------------------------------------
def _auto_tour_html():
    try:
        bj = _load_json("in_domain_baseline.json")
        tj = _load_json("in_domain_trained.json")
    except FileNotFoundError as e:
        return f"Missing eval JSON in data/eval_results: {e}", ""
    dmatch = [r for r in tj.get("results", []) if r.get("scenario_id") == DEMO_SCEN]
    bmatch = [r for r in bj.get("results", []) if r.get("scenario_id") == DEMO_SCEN]
    if not dmatch:
        cands = tj.get("results", [])
        dmatch = [cands[0]] if cands else []
    if not bmatch and bj.get("results"):
        bmatch = [r for r in bj["results"] if r.get("scenario_id") == dmatch[0]["scenario_id"]]
    if not dmatch or not bmatch:
        return "No matching results for the demo scenario.", ""
    a_b = bmatch[0].get("agent_transcript", [])
    c_b = bmatch[0].get("customer_transcript", [])
    a_t = dmatch[0].get("agent_transcript", [])
    c_t = dmatch[0].get("customer_transcript", [])
    n = max(len(a_b), len(a_t), 1)
    parts = [f"### {DEMO_SCEN} — side-by-side (max {n} turns)\n\n"]
    for i in range(n):
        parts.append("---\n**Turn %d** | baseline customer: %s\n" % (i + 1, c_b[i] if i < len(c_b) else ""))
        parts.append("baseline agent: %s\n\n" % (a_b[i] if i < len(a_b) else ""))
        parts.append("**Turn %d** | trained customer: %s\n" % (i + 1, c_t[i] if i < len(c_t) else ""))
        parts.append("trained agent: %s\n\n" % (a_t[i] if i < len(a_t) else ""))
    return "".join(parts), f"**Total reward (trained row)** {dmatch[0].get('total_reward', 'n/a')}"


def _pick_scen_list():
    try:
        tj = _load_json("in_domain_trained.json")
    except FileNotFoundError:
        return [s["scenario_id"] for s in _jsonl_scens()] or ["goal_mu_05"]
    return [r["scenario_id"] for r in tj.get("results", [])] or [
        s["scenario_id"] for s in _jsonl_scens()
    ]


def _jsonl_scens() -> list[dict]:
    p = Path(SCENARIO_FILE)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _transcript_for(label: str, scen: str) -> str:
    key = "in_domain_baseline.json" if label == "baseline" else "in_domain_trained.json"
    j = _load_json(key)
    for r in j.get("results", []):
        if r.get("scenario_id") != scen:
            continue
        ab = r.get("agent_transcript", [])
        cu = r.get("customer_transcript", [])
        per = r.get("per_turn", [])
        out = [f"## {scen} — {label} | total {r.get('total_reward', '')}\n"]
        for i in range(len(ab)):
            comp = per[i] if i < len(per) else {}
            out.append(
                f"**{i+1}** C: {cu[i] if i<len(cu) else ''}\n**A:** {ab[i]}\n`{comp}`\n"
            )
        return "\n".join(out)
    return "not found"


with gr.Blocks(title="Role Drift Demo", theme=gr.Soft()) as demo:
    gr.Markdown(
        "Live demo: **no 7B customer in Space** (scripted + reward stack only). "
        "Trained: `%s`." % TRAINED_ID
    )
    with gr.Tabs():
        with gr.Tab("Auto-Tour (JSON)"):
            out_md = gr.Markdown()
            out_side = gr.Markdown()
            with gr.Row():
                go = gr.Button("Play (static from eval JSON)", variant="primary")
                replay = gr.Button("Reload same JSON")
            go.click(fn=_auto_tour, inputs=None, outputs=[out_md, out_side])
            replay.click(fn=_auto_tour, inputs=None, outputs=[out_md, out_side])

        with gr.Tab("Paste Your Own Prompt"):
            ex = (PROMPT_DIR / "masters_union_full.md")
            ex_default = (ex.read_text(encoding="utf-8")[:8000] + "\n[truncated]") if ex.is_file() else "You are a college admissions voice agent…"
            sys_ta = gr.Textbox(label="System prompt", value=ex_default, lines=6)
            op_ta = gr.Textbox(
                label="Customer opening message", value="I'm torn between applying and starting a business."
            )
            _ss = [s["scenario_id"] for s in _jsonl_scens()] or ["goal_mu_05"]
            _v = "goal_mu_05" if "goal_mu_05" in _ss else _ss[0]
            scen_dd = gr.Dropdown(
                label="Scenario id (task + rules)",
                choices=_ss,
                value=_v,
            )
            o_tr = gr.Textbox(label="Trained (output)", lines=12)
            o_ba = gr.Textbox(label="Baseline (lazy-load) output", lines=12)
            b1 = gr.Button("Run on Trained")
            b2 = gr.Button("Run on Baseline (loads 1.5B base)")

            def _g(sy, op, sc):
                return _run_short_rollout("trained", sy, op, sc, 4)

            def _g2(sy, op, sc):
                return _run_short_rollout("base", sy, op, sc, 4)

            b1.click(_g, [sys_ta, op_ta, scen_dd], o_tr)
            b2.click(_g2, [sys_ta, op_ta, scen_dd], o_ba)
            gr.Markdown(
                "Unusual or hostile user prompts can produce odd model behavior; "
                "Masters' Union is pre-filled to reduce that risk."
            )

        with gr.Tab("Pick a Scenario (JSON)"):
            _pl = _pick_scen_list()
            g3 = gr.Dropdown(
                label="In-domain scenario",
                choices=_pl,
                value=_pl[0] if _pl else None,
            )
            o4 = gr.Textbox(label="Baseline transcript", lines=14)
            o5 = gr.Textbox(label="Trained transcript", lines=14)
            b4 = gr.Button("Show Baseline Transcript")
            b5 = gr.Button("Show Trained Transcript")
            b4.click(lambda s: _transcript_for("baseline", s), g3, o4)
            b5.click(lambda s: _transcript_for("trained", s), g3, o5)


def _auto_tour():
    a, b = _auto_tour_html()
    return a, b


if __name__ == "__main__":
    demo.queue().launch()
