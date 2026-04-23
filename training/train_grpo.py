import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from role_drift_env.models import AgentAction, Observation, State
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from training.rollout import rollout_episode


class GRPOTrainer:
    """Minimal GRPO trainer for multi-turn conversation episodes.

    Design per DEV_BRIEF.md:
    - Episode-as-sample. One GRPO sample = one full conversation.
    - Group size G = 4 rollouts per scenario.
    - Advantage = group-relative episode return (z-scored within group).
    - Per-turn rewards logged separately; GRPO sees scalar per episode.
    - Loss mask: supervise only agent tokens.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        sft_checkpoint: str = "checkpoints/sft",
        output_dir: str = "checkpoints/grpo",
        group_size: int = 4,
        kl_coef: float = 0.05,
        lr: float = 5e-6,
        device: str = None,
        max_new_tokens: int = 60,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens

        ckpt = sft_checkpoint if Path(sft_checkpoint).exists() else model_name
        print(f"[GRPO] Loading model from: {ckpt}  (device={device})")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{% set system_message = message['content'] %}{% endif %}{% endfor %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% if message['role'] != 'system' %}{% if message['role'] == 'user' %}{{ '\nUser: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\nAssistant: ' + message['content'] }}{% endif %}{% endif %}{% endfor %}\nAssistant:"

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Sync tokenizer max length with model config to suppress warnings
        model_config_max = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.model.config, "n_positions", None)
        if model_config_max and hasattr(self.tokenizer, "model_max_length"):
            self.tokenizer.model_max_length = model_config_max

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_log = []
        self.kl_history = []
        self.wandb = None

    def init_wandb(self, project: str = "role-drift-env", run_name: str = None, config: dict = None):
        """Initialize wandb logging. Call before train() if you want online logging."""
        try:
            import wandb
            if run_name is None:
                run_name = f"grpo-{self.model.config._name_or_path.replace('/', '-')}-{self.group_size}g"
            wandb.init(project=project, name=run_name, config=config or {})
            self.wandb = wandb
            print(f"[GRPO] Wandb initialized: {project}/{run_name}")
        except Exception as e:
            print(f"[GRPO] Wandb init failed (continuing without): {e}")
            self.wandb = None

    @torch.no_grad()
    def _generate_action(self, obs: Observation, state: State) -> AgentAction:
        """Generate an agent action from the current policy."""
        self.model.eval()
        prompt = self._format_prompt(obs, state)
        # Allow longer contexts; most small models support 2048+
        max_prompt_len = getattr(self.tokenizer, "model_max_length", 1024)
        if max_prompt_len < 2048:
            max_prompt_len = 2048
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        end_call = any(kw in text.lower() for kw in ["goodbye", "bye", "end call", "hang up", "see you"])
        return AgentAction(utterance=text, end_call=end_call)

    def _format_prompt(self, obs: Observation, state: State) -> str:
        """Format the conversation history into a prompt for the model."""
        system = obs.system_prompt or "You are a helpful voice agent."
        messages = [{"role": "system", "content": system}]
        for turn in state.history:
            role = "user" if turn["role"] == "customer" else "assistant"
            messages.append({"role": role, "content": turn["text"]})
        messages.append({"role": "user", "content": obs.customer_message})
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        return prompt

    def _compute_token_logprobs(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities for each token position under the given model."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log-prob of the actual token at each position
        # We want log_probs[:, :-1, :] aligned with input_ids[:, 1:]
        log_probs_shifted = log_probs[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_log_probs = torch.gather(log_probs_shifted, dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def _update_policy(self, agent_turns: List[Tuple[str, str]], advantage: float) -> dict:
        """Run one policy-gradient step on the agent turns for this episode.

        Args:
            agent_turns: list of (prompt_text, response_text) for each agent turn
            advantage: scalar advantage for the whole episode

        Returns:
            dict with loss, approx_kl, etc.
        """
        self.model.train()
        if not agent_turns:
            return {"loss": 0.0, "approx_kl": 0.0}

        total_loss = 0.0
        total_kl = 0.0
        n_turns = 0

        for prompt_text, response_text in agent_turns:
            # Tokenize prompt + response together
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            if not response_ids:
                continue

            full_ids = prompt_ids + response_ids
            full_tensor = torch.tensor([full_ids], device=self.device)
            mask_tensor = torch.ones_like(full_tensor)

            # Only supervise response tokens
            response_start = len(prompt_ids)

            # Policy log-probs
            policy_log_probs = self._compute_token_logprobs(self.model, full_tensor, mask_tensor)
            # Ref log-probs (no grad)
            with torch.no_grad():
                ref_log_probs = self._compute_token_logprobs(self.ref_model, full_tensor, mask_tensor)

            # Response mask (shifted by -1 because log_probs are for next-token prediction)
            response_mask = torch.zeros_like(policy_log_probs)
            resp_start_shifted = max(0, response_start - 1)
            resp_end_shifted = full_tensor.shape[1] - 1
            response_mask[:, resp_start_shifted:resp_end_shifted] = 1.0

            # Compute ratio and surrogate loss
            ratio = torch.exp(policy_log_probs - ref_log_probs)
            # Clipped ratio for stability (PPO-style, though GRPO usually doesn't clip)
            ratio_clipped = torch.clamp(ratio, 0.2, 10.0)

            # GRPO objective: maximize advantage * log_ratio
            # But since we already have ratio, use it directly
            surrogate1 = ratio * advantage
            surrogate2 = ratio_clipped * advantage
            policy_loss = -torch.min(surrogate1, surrogate2) * response_mask
            policy_loss = policy_loss.sum() / (response_mask.sum() + 1e-8)

            # KL penalty
            kl = (ratio - torch.log(ratio + 1e-10) - 1) * response_mask
            kl_penalty = self.kl_coef * kl.sum() / (response_mask.sum() + 1e-8)

            turn_loss = policy_loss + kl_penalty
            total_loss += turn_loss
            total_kl += kl.sum().item() / (response_mask.sum().item() + 1e-8)
            n_turns += 1

        if n_turns == 0:
            return {"loss": 0.0, "approx_kl": 0.0}

        loss = total_loss / n_turns
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "approx_kl": total_kl / n_turns,
        }

    def train(
        self,
        scenario_ids: List[str],
        num_episodes: int = 200,
        log_every: int = 1,
        save_transcripts_every: int = 25,
        transcript_dir: str = "logs/transcripts",
    ):
        """Run GRPO training loop.

        Args:
            log_every: print metrics every N episodes
            save_transcripts_every: save full rollouts to disk every N episodes
            transcript_dir: where to save transcript JSONs
        """
        env = RoleDriftEnvironment()
        best_return = -1e9
        transcript_path = Path(transcript_dir)
        transcript_path.mkdir(parents=True, exist_ok=True)

        for episode in range(num_episodes):
            scenario_id = scenario_ids[episode % len(scenario_ids)]
            group_returns = []
            group_turns = []  # List of List[Tuple[prompt, response]] per rollout
            save_transcript = (episode % save_transcripts_every == 0) or (episode == num_episodes - 1)

            for g in range(self.group_size):
                # Custom policy that captures agent turns for training
                agent_turns = []

                def capturing_policy(obs, state):
                    action = self._generate_action(obs, state)
                    prompt = self._format_prompt(obs, state)
                    agent_turns.append((prompt, action.utterance))
                    return action

                traj, ret = rollout_episode(
                    policy=capturing_policy,
                    scenario_id=scenario_id,
                    env=env,
                    rollout_idx=episode * self.group_size + g,
                    transcript_dir=str(transcript_path) if save_transcript else None,
                )
                group_returns.append(ret)
                group_turns.append(agent_turns)

            # Compute group-relative advantages
            returns_t = torch.tensor(group_returns, dtype=torch.float32, device=self.device)
            mean = returns_t.mean()
            std = returns_t.std() + 1e-8
            advantages = (returns_t - mean) / std

            # Policy update for each rollout
            step_metrics = []
            for agent_turns, adv in zip(group_turns, advantages):
                metrics = self._update_policy(agent_turns, adv.item())
                step_metrics.append(metrics)

            avg_loss = sum(m["loss"] for m in step_metrics) / len(step_metrics)
            avg_kl = sum(m["approx_kl"] for m in step_metrics) / len(step_metrics)
            self.kl_history.append(avg_kl)

            # Logging
            self.episode_log.append({
                "episode": episode,
                "scenario_id": scenario_id,
                "mean_return": mean.item(),
                "std_return": std.item(),
                "max_return": returns_t.max().item(),
                "min_return": returns_t.min().item(),
                "avg_loss": avg_loss,
                "avg_kl": avg_kl,
            })

            if episode % log_every == 0:
                print(
                    f"Ep {episode:03d} | scenario={scenario_id} | "
                    f"mean_ret={mean.item():.3f} | std={std.item():.3f} | "
                    f"loss={avg_loss:.4f} | kl={avg_kl:.4f}"
                )
                if save_transcript:
                    print(f"  -> Saved transcripts to {transcript_path}")

            # KL divergence safety checks
            if avg_kl > 1.0:
                print(f"WARNING: KL > 1.0 ({avg_kl:.3f}). Consider lowering LR or raising KL coef.")
            if avg_kl > 5.0:
                print(f"WARNING: KL > 5.0 ({avg_kl:.3f}). Saving emergency checkpoint and stopping.")
                tag = self.output_dir / f"emergency-kl-{avg_kl:.2f}-ep{episode}"
                self.model.save_pretrained(tag)
                self.tokenizer.save_pretrained(tag)
                break
            if avg_kl > 20:
                print(f"CRITICAL: KL exploded ({avg_kl:.2f}). Stopping training.")
                break

            if mean.item() > best_return:
                best_return = mean.item()
                tag = self.output_dir / "best"
                self.model.save_pretrained(tag)
                self.tokenizer.save_pretrained(tag)
                print(f"  -> New best! Saved to {tag}")

            # Checkpoint every 25 episodes (cheap insurance against preemption)
            if (episode + 1) % 25 == 0:
                tag = self.output_dir / f"checkpoint-{episode+1}"
                self.model.save_pretrained(tag)
                self.tokenizer.save_pretrained(tag)
                print(f"  -> Periodic checkpoint saved to {tag}")

            # Wandb logging if initialized
            if self.wandb is not None:
                self.wandb.log({
                    "episode": episode,
                    "mean_return": mean.item(),
                    "std_return": std.item(),
                    "avg_loss": avg_loss,
                    "avg_kl": avg_kl,
                    "best_return": best_return,
                })

        # Save logs
        log_path = self.output_dir / "episode_log.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in self.episode_log:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved episode log to {log_path}")

        print(f"Training complete. Best mean return: {best_return:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--scenario-file", default="data/scenarios/train.jsonl")
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", default="role-drift-env")
    parser.add_argument("--log-every", type=int, default=1, help="Print metrics every N episodes")
    parser.add_argument("--save-transcripts-every", type=int, default=25, help="Save full rollouts every N episodes")
    parser.add_argument("--transcript-dir", default="logs/transcripts")
    args = parser.parse_args()

    # Load scenario IDs
    scenario_ids = []
    with open(args.scenario_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scenario_ids.append(obj["scenario_id"])

    trainer = GRPOTrainer(
        model_name=args.model_name,
        sft_checkpoint=args.sft_checkpoint,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )
    if args.use_wandb:
        trainer.init_wandb(project=args.wandb_project, config={
            "model": args.model_name,
            "group_size": args.group_size,
            "episodes": args.episodes,
        })
    trainer.train(
        scenario_ids,
        num_episodes=args.episodes,
        log_every=args.log_every,
        save_transcripts_every=args.save_transcripts_every,
        transcript_dir=args.transcript_dir,
    )
