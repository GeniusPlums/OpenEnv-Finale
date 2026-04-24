#!/usr/bin/env python3
"""Quick speed test: can Qwen 1.5B run on CPU?"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from role_drift_env.server.environment import RoleDriftEnvironment
from role_drift_env.server.customer_sim import CustomerSimulator
from role_drift_env.models import AgentAction

# Load model
print("Loading model...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True).to("cpu")
model.eval()
print(f"Model loaded in {time.time()-start:.1f}s")

# Test generation speed
prompt = "User: Hello, how can you help me?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
print("Testing generation speed...")
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
elapsed = time.time() - start
print(f"Generated 30 tokens in {elapsed:.2f}s ({30/elapsed:.1f} tokens/sec)")

# Test 1 full episode
print("\nTesting 1 full episode...")
env = RoleDriftEnvironment()
obs, state = env.reset(scenario_id="goal_kk_01", rollout_idx=0)
sim = CustomerSimulator.from_scenario(state.scenario)

start = time.time()
turn_count = 0
while turn_count < 5:  # Just 5 turns
    # Format prompt
    messages = [{"role": "system", "content": obs.system_prompt or "You are helpful."}]
    for turn in state.history:
        role = "user" if turn["role"] == "customer" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    messages.append({"role": "user", "content": obs.customer_message})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    action = AgentAction(utterance=text)
    obs, reward, done, info = env.step(state, action, sim)
    turn_count += 1
    if done:
        break

elapsed = time.time() - start
print(f"5 turns completed in {elapsed:.1f}s ({elapsed/5:.2f}s per turn)")
print(f"Estimated time for 20 episodes @ 10 turns = {20*10*elapsed/5/60:.1f} minutes")