from role_drift_env.server.environment import RoleDriftEnvironment

env = RoleDriftEnvironment()
obs, state = env.reset('term_kk_01', 0)
prompt = obs.system_prompt
print(f'Prompt loaded for term_kk_01: {len(prompt)} chars, {len(prompt.split())} words')
print('First 200 chars:', prompt[:200])