$ErrorActionPreference = "Stop"

function Assert-LastExit($stepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$stepName failed with exit code $LASTEXITCODE"
    }
}

$RepoUrl = if ($env:REPO_URL) { $env:REPO_URL } else { "https://github.com/GeniusPlums/OpenEnv-Finale.git" }
$WorkDir = if ($env:WORKDIR) { $env:WORKDIR } else { "$env:TEMP\role-drift-env-hf-smoke" }
$VllmModel = if ($env:VLLM_MODEL) { $env:VLLM_MODEL } else { "Qwen/Qwen2.5-7B-Instruct" }
$PolicyModel = if ($env:POLICY_MODEL) { $env:POLICY_MODEL } else { "Qwen/Qwen2.5-1.5B-Instruct" }

Write-Host "[1/7] Clone + install"
if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
git clone $RepoUrl $WorkDir
Assert-LastExit "git clone"
Set-Location $WorkDir

python -m pip install -e .
Assert-LastExit "pip install -e ."
python -m pip install vllm sentence-transformers langdetect peft accelerate bitsandbytes
Assert-LastExit "pip install dependencies"

Write-Host "[2/7] GPU info"
nvidia-smi
Assert-LastExit "nvidia-smi"

Write-Host "[3/7] Start vLLM"
$vllmArgs = @(
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", $VllmModel,
    "--port", "8000",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.45"
)
$vllmLog = "$env:TEMP\vllm.log"
if (Test-Path $vllmLog) { Remove-Item -Force $vllmLog }
$vllmErr = "$env:TEMP\vllm.err.log"
if (Test-Path $vllmErr) { Remove-Item -Force $vllmErr }
$proc = Start-Process -FilePath "python" -ArgumentList $vllmArgs -RedirectStandardOutput $vllmLog -RedirectStandardError $vllmErr -PassThru
Start-Sleep -Seconds 120

Write-Host "[4/7] Check vLLM /v1/models"
try {
    $models = Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -Method Get -TimeoutSec 30
    $models | ConvertTo-Json -Depth 6
} catch {
    Write-Host "vLLM /v1/models check failed. Last vLLM log lines:"
    if (Test-Path $vllmLog) { Get-Content $vllmLog -Tail 120 }
    throw
}

Write-Host "[5/7] Persona fallback check"
python -c "from role_drift_env.server.personas.llm_backed import LLMPersona; from role_drift_env.server.environment import RoleDriftEnvironment; env=RoleDriftEnvironment(); obs,state=env.reset('term_kk_01'); p=LLMPersona(persona_id='hh_probe', system_prompt='You are a polite customer.', model='Qwen/Qwen2.5-7B-Instruct'); out=p.next_utterance(state, rng_seed=42); print('PERSONA OUTPUT:', out[:200]); print('IS_FALLBACK:', 'I think I have what I need' in out)"
Assert-LastExit "persona fallback check"

Write-Host "[6/7] 2-episode timing smoke"
python training/train_grpo.py `
  --episodes 2 `
  --group-size 4 `
  --lr 5e-6 `
  --kl-coef 0.125 `
  --curriculum adversarial `
  --policy-model $PolicyModel `
  --checkpoint-every 1 `
  --output-dir "$env:TEMP\smoke_real" `
  --checkpoint-dir "$env:TEMP\smoke_ckpt" `
  --time-each-episode
Assert-LastExit "2-episode timing smoke"

Write-Host "[7/7] Summarize smoke outputs"
python -c "import json, pathlib, os; p=pathlib.Path(os.path.join(os.environ['TEMP'],'smoke_real','episode_log.jsonl')); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; t=sum(r.get('episode_seconds',0.0) for r in rows)/len(rows); cost=(100*t/3600)*4.0; print(f'episodes={len(rows)} mean_episode_seconds={t:.2f}'); print(f'projected_cost_100ep_usd={cost:.2f}'); print(f'avg_kl={sum(r.get(\"avg_kl\",0.0) for r in rows)/len(rows):.4f}'); print(f'component_means_present={all(\"component_means\" in r for r in rows)}')"
Assert-LastExit "summary parse"

Write-Host "Mini-smoke complete."
