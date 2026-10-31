# SWR on Qwen3

Tests Selective Weight Reinitialization on sequential full-parameter SFT of `Qwen/Qwen3-0.6B-Base` over 100 SuperNI tasks.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Run

```bash
python data.py config.yaml    # writes data/tasks.json

for seed in 0 1 2; do
  python train.py config.yaml run_name=sft seed=$seed
  python train.py config.yaml run_name=swr seed=$seed swr.enabled=true
done

python analyze.py runs/*_s*
```

Any config key can be overridden with `key=value`, for example `swr.reinit_factor=0.001`. Rerun a command to resume from `latest.pt`.

## Outputs

`runs/<name>_s<seed>/` holds `train.csv`, `eval.csv`, `config.json` and `latest.pt`. `runs/summary.csv` holds per-run metrics; the definitions are at the top of `analyze.py`.

## Notes

- SWR runs on `mlp.{gate,up,down}_proj.weight` after each AdamW step, and resamples from Qwen3's init, N(0, 0.02).
- As in the paper, AdamW moments of reinitialized weights are not reset.
- SWR defaults come from the paper's ViT experiment, and are not tuned for Qwen.
- Parameters stay float32 under BF16 autocast, because small AdamW updates round away in BF16 weights.
- Checkpoints are about 7 GB. Resuming restarts from the last finished task.
