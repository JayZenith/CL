"""Sequential full-parameter SFT: python train.py config.yaml [key=value ...]"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from data import encode, load_config, load_tokenizer
from swr import build_swr

TRAIN_FIELDS = ["task", "task_name", "step", "task_step", "loss", "grad_norm", "num_reinit"]
EVAL_FIELDS = ["trained_task", "task_step", "eval_task", "nll", "token_acc", "exact_match"]


def collate(examples, pad_id, device):
    length = max(len(ids) for ids, _ in examples)
    ids = [ids + [pad_id] * (length - len(ids)) for ids, _ in examples]
    labels = [labels + [-100] * (length - len(labels)) for _, labels in examples]
    mask = [[1] * len(e[0]) + [0] * (length - len(e[0])) for e in examples]
    return [torch.tensor(x, device=device) for x in (ids, mask, labels)]


def score(model, batch, bf16):
    """Runs lm_head on target positions only; full logits would cost 150k * length * batch floats."""
    ids, mask, labels = batch
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=bf16):
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
        keep = labels[:, 1:] != -100
        logits = model.lm_head(hidden[keep]).float()
    targets = labels[:, 1:][keep]
    return F.cross_entropy(logits, targets, reduction="none"), logits.argmax(-1) == targets, keep.nonzero()[:, 0]


@torch.no_grad()
def evaluate(model, examples, cfg, pad_id, device):
    """With teacher forcing, all target tokens correct is equivalent to a greedy exact match."""
    model.eval()
    examples = sorted(examples, key=lambda e: len(e[0]))
    nll = hits = tokens = exact = 0
    for i in range(0, len(examples), cfg["eval_batch_size"]):
        chunk = examples[i:i + cfg["eval_batch_size"]]
        losses, correct, rows = score(model, collate(chunk, pad_id, device), cfg["bf16"])
        nll += losses.sum().item()
        hits += correct.sum().item()
        tokens += correct.numel()
        exact += (torch.bincount(rows[~correct], minlength=len(chunk)) == 0).sum().item()
    model.train()
    return nll / tokens, hits / tokens, exact / len(examples)


def truncate(path, key, stop):
    """Drops rows from tasks that the checkpoint does not contain."""
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(r for r in rows if int(r[key]) < stop)


def open_log(path, fields):
    is_new = not path.exists() or path.stat().st_size == 0
    writer = csv.writer(open(path, "a", newline="", buffering=1))
    if is_new:
        writer.writerow(fields)
    return writer


def main():
    cfg = load_config(sys.argv[1:])
    run_dir = Path(cfg["output_dir"]) / f"{cfg['run_name']}_s{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    pad_id = tokenizer.pad_token_id
    tasks_bytes = Path(cfg["data"]["path"]).read_bytes()
    tasks = json.loads(tasks_bytes)
    for task in tasks:
        task["train"] = [encode(tokenizer, e) for e in task["train"]]
        task["eval"] = [encode(tokenizer, e) for e in task["eval"]]

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # As in the paper: weight decay on parameters with at least 2 dimensions.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg["weight_decay"]},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg["lr"], fused=device.type == "cuda")
    swr = build_swr(model, cfg["swr"]) if cfg["swr"]["enabled"] else None

    start_task, step = 0, 0
    ckpt_path = run_dir / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if swr:
            swr.load_state_dict(ckpt["swr"])
        start_task, step = ckpt["next_task"], ckpt["step"]
        torch.set_rng_state(ckpt["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        print(f"Resumed at task {start_task}")
    truncate(run_dir / "train.csv", "task", start_task)
    truncate(run_dir / "eval.csv", "trained_task", start_task)
    train_log = open_log(run_dir / "train.csv", TRAIN_FIELDS)
    eval_log = open_log(run_dir / "eval.csv", EVAL_FIELDS)

    (run_dir / "config.json").write_text(json.dumps({
        **cfg, "tasks_sha256": hashlib.sha256(tasks_bytes).hexdigest(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda}, indent=2))

    def log_eval(task, task_step, eval_task):
        metrics = evaluate(model, tasks[eval_task]["eval"], cfg, pad_id, device)
        eval_log.writerow([task, task_step, eval_task, *(f"{m:.5f}" for m in metrics)])

    last_task = len(tasks) - 1
    micro_batches = cfg["steps_per_task"] * cfg["grad_accum"]
    for t in range(start_task, len(tasks)):
        train = tasks[t]["train"]
        # Depends only on seed and task, so SFT and SWR runs see identical batches.
        generator = torch.Generator().manual_seed(cfg["seed"] * 100_003 + t)
        needed = micro_batches * cfg["batch_size"]
        order = torch.cat([torch.randperm(len(train), generator=generator)
                           for _ in range(needed // len(train) + 1)])[:needed].tolist()

        log_eval(t, 0, t)
        for task_step in range(1, cfg["steps_per_task"] + 1):
            loss_sum = 0.0
            for micro in range(cfg["grad_accum"]):
                offset = ((task_step - 1) * cfg["grad_accum"] + micro) * cfg["batch_size"]
                chunk = [train[i] for i in order[offset:offset + cfg["batch_size"]]]
                loss = score(model, collate(chunk, pad_id, device), cfg["bf16"])[0].mean() / cfg["grad_accum"]
                loss.backward()
                loss_sum += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            optimizer.step()
            num_reinit = 0
            if swr:  # After the AdamW step and before zero_grad, as gradient utility reads p.grad.
                swr.step()
                num_reinit = swr.num_replaced
                swr.reset_reinit_indicator()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            train_log.writerow([t, tasks[t]["name"], step, task_step, f"{loss_sum:.5f}", f"{grad_norm:.4f}", num_reinit])
            if task_step % cfg["eval_curve_every"] == 0 or task_step == cfg["steps_per_task"]:
                log_eval(t, task_step, t)

        if (t + 1) % cfg["eval_seen_every"] == 0 or t == last_task:
            for seen in range(t):
                log_eval(t, cfg["steps_per_task"], seen)

        if (t + 1) % cfg["ckpt_every"] == 0 or t == last_task:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "swr": swr.state_dict() if swr else None, "next_task": t + 1, "step": step,
                    "rng": torch.get_rng_state()}
            if device.type == "cuda":
                ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
            torch.save(ckpt, run_dir / "latest.tmp")
            (run_dir / "latest.tmp").replace(ckpt_path)
        print(f"task {t + 1}/{len(tasks)} {tasks[t]['name']} step {step}", flush=True)


if __name__ == "__main__":
    main()
