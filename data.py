"""Builds the sequential SuperNI tasks: python data.py config.yaml [key=value ...]"""
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml


def load_config(args):
    cfg = yaml.safe_load(Path(args[0]).read_text())
    for arg in args[1:]:
        key, value = arg.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for parent in parents:
            node = node[parent]
        assert leaf in node, f"Unknown config key: {key}"
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])


def encode(tokenizer, example):
    """Base-model prompt. Only target tokens, including EOS, are scored."""
    prompt = tokenizer(f"Definition: {example['definition']}\n\nInput: {example['inputs']}\n\nOutput:").input_ids
    target = tokenizer(" " + example["targets"]).input_ids + [tokenizer.eos_token_id]
    return prompt + target, [-100] * len(prompt) + target


def build_task(path, tokenizer, data, rng):
    """Returns a disjoint split, or None if the task is too small."""
    unique = {}
    for line in open(path):
        example = json.loads(line)
        # Inputs repeat with different targets, so unique inputs keep train and eval disjoint.
        key = example["inputs"].strip()
        if key not in unique and example["targets"].strip():
            unique[key] = example
    examples = [{k: e[k] for k in ("definition", "inputs", "targets")} for e in unique.values()
                if len(encode(tokenizer, e)[0]) <= data["max_len"]]
    if len(examples) < data["n_train"] + data["n_eval"]:
        return None
    rng.shuffle(examples)
    return {"eval": examples[:data["n_eval"]],
            "train": examples[data["n_eval"]:data["n_eval"] + data["n_train"]]}


def main():
    from huggingface_hub import HfApi, hf_hub_download

    cfg = load_config(sys.argv[1:])
    data = cfg["data"]
    tokenizer = load_tokenizer(cfg)
    repo = dict(repo_id=data["dataset"], repo_type="dataset", revision=data["dataset_revision"])
    # train/ holds the 757 tasks that have enough instances for per-task splits;
    # test/ holds 119 different tasks.
    files = sorted(f for f in HfApi().list_repo_files(**repo) if f.startswith("train/") and f.endswith(".jsonl"))
    rng = random.Random(data["seed"])
    rng.shuffle(files)

    tasks = []
    for file in files:
        task = build_task(hf_hub_download(filename=file, **repo), tokenizer, data, rng)
        if task:
            tasks.append({"name": Path(file).stem.removesuffix("_train"), **task})
            print(f"{len(tasks):3d}/{data['n_tasks']} {tasks[-1]['name']}")
        if len(tasks) == data["n_tasks"]:
            break
    assert len(tasks) == data["n_tasks"], f"Only {len(tasks)} eligible tasks"

    path = Path(data["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks))
    print(f"{path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
