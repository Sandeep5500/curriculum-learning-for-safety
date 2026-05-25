"""
Interpretability analysis for Curri-DPO: linear probing, Fisher Discriminant Ratio, UMAP.

Compares 5 model variants across all layers:
  raw, baseline DPO, curri_s0, curri_s1, curri_s2

Data: chosen/rejected pairs from curriculum_dataset_test.jsonl.
Extracts last-token hidden states at each layer, then:
  1. Linear probing — 5-fold CV logistic regression per layer (accuracy + AUC)
  2. Fisher Discriminant Ratio — centroid separation per layer
  3. UMAP — 2D projection at a target layer showing chosen vs rejected clusters

Usage:
  python -m src.interpretability.probe_safety \
    --base-model informatiker/Llama-3-8B-Instruct-abliterated \
    --exp-id exp_109 \
    --dataset-path data/processed/exp_073/combined_pku_hh/llama3/curriculum_dataset_test.jsonl \
    --output results/exp_109/combined_pku_hh/llama3/interpretability \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 \
    --n-samples 500
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_pairs(dataset_path: Path, n_samples: int | None = None) -> tuple[list[str], list[str]]:
    """Load chosen and rejected texts from JSONL dataset."""
    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append((d["chosen"], d["rejected"]))

    if n_samples is not None and n_samples < len(pairs):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), n_samples, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]

    chosen = [p[0] for p in pairs]
    rejected = [p[1] for p in pairs]
    print(f"Loaded {len(chosen)} chosen + {len(rejected)} rejected pairs from {dataset_path}")
    return chosen, rejected


def load_model(base_model: str, adapter_path: str | None):
    """Load model with optional LoRA adapter. Merges adapter for efficiency."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        from peft import PeftModel
        print(f"  Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """Extract last-token hidden states at specified layers.

    Returns dict mapping layer_idx -> numpy array of shape [n_prompts, hidden_dim].
    """
    import torch
    from tqdm import tqdm

    # Apply chat template where available
    formatted = []
    for p in prompts:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = p
        formatted.append(text)

    layer_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for i in tqdm(range(0, len(formatted), batch_size), desc="Extracting activations"):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors [batch, seq_len, hidden_dim]
        # Index 0 = embedding, index i+1 = after transformer layer i
        hidden_states = outputs.hidden_states

        # Find last real token (left-padded)
        seq_lens = inputs["attention_mask"].sum(dim=1)
        last_token_idx = seq_lens - 1

        max_layer = len(hidden_states) - 2  # hidden_states[0]=embeddings, so max valid layer = len-2
        for layer_idx in layers:
            if layer_idx > max_layer:
                continue
            hs = hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings
            batch_indices = torch.arange(hs.shape[0], device=hs.device)
            last_hidden = hs[batch_indices, last_token_idx]
            layer_acts[layer_idx].append(last_hidden.float().cpu().numpy())

    return {l: np.concatenate(arrs, axis=0) for l, arrs in layer_acts.items() if arrs}


def compute_linear_probe(
    chosen_acts: np.ndarray,
    rejected_acts: np.ndarray,
) -> dict[str, float]:
    """5-fold stratified CV logistic regression. Returns accuracy and AUC stats."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X = np.concatenate([chosen_acts, rejected_acts], axis=0)
    y = np.concatenate([np.ones(len(chosen_acts)), np.zeros(len(rejected_acts))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs = [], []

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y[train_idx])

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        accs.append((preds == y[val_idx]).mean())
        aucs.append(roc_auc_score(y[val_idx], probs))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def compute_fdr(chosen_acts: np.ndarray, rejected_acts: np.ndarray) -> float:
    """Fisher Discriminant Ratio: ||mu_c - mu_r||^2 / (var_c + var_r)."""
    mu_c = chosen_acts.mean(axis=0)
    mu_r = rejected_acts.mean(axis=0)
    var_c = chosen_acts.var(axis=0).sum()
    var_r = rejected_acts.var(axis=0).sum()
    numerator = float(np.sum((mu_c - mu_r) ** 2))
    denominator = float(var_c + var_r) + 1e-10
    return numerator / denominator


def run_probing(
    activations: dict[str, dict[int, np.ndarray]],
    n_chosen: int,
    n_rejected: int,
    layers: list[int],
) -> dict[str, Any]:
    """Run linear probing and FDR for all variants × layers."""
    results: dict[str, Any] = {}

    for vname, layer_acts in activations.items():
        print(f"  Probing variant: {vname}")
        variant_results = {}
        for layer_idx in sorted(layer_acts.keys()):
            acts = layer_acts[layer_idx]
            chosen_acts = acts[:n_chosen]
            rejected_acts = acts[n_chosen : n_chosen + n_rejected]

            probe = compute_linear_probe(chosen_acts, rejected_acts)
            fdr = compute_fdr(chosen_acts, rejected_acts)

            variant_results[layer_idx] = {**probe, "fdr": fdr}
            print(
                f"    layer {layer_idx:2d}: acc={probe['accuracy_mean']:.3f}±{probe['accuracy_std']:.3f} "
                f"auc={probe['auc_mean']:.3f} fdr={fdr:.2f}"
            )
        results[vname] = variant_results

    return results


def plot_probe_curves(
    probe_results: dict[str, Any],
    layers: list[int],
    output_dir: Path,
    variant_order: list[str],
) -> None:
    """Line plot: probe accuracy vs layer, one line per variant."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = cm.tab10(np.linspace(0, 0.9, len(variant_order)))

    for vname, color in zip(variant_order, colors):
        if vname not in probe_results:
            continue
        vr = probe_results[vname]
        accs = [vr[l]["accuracy_mean"] for l in layers if l in vr]
        acc_stds = [vr[l]["accuracy_std"] for l in layers if l in vr]
        aucs = [vr[l]["auc_mean"] for l in layers if l in vr]
        valid_layers = [l for l in layers if l in vr]

        ax1.plot(valid_layers, accs, label=vname, color=color, marker="o", ms=4)
        ax1.fill_between(
            valid_layers,
            [a - s for a, s in zip(accs, acc_stds)],
            [a + s for a, s in zip(accs, acc_stds)],
            alpha=0.15,
            color=color,
        )
        ax2.plot(valid_layers, aucs, label=vname, color=color, marker="o", ms=4)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe Accuracy (5-fold CV)")
    ax1.set_title("Linear Probe Accuracy vs Layer")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("AUC (5-fold CV)")
    ax2.set_title("Linear Probe AUC vs Layer")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Linear Probe: Chosen vs Rejected Representations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "probe_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_fdr_curve(
    probe_results: dict[str, Any],
    output_dir: Path,
    variant_order: list[str],
) -> None:
    """Line plot: FDR vs layer, one line per variant."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.tab10(np.linspace(0, 0.9, len(variant_order)))

    for vname, color in zip(variant_order, colors):
        if vname not in probe_results:
            continue
        vr = probe_results[vname]
        valid_layers = sorted(vr.keys())
        fdrs = [vr[l]["fdr"] for l in valid_layers]
        ax.plot(valid_layers, fdrs, label=vname, color=color, marker="o", ms=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Fisher Discriminant Ratio vs Layer\n(Chosen vs Rejected)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "fdr_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_umap(
    activations: dict[str, dict[int, np.ndarray]],
    n_chosen: int,
    n_rejected: int,
    target_layer: int,
    output_dir: Path,
    variant_order: list[str],
) -> None:
    """UMAP 2D projection at target layer, 5-panel figure."""
    import umap
    import matplotlib.pyplot as plt

    valid_variants = [v for v in variant_order if v in activations]
    n_panels = len(valid_variants)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Fit UMAP jointly on all variants for comparable projections
    all_acts = np.concatenate(
        [activations[v][target_layer] for v in valid_variants], axis=0
    )
    norms = np.linalg.norm(all_acts, axis=1, keepdims=True)
    all_acts_norm = all_acts / np.maximum(norms, 1e-8)

    print(f"  Fitting UMAP on {len(all_acts_norm)} points (all variants combined)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    all_emb = reducer.fit_transform(all_acts_norm)

    offset = 0
    n_per_variant = n_chosen + n_rejected
    for ax, vname in zip(axes, valid_variants):
        emb = all_emb[offset : offset + n_per_variant]
        offset += n_per_variant

        chosen_pts = emb[:n_chosen]
        rejected_pts = emb[n_chosen : n_chosen + n_rejected]

        ax.scatter(rejected_pts[:, 0], rejected_pts[:, 1], c="red", alpha=0.4, s=10, label="Rejected")
        ax.scatter(chosen_pts[:, 0], chosen_pts[:, 1], c="green", alpha=0.4, s=10, label="Chosen")
        ax.set_title(vname, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"UMAP at Layer {target_layer} — Chosen vs Rejected Representations",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / f"umap_layer{target_layer}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def run(
    base_model: str,
    exp_id: str,
    baseline_exp: str,
    model_short: str,
    dataset_path: Path,
    output_dir: Path,
    layers: list[int],
    n_samples: int | None,
    batch_size: int,
    umap_layer: int,
    repo_root: Path,
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve adapter paths
    curri_dir = repo_root / "outputs" / exp_id / "combined_pku_hh" / model_short / "dpo_curriculum"
    baseline_adapter = str(repo_root / "outputs" / baseline_exp / "combined_pku_hh" / model_short / "dpo_baseline")

    # Read curri base model (= stage_1/merged)
    curri_base_txt = curri_dir / "curri_base_model.txt"
    curri_base = str(curri_base_txt.read_text().strip()) if curri_base_txt.exists() else base_model

    stage0_adapter = str(curri_dir / "stage_0" / "checkpoint-460")
    stage1_base = str(curri_dir / "stage_1" / "merged")
    stage1_adapter = str(curri_dir / "stage_1" / "checkpoint-460")
    final_adapter = str(curri_dir)

    variants = [
        ("raw",       base_model,    None),
        ("baseline",  base_model,    baseline_adapter),
        ("curri_s0",  base_model,    stage0_adapter),
        ("curri_s1",  stage1_base,   stage1_adapter),
        ("curri_s2",  curri_base,    final_adapter),
    ]
    variant_order = [v[0] for v in variants]

    # Load data
    chosen, rejected = load_pairs(dataset_path, n_samples)
    all_prompts = chosen + rejected
    n_chosen = len(chosen)
    n_rejected = len(rejected)

    # Check cache
    cache_path = output_dir / "activations.npz"
    if cache_path.exists():
        print(f"Loading cached activations from {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        activations: dict[str, dict[int, np.ndarray]] = {}
        for vname in variant_order:
            activations[vname] = {}
            for l in layers:
                key = f"{vname}_layer{l}"
                if key in data:
                    activations[vname][l] = data[key]
    else:
        activations = {}
        for vname, base, adapter in variants:
            print(f"\n=== Extracting: {vname} (base={base}) ===")
            if adapter and not Path(adapter).exists():
                print(f"  WARNING: adapter not found at {adapter}, skipping.")
                continue
            model, tokenizer = load_model(base, adapter)
            acts = extract_activations(model, tokenizer, all_prompts, layers, batch_size)
            activations[vname] = acts
            del model
            torch.cuda.empty_cache()

        # Save activations
        np_data: dict[str, np.ndarray] = {}
        for vname, layer_acts in activations.items():
            for l, arr in layer_acts.items():
                np_data[f"{vname}_layer{l}"] = arr
        np_data["n_chosen"] = np.array([n_chosen])
        np_data["n_rejected"] = np.array([n_rejected])
        np.savez_compressed(cache_path, **np_data)
        print(f"\nSaved activations to {cache_path}")

    # Run probing analyses
    print("\n=== Running Linear Probing + FDR ===")
    probe_results = run_probing(activations, n_chosen, n_rejected, layers)

    # Save probe results
    probe_output: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "exp_id": exp_id,
        "n_chosen": n_chosen,
        "n_rejected": n_rejected,
        "layers": layers,
        "variants": variant_order,
    }
    for vname, vr in probe_results.items():
        probe_output[vname] = {str(l): stats for l, stats in vr.items()}

    probe_json = output_dir / "probe_results.json"
    with open(probe_json, "w") as f:
        json.dump(probe_output, f, indent=2)
    print(f"Saved probe results to {probe_json}")

    # Plots
    print("\n=== Generating plots ===")
    plot_probe_curves(probe_results, layers, output_dir, variant_order)
    plot_fdr_curve(probe_results, output_dir, variant_order)

    try:
        import umap as _umap_check  # noqa: F401
        plot_umap(activations, n_chosen, n_rejected, umap_layer, output_dir, variant_order)
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plot.")

    # Print summary table
    print("\n=== Summary: Probe Accuracy at Final Layer ===")
    final_layer = layers[-1]
    print(f"{'Variant':<12} {'Acc':>6} {'AUC':>6} {'FDR':>8}")
    print("-" * 36)
    for vname in variant_order:
        if vname in probe_results and final_layer in probe_results[vname]:
            s = probe_results[vname][final_layer]
            print(
                f"{vname:<12} {s['accuracy_mean']:.3f}  {s['auc_mean']:.3f}  {s['fdr']:8.2f}"
            )

    print(f"\nAll outputs saved to: {output_dir}")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Interpretability analysis: linear probing, FDR, UMAP.")
    p.add_argument("--base-model", default="informatiker/Llama-3-8B-Instruct-abliterated")
    p.add_argument("--exp-id", default="exp_109", help="Curri-DPO experiment ID")
    p.add_argument("--baseline-exp", default="exp_098", help="Baseline DPO experiment ID")
    p.add_argument("--model-short", default="llama3", help="Model short name used in path (e.g. llama3, qwen3, qwen3_4b)")
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=root / "data/processed/exp_073/combined_pku_hh/llama3/curriculum_dataset_test.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "results/exp_109/combined_pku_hh/llama3/interpretability",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(0, 32, 2)),  # 0,2,4,...,30
        help="Layer indices to probe",
    )
    p.add_argument("--n-samples", type=int, default=None, help="Subset size (default: all)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--umap-layer", type=int, default=28, help="Layer for UMAP + FDR bar chart")
    args = p.parse_args()

    run(
        base_model=args.base_model,
        exp_id=args.exp_id,
        baseline_exp=args.baseline_exp,
        model_short=args.model_short,
        dataset_path=args.dataset_path,
        output_dir=args.output,
        layers=args.layers,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        umap_layer=args.umap_layer,
        repo_root=root,
    )


if __name__ == "__main__":
    cli()
