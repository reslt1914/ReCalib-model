import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_matrix(path: Path):
    # Python engine avoids rare pandas C-parser crashes on very wide CSV files.
    df = pd.read_csv(path, index_col=0, engine="python")
    return df.index.astype(str).to_numpy(), df.columns.astype(str).to_numpy(), df.to_numpy(dtype=np.float32)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def load_split(base_dir: Path, split: str):
    d = base_dir / split
    conds, genes, x_true = read_matrix(d / "x_true_condition_mean.csv")
    _, _, x_pred = read_matrix(d / "x_crisp_condition_mean.csv")
    _, _, x_ctrl = read_matrix(d / "x_ctrl_condition_mean.csv")
    _, _, residual = read_matrix(d / "residual_condition_mean.csv")

    meta = pd.read_csv(d / "condition_metadata.csv")
    meta["condition"] = meta["condition"].astype(str)
    meta = meta.set_index("condition").loc[conds].reset_index()

    return {
        "split": split,
        "conds": conds,
        "genes": genes,
        "true": x_true,
        "pred": x_pred,
        "ctrl": x_ctrl,
        "residual": residual,
        "meta": meta,
    }


def build_drug_features(train_meta, test_meta):
    """Build drug identity features from the fitting split only.

    Drugs not observed in the fitting split are represented by an
    all-zero drug vector. This avoids introducing untrained one-hot
    dimensions for truly unseen drugs at validation/OOD inference.
    """
    drugs = sorted(
        set(train_meta["drug"].astype(str).tolist())
    )
    drug_to_id = {
        drug: index
        for index, drug in enumerate(drugs)
    }

    def encode(meta, split_name):
        x = np.zeros(
            (len(meta), len(drugs)),
            dtype=np.float32,
        )

        unknown = []

        for i, drug in enumerate(
            meta["drug"].astype(str)
        ):
            index = drug_to_id.get(drug)

            if index is None:
                unknown.append(drug)
                continue

            x[i, index] = 1.0

        if unknown:
            unique_unknown = sorted(set(unknown))
            print(
                f"[DRUG] {split_name}: unseen-to-fitting "
                f"drugs encoded as all-zero one-hot: "
                f"{unique_unknown}"
            )
            print(
                f"[DRUG] {split_name}: affected conditions="
                f"{len(unknown)}"
            )

        return x

    return (
        encode(train_meta, "train"),
        encode(test_meta, "test"),
        drugs,
    )


def build_gene_alias_index(adata, exported_genes: np.ndarray) -> Dict[str, int]:
    """
    Map gene symbols / var_names / gene_id aliases to exported matrix column index.

    export_crisp_baseline_residual.py may use var["gene_id"] or var_names as columns.
    rank_genes_groups_cov often stores gene symbols. This function bridges them.
    """
    alias_to_idx: Dict[str, int] = {}

    for i, g in enumerate(exported_genes):
        alias_to_idx[str(g)] = i

    # Add adata.var_names aliases by position.
    if adata.n_vars == len(exported_genes):
        for i, g in enumerate(adata.var_names.astype(str).tolist()):
            alias_to_idx[str(g)] = i

        for col in adata.var.columns:
            if col in ["gene_id", "gene_name", "symbol", "gene_symbols", "features"]:
                vals = adata.var[col].astype(str).tolist()
                for i, g in enumerate(vals):
                    alias_to_idx[str(g)] = i

    return alias_to_idx


def get_rank_key_variants(cond: str) -> List[str]:
    variants = [cond]
    # Conservative variants only; do not invent too much.
    variants.append(cond.replace("__", "_"))
    variants.append(cond.replace(" ", " "))
    return list(dict.fromkeys(variants))


def build_de_masks(adata_path: str, train, test, de_key: str = "rank_genes_groups_cov"):
    adata = sc.read_h5ad(adata_path, backed="r")
    if de_key not in adata.uns:
        raise KeyError(f"{de_key} not found in adata.uns. Available: {list(adata.uns.keys())}")

    rg = adata.uns[de_key]
    if not isinstance(rg, dict):
        raise TypeError(f"Expected adata.uns[{de_key}] to be dict, got {type(rg)}")

    genes = train["genes"]
    alias_to_idx = build_gene_alias_index(adata, genes)

    def mask_for_split(split_data):
        masks = np.zeros((len(split_data["conds"]), len(genes)), dtype=bool)
        coverage = []

        for i, cond in enumerate(split_data["conds"]):
            found_key = None
            for key in get_rank_key_variants(str(cond)):
                if key in rg:
                    found_key = key
                    break

            if found_key is None:
                coverage.append((cond, 0, 0, "missing_condition"))
                continue

            de_genes = list(rg[found_key])
            hit = 0
            for g in de_genes:
                gs = str(g)
                if gs in alias_to_idx:
                    masks[i, alias_to_idx[gs]] = True
                    hit += 1
            coverage.append((cond, len(de_genes), hit, "ok"))

        cov = pd.DataFrame(coverage, columns=["condition", "n_de_listed", "n_de_mapped", "status"])
        return masks, cov

    train_mask, train_cov = mask_for_split(train)
    test_mask, test_cov = mask_for_split(test)

    return train_mask, test_mask, train_cov, test_cov


class ResidualCalibrator(nn.Module):
    def __init__(self, n_genes, drug_dim, gene_emb_dim=64, hidden=256, dropout=0.05):
        super().__init__()
        self.gene_emb = nn.Embedding(n_genes, gene_emb_dim)

        # scalar features per condition-gene:
        # crisp_pred, ctrl, delta_pred, abs_delta_pred, rel_delta
        scalar_dim = 5
        in_dim = gene_emb_dim + drug_dim + scalar_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, gene_idx, drug_feat, scalar_feat):
        ge = self.gene_emb(gene_idx)
        x = torch.cat([ge, drug_feat, scalar_feat], dim=1)
        return self.net(x).squeeze(1)


def make_pairs(n_conditions, n_genes):
    c = np.repeat(np.arange(n_conditions, dtype=np.int64), n_genes)
    g = np.tile(np.arange(n_genes, dtype=np.int64), n_conditions)
    return c, g


def eval_by_condition(split_name, model_name, true, pred, ctrl, conds, meta, de_mask, topk=50):
    rows = []
    for i, cond in enumerate(conds):
        y = true[i]
        p = pred[i]
        c = ctrl[i]

        delta_true = y - c
        delta_pred = p - c

        official_idx = np.where(de_mask[i])[0]
        if len(official_idx) == 0:
            official_idx = np.argsort(np.abs(delta_true))[::-1][:topk]

        top_idx = np.argsort(np.abs(delta_true))[::-1][:topk]

        row = {
            "split": split_name,
            "model": model_name,
            "condition": cond,
            "cell_type": str(meta.iloc[i].get("cell_type", "")),
            "drug": str(meta.iloc[i].get("drug", "")),
            "n_de_official": int(len(official_idx)),
            "mse": mse(y, p),
            "mse_de": mse(y[official_idx], p[official_idx]),
            "pearson": pearson(y, p),
            "pearson_de": pearson(y[official_idx], p[official_idx]),
            "pearson_delta": pearson(delta_true, delta_pred),
            "pearson_delta_de": pearson(delta_true[official_idx], delta_pred[official_idx]),
            "r2score": safe_r2(y, p),
            "r2score_de": safe_r2(y[official_idx], p[official_idx]),
            "r2score_delta": safe_r2(delta_true, delta_pred),
            "r2score_delta_de": safe_r2(delta_true[official_idx], delta_pred[official_idx]),
            # Extra top-k metrics for diagnosis
            "pearson_topk_delta": pearson(delta_true[top_idx], delta_pred[top_idx]),
            "r2score_topk_delta": safe_r2(delta_true[top_idx], delta_pred[top_idx]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, default="crisp_outputs/baseline_small50k")
    parser.add_argument("--adata", type=str, default="data/nips/nips_pp_scFM_resplit_small50k.h5ad")
    parser.add_argument("--de_key", type=str, default="rank_genes_groups_cov")
    parser.add_argument("--out_dir", type=str, default="crisp_outputs/rc_deaware_small50k_alpha005")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gene_emb_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--de_weight", type=float, default=30.0)
    parser.add_argument("--l2_pred_residual", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max_train_pairs", type=int, default=0, help="0 means all pairs. Otherwise randomly sample this many pairs per epoch.")
    parser.add_argument("--shuffle_residual", action="store_true", help="Negative control: shuffle IID residuals across conditions before training.")
    parser.add_argument("--shuffle_residual_seed", type=int, default=2024)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    base_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(base_dir, "iid")
    test = load_split(base_dir, "ood")

    assert list(train["genes"]) == list(test["genes"])
    genes = train["genes"]
    n_genes = len(genes)

    train_de_mask, test_de_mask, train_cov, test_cov = build_de_masks(args.adata, train, test, de_key=args.de_key)
    train_cov.to_csv(out_dir / "de_mapping_iid.csv", index=False)
    test_cov.to_csv(out_dir / "de_mapping_ood.csv", index=False)

    print("DE mapping IID:")
    print(train_cov["n_de_mapped"].describe())
    print("DE mapping OOD:")
    print(test_cov["n_de_mapped"].describe())

    if train_cov["n_de_mapped"].sum() == 0:
        raise RuntimeError("No IID DE genes mapped. Check gene names / adata var columns.")
    if test_cov["n_de_mapped"].sum() == 0:
        print("[WARN] No OOD DE genes mapped; evaluation will fallback to top-k.")

    # NEGATIVE CONTROL: shuffle IID residual targets across conditions.
    # This keeps CRISP predictions, controls, drugs and DE masks unchanged,
    # but destroys the condition-specific residual correspondence.
    if args.shuffle_residual:
        rng_shuffle = np.random.default_rng(args.shuffle_residual_seed)
        perm = rng_shuffle.permutation(len(train["conds"]))
        pd.DataFrame({
            "target_condition": train["conds"],
            "source_residual_condition": train["conds"][perm],
        }).to_csv(out_dir / "shuffle_residual_permutation.csv", index=False)
        train["residual"] = train["residual"][perm].copy()
        print("[NEGATIVE CONTROL] Shuffled IID residuals across conditions.")
        print("[NEGATIVE CONTROL] shuffle_residual_seed =", args.shuffle_residual_seed)

    train_drug, test_drug, drugs = build_drug_features(train["meta"], test["meta"])

    print("=" * 80)
    print("DE-aware CRISP-RC residual calibrator")
    print("baseline_dir:", base_dir.resolve())
    print("out_dir     :", out_dir.resolve())
    print("device      :", device)
    print("train iid conditions:", len(train["conds"]))
    print("test ood conditions :", len(test["conds"]))
    print("genes:", n_genes)
    print("drugs:", len(drugs))
    print("alpha:", args.alpha)
    print("de_weight:", args.de_weight)
    print("=" * 80)

    model = ResidualCalibrator(
        n_genes=n_genes,
        drug_dim=train_drug.shape[1],
        gene_emb_dim=args.gene_emb_dim,
        hidden=args.hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_pred = torch.tensor(train["pred"], dtype=torch.float32, device=device)
    train_ctrl = torch.tensor(train["ctrl"], dtype=torch.float32, device=device)
    train_resid = torch.tensor(train["residual"], dtype=torch.float32, device=device)
    train_de = torch.tensor(train_de_mask, dtype=torch.bool, device=device)
    train_drug_t = torch.tensor(train_drug, dtype=torch.float32, device=device)

    all_c, all_g = make_pairs(len(train["conds"]), n_genes)
    n_pairs = len(all_c)
    rng = np.random.default_rng(args.seed)

    best_loss = float("inf")
    best_state = None
    bad = 0
    patience = 40

    for epoch in range(1, args.epochs + 1):
        model.train()

        if args.max_train_pairs and args.max_train_pairs > 0 and args.max_train_pairs < n_pairs:
            epoch_idx = rng.choice(n_pairs, size=args.max_train_pairs, replace=False)
        else:
            epoch_idx = rng.permutation(n_pairs)

        losses = []

        for st in range(0, len(epoch_idx), args.batch_size):
            idx = epoch_idx[st: st + args.batch_size]
            c_np = all_c[idx]
            g_np = all_g[idx]

            c = torch.tensor(c_np, dtype=torch.long, device=device)
            g = torch.tensor(g_np, dtype=torch.long, device=device)

            pred = train_pred[c, g]
            ctrl = train_ctrl[c, g]
            delta = pred - ctrl
            rel_delta = delta / (torch.abs(ctrl) + 1e-3)

            scalar = torch.stack([
                pred,
                ctrl,
                delta,
                torch.abs(delta),
                rel_delta,
            ], dim=1)

            target = train_resid[c, g]
            drug_feat = train_drug_t[c]

            pred_resid = model(g, drug_feat, scalar)

            is_de = train_de[c, g].float()

            # Core DE-aware weighting.
            # DE points get a large constant weight.
            w = torch.ones_like(target)
            w = w + args.de_weight * is_de

            # Add mild residual/delta weighting, but keep it bounded.
            w = w + 0.5 * torch.clamp(torch.abs(target) / (torch.abs(target).mean().detach() + 1e-6), 0.0, 4.0)
            w = w + 0.25 * torch.clamp(torch.abs(delta) / (torch.abs(delta).mean().detach() + 1e-6), 0.0, 4.0)

            loss = (w * (pred_resid - target) ** 2).mean()
            loss = loss + args.l2_pred_residual * (pred_resid ** 2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(losses))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:04d} train_loss={avg_loss:.6f} best={best_loss:.6f}")

        if bad >= patience:
            print("early stopping at epoch", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_residual(split_data, drug_np):
        model.eval()
        pred_mat = split_data["pred"]
        ctrl_mat = split_data["ctrl"]
        n_cond = len(split_data["conds"])
        out = np.zeros_like(pred_mat, dtype=np.float32)

        pred_t = torch.tensor(pred_mat, dtype=torch.float32, device=device)
        ctrl_t = torch.tensor(ctrl_mat, dtype=torch.float32, device=device)
        drug_t = torch.tensor(drug_np, dtype=torch.float32, device=device)

        c_all, g_all = make_pairs(n_cond, n_genes)
        with torch.no_grad():
            for st in range(0, len(c_all), args.batch_size):
                c_np = c_all[st: st + args.batch_size]
                g_np = g_all[st: st + args.batch_size]

                c = torch.tensor(c_np, dtype=torch.long, device=device)
                g = torch.tensor(g_np, dtype=torch.long, device=device)

                pred = pred_t[c, g]
                ctrl = ctrl_t[c, g]
                delta = pred - ctrl
                rel_delta = delta / (torch.abs(ctrl) + 1e-3)

                scalar = torch.stack([
                    pred,
                    ctrl,
                    delta,
                    torch.abs(delta),
                    rel_delta,
                ], dim=1)

                pr = model(g, drug_t[c], scalar).detach().cpu().numpy()
                out[c_np, g_np] = pr.astype(np.float32)

        return out

    pred_resid_iid = predict_residual(train, train_drug)
    pred_resid_ood = predict_residual(test, test_drug)

    x_cal_iid = train["pred"] + args.alpha * pred_resid_iid
    x_cal_ood = test["pred"] + args.alpha * pred_resid_ood

    for split_name, split_data, pred_resid, x_cal in [
        ("iid", train, pred_resid_iid, x_cal_iid),
        ("ood", test, pred_resid_ood, x_cal_ood),
    ]:
        sd = out_dir / split_name
        sd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_resid, index=split_data["conds"], columns=genes).to_csv(sd / "predicted_residual_condition_mean.csv")
        pd.DataFrame(x_cal, index=split_data["conds"], columns=genes).to_csv(sd / "x_calibrated_condition_mean.csv")

    m_base_iid = eval_by_condition("iid", "CRISP", train["true"], train["pred"], train["ctrl"], train["conds"], train["meta"], train_de_mask, topk=args.topk)
    m_base_ood = eval_by_condition("ood", "CRISP", test["true"], test["pred"], test["ctrl"], test["conds"], test["meta"], test_de_mask, topk=args.topk)
    m_rc_iid = eval_by_condition("iid", f"CRISP_DEawareRC_alpha{args.alpha}", train["true"], x_cal_iid, train["ctrl"], train["conds"], train["meta"], train_de_mask, topk=args.topk)
    m_rc_ood = eval_by_condition("ood", f"CRISP_DEawareRC_alpha{args.alpha}", test["true"], x_cal_ood, test["ctrl"], test["conds"], test["meta"], test_de_mask, topk=args.topk)

    metrics = pd.concat([m_base_iid, m_base_ood, m_rc_iid, m_rc_ood], ignore_index=True)
    metrics.to_csv(out_dir / "metrics_by_condition.csv", index=False)

    summary = metrics.groupby(["split", "model"]).mean(numeric_only=True).reset_index()
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "genes": genes.tolist(),
        "drugs": drugs,
        "best_loss": best_loss,
    }, out_dir / "crisp_deaware_rc_model.pt")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Summary:")
    print(summary.to_string(index=False))
    print("=" * 80)
    print("Saved to:", out_dir.resolve())
    print("重点看 ood / CRISP_DEawareRC_alpha... 的 pearson_de、r2score_de、pearson_delta_de。")


if __name__ == "__main__":
    main()
