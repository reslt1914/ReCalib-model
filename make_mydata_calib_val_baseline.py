import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


MATRIX_FILES = [
    "x_true_condition_mean.csv",
    "x_crisp_condition_mean.csv",
    "x_ctrl_condition_mean.csv",
    "residual_condition_mean.csv",
    "delta_true_condition_mean.csv",
    "delta_crisp_condition_mean.csv",
]


def subset_matrix(src, dst, keep_conditions):
    df = pd.read_csv(src, index_col=0, engine="python")
    df = df.loc[keep_conditions]
    df.to_csv(dst)


def subset_table(src, dst, keep_conditions):
    df = pd.read_csv(src)
    df["condition"] = df["condition"].astype(str)
    df = df[df["condition"].isin(set(keep_conditions))].copy()
    df.to_csv(dst, index=False)

    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.mean().to_frame("mean").T
    return df, summary


def write_split(src_split_dir, dst_split_dir, keep_conditions, split_name):
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    for f in MATRIX_FILES:
        subset_matrix(src_split_dir / f, dst_split_dir / f, keep_conditions)

    subset_table(src_split_dir / "condition_metadata.csv", dst_split_dir / "condition_metadata.csv", keep_conditions)
    metrics, summary = subset_table(src_split_dir / "metrics_by_condition.csv", dst_split_dir / "metrics_by_condition.csv", keep_conditions)

    summary.insert(0, "split", split_name)
    summary.insert(1, "n_conditions", len(keep_conditions))
    summary.to_csv(dst_split_dir / "metrics_summary.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default="crisp_outputs/mydata_baseline_full")
    parser.add_argument("--out_dir", default="crisp_outputs/mydata_baseline_calib_seed2024")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    baseline = Path(args.baseline_dir)
    out = Path(args.out_dir)

    if out.exists():
        print("remove old:", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(baseline / "gene_names.csv", out / "gene_names.csv")

    meta = pd.read_csv(baseline / "iid" / "condition_metadata.csv")
    meta["condition"] = meta["condition"].astype(str)

    rng = np.random.default_rng(args.seed)

    train_conds = []
    val_conds = []

    # 按 cell_type 分层切，避免某个 cell_type 全进 train 或 val
    for cell_type, sub in meta.groupby("cell_type"):
        conds = sub["condition"].astype(str).to_numpy()
        rng.shuffle(conds)
        n_val = max(1, int(round(len(conds) * args.val_frac)))
        val_conds.extend(conds[:n_val].tolist())
        train_conds.extend(conds[n_val:].tolist())

    train_conds = sorted(train_conds)
    val_conds = sorted(val_conds)

    print("iid total:", len(meta))
    print("calib train:", len(train_conds))
    print("calib val:", len(val_conds))

    pd.DataFrame({"condition": train_conds}).to_csv(out / "calib_train_conditions.csv", index=False)
    pd.DataFrame({"condition": val_conds}).to_csv(out / "calib_val_conditions.csv", index=False)

    m_train = write_split(baseline / "iid", out / "iid", train_conds, "iid")
    m_val = write_split(baseline / "iid", out / "ood", val_conds, "ood")

    all_metrics = pd.concat([m_train, m_val], ignore_index=True)
    all_metrics.to_csv(out / "metrics_all_splits.csv", index=False)

    summary = all_metrics.groupby("split").mean(numeric_only=True).reset_index()
    summary.to_csv(out / "metrics_summary_all_splits.csv", index=False)

    print("saved:", out)


if __name__ == "__main__":
    main()
