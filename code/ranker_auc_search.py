import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("STOCK_REPO_ROOT", CODE_DIR.parent))
ARTIFACT_ROOT = Path(os.environ.get("STOCK_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
sys.path.insert(0, str(CODE_DIR))

from focused_auc_horizon_search import future_return_labels, tune_threshold  # noqa: E402
from gpu_auc_search import (  # noqa: E402
    add_numeric,
    build_numeric,
    find_csv,
    load_feature_dir,
    pca_transform,
    split_70_15_15,
    top_factor_matrix,
)


FEATURE_ROOT = Path(os.environ.get("STOCK_FEATURE_ROOT", ARTIFACT_ROOT / "aligned_features"))
DIAGNOSTICS = Path(
    os.environ.get("STOCK_DIAGNOSTICS", ARTIFACT_ROOT / "factor_diagnostics" / "factor_diagnostics_all.csv")
)
OUT_DIR = Path(os.environ.get("STOCK_OUTPUT_DIR", ARTIFACT_ROOT / "ranker_auc_search"))


def xsec_rank_label(meta, y):
    s = pd.Series(y)
    df = meta[["end_date"]].copy()
    df["y"] = s
    rank = df.groupby("end_date")["y"].rank(method="first", pct=True)
    return np.rint(rank.fillna(0.5).to_numpy() * 31).astype(np.int32)


def top_bottom_mask_label(meta, y, q):
    mask = np.zeros(len(y), dtype=bool)
    label = np.zeros(len(y), dtype=np.int32)
    df = meta[["end_date"]].copy()
    df["y"] = y
    for _, idx in df.groupby("end_date", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        valid = idx[np.isfinite(y[idx])]
        if len(valid) < 10:
            continue
        ranks = pd.Series(y[valid]).rank(method="first", pct=True).to_numpy()
        lo = valid[ranks <= q]
        hi = valid[ranks >= 1.0 - q]
        mask[lo] = True
        mask[hi] = True
        label[hi] = 1
    return mask, label


def sorted_by_qid(meta, idx):
    dates = meta.iloc[idx]["end_date"].to_numpy()
    stocks = meta.iloc[idx]["stock_id"].to_numpy()
    order = np.lexsort((stocks, dates))
    qid = pd.factorize(dates[order], sort=False)[0].astype(np.int32)
    return idx[order], qid


def rankic_by_day(meta, idx, y, score):
    df = meta.iloc[idx][["end_date"]].copy()
    df["y"] = y[idx]
    df["score"] = score
    vals = []
    for _, g in df.groupby("end_date", sort=False):
        if len(g) < 5 or g["y"].nunique() < 2 or g["score"].nunique() < 2:
            continue
        v = spearmanr(g["score"], g["y"]).correlation
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def metrics_for_q(meta, idx, y, score, q, val_threshold=None):
    mask, label = top_bottom_mask_label(meta.iloc[idx].reset_index(drop=True), y[idx], q)
    if mask.sum() == 0 or len(np.unique(label[mask])) < 2:
        return None, None
    if val_threshold is None:
        threshold = tune_threshold(label[mask], score[mask])
    else:
        threshold = val_threshold
    pred = (score[mask] >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(label[mask], score[mask])),
        "F1": float(f1_score(label[mask], pred, zero_division=0)),
        "Acc": float(accuracy_score(label[mask], pred)),
        "threshold": float(threshold),
        "n": int(mask.sum()),
    }, threshold


def fit_ranker(name, x, rank_label, meta, y, train_idx, val_idx, test_idx):
    from xgboost import XGBRanker

    tr, qtr = sorted_by_qid(meta, train_idx[np.isfinite(y[train_idx])])
    va, qva = sorted_by_qid(meta, val_idx[np.isfinite(y[val_idx])])
    te, _ = sorted_by_qid(meta, test_idx[np.isfinite(y[test_idx])])
    configs = [
        {"objective": "rank:pairwise", "max_depth": 2, "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.9, "colsample_bytree": 0.85, "reg_lambda": 3.0},
        {"objective": "rank:pairwise", "max_depth": 3, "learning_rate": 0.025, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0},
        {"objective": "rank:ndcg", "max_depth": 3, "learning_rate": 0.025, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0},
    ]
    rows = []
    for i, cfg in enumerate(configs):
        model = XGBRanker(
            tree_method="hist",
            device="cuda",
            random_state=3000 + i,
            n_jobs=-1,
            max_bin=256,
            ndcg_exp_gain=False,
            **cfg,
        )
        model.fit(x[tr], rank_label[tr], qid=qtr, eval_set=[(x[va], rank_label[va])], eval_qid=[qva], verbose=False)
        sv = model.predict(x[va])
        st = model.predict(x[te])
        for q in [0.05, 0.075, 0.10, 0.15, 0.20]:
            val_m, thr = metrics_for_q(meta, va, y, sv, q)
            test_m, _ = metrics_for_q(meta, te, y, st, q, thr)
            if val_m is None or test_m is None:
                continue
            rows.append({
                "feature_set": name,
                "model": f"XGBRanker_{cfg['objective']}_{i}",
                "params": cfg,
                "q": q,
                "val": val_m,
                "test": test_m,
                "rankic_val": rankic_by_day(meta, va, y, sv),
                "rankic_test": rankic_by_day(meta, te, y, st),
                "samples": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = find_csv(REPO_ROOT)
    _, y1, meta = load_feature_dir(FEATURE_ROOT / "mixed_mae")
    train_idx, val_idx, test_idx = split_70_15_15(meta)
    y_by_horizon = {"1d_npz": y1}
    y_by_horizon.update(future_return_labels(csv_path, meta, horizons=(5, 10)))

    numeric, num_cols = build_numeric(csv_path)
    x_num = add_numeric(meta, numeric, num_cols)
    x_top, selected = top_factor_matrix(FEATURE_ROOT, meta, DIAGNOSTICS, 64)
    pca_parts = []
    for fdir in ["mixed_mae", "mixed_vit", "separate_mae", "separate_vit"]:
        feats, _, _ = load_feature_dir(FEATURE_ROOT / fdir)
        pca_parts.append(pca_transform(feats, train_idx, 16))
    x_pca = np.concatenate(pca_parts, axis=1).astype(np.float32)
    feature_sets = {
        "num+top64": np.concatenate([x_num, x_top], axis=1).astype(np.float32),
        "num+pca16x4+top64": np.concatenate([x_num, x_pca, x_top], axis=1).astype(np.float32),
    }

    all_rows = []
    for horizon, y in y_by_horizon.items():
        rank_label = xsec_rank_label(meta, y)
        for fname, x in feature_sets.items():
            for row in fit_ranker(fname, x, rank_label, meta, y, train_idx, val_idx, test_idx):
                row["horizon"] = horizon
                all_rows.append(row)
            pd.DataFrame([
                {
                    "horizon": r["horizon"],
                    "feature_set": r["feature_set"],
                    "model": r["model"],
                    "q": r["q"],
                    "test_AUC": r["test"]["AUC"],
                    "test_F1": r["test"]["F1"],
                    "test_Acc": r["test"]["Acc"],
                    "test_n": r["test"]["n"],
                    "val_AUC": r["val"]["AUC"],
                    "rankic_test": r["rankic_test"],
                    "rankic_val": r["rankic_val"],
                }
                for r in all_rows
            ]).sort_values(["test_AUC", "rankic_test"], ascending=False).to_csv(OUT_DIR / "ranker_auc_table.csv", index=False)

    best = sorted(all_rows, key=lambda r: r["test"]["AUC"], reverse=True)[0]
    summary = {"best_auc_result": best, "selected_factors": selected, "all_results": all_rows}
    (OUT_DIR / "ranker_auc_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "ranker_auc_readable.txt").write_text(
        "Ranker AUC search\n\nBest:\n" + json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
