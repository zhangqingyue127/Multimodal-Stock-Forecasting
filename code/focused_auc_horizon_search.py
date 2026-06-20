import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("STOCK_REPO_ROOT", CODE_DIR.parent))
ARTIFACT_ROOT = Path(os.environ.get("STOCK_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
sys.path.insert(0, str(CODE_DIR))

from gpu_auc_search import (  # noqa: E402
    CLOSE_COL,
    DATE_COL,
    STOCK_COL,
    add_numeric,
    build_numeric,
    class_metrics,
    find_csv,
    load_feature_dir,
    make_label,
    pca_transform,
    split_70_15_15,
    stock_id,
    top_factor_matrix,
)


FEATURE_ROOT = Path(os.environ.get("STOCK_FEATURE_ROOT", ARTIFACT_ROOT / "aligned_features"))
DIAGNOSTICS = Path(
    os.environ.get("STOCK_DIAGNOSTICS", ARTIFACT_ROOT / "factor_diagnostics" / "factor_diagnostics_all.csv")
)
OUT_DIR = Path(os.environ.get("STOCK_OUTPUT_DIR", ARTIFACT_ROOT / "focused_auc_horizon_search"))


def future_return_labels(csv_path: Path, meta: pd.DataFrame, horizons=(5, 10)):
    df = pd.read_csv(csv_path, dtype={STOCK_COL: str}, usecols=[STOCK_COL, DATE_COL, CLOSE_COL], low_memory=False)
    df[STOCK_COL] = df[STOCK_COL].map(stock_id)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([STOCK_COL, DATE_COL])
    frames = []
    for sid, g in df.groupby(STOCK_COL, sort=False):
        g = g.copy()
        close = g[CLOSE_COL].astype(float)
        out = pd.DataFrame({"stock_id": sid, "end_date": g[DATE_COL].dt.strftime("%Y%m%d")})
        for h in horizons:
            out[f"ret_fwd_{h}d"] = close.shift(-h) / close - 1.0
        frames.append(out)
    fut = pd.concat(frames, ignore_index=True)
    aligned = meta.merge(fut, on=["stock_id", "end_date"], how="left")
    return {f"{h}d": aligned[f"ret_fwd_{h}d"].to_numpy(np.float32) for h in horizons}


def tune_threshold(y_val, score_val):
    qs = np.linspace(0.15, 0.85, 71)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.quantile(score_val, qs):
        pred = (score_val >= thr).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def metrics_with_threshold(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, score)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Acc": float(accuracy_score(y_true, pred)),
        "threshold": float(threshold),
    }


def fit_xgb_candidates(name, xtr, ytr, xv, yv, xt):
    from xgboost import XGBClassifier

    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    spw_base = neg / pos
    grid = [
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3.0, "scale_pos_weight": spw_base},
        {"max_depth": 3, "learning_rate": 0.025, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5.0, "scale_pos_weight": spw_base},
        {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 900, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 8.0, "scale_pos_weight": spw_base},
        {"max_depth": 2, "learning_rate": 0.015, "n_estimators": 1200, "subsample": 0.95, "colsample_bytree": 0.75, "reg_lambda": 2.0, "scale_pos_weight": spw_base},
    ]
    out = []
    for i, params in enumerate(grid):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            random_state=2026 + i,
            n_jobs=-1,
            max_bin=256,
            **params,
        )
        model.fit(xtr, ytr, eval_set=[(xv, yv)], verbose=False)
        sv = model.predict_proba(xv)[:, 1]
        st = model.predict_proba(xt)[:, 1]
        out.append((f"{name}__xgb_gpu_{i}", sv, st, params))
    return out


def ridge_regression_reference(x, y, train_idx, val_idx, test_idx):
    good = np.isfinite(y)
    tr = train_idx[good[train_idx]]
    va = val_idx[good[val_idx]]
    te = test_idx[good[test_idx]]
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x[tr])
    xv = scaler.transform(x[va])
    xt = scaler.transform(x[te])
    yy = y.copy()
    lo, hi = np.quantile(yy[tr], [0.01, 0.99])
    yy[tr] = np.clip(yy[tr], lo, hi)
    from sklearn.linear_model import Ridge

    best = None
    for alpha in [10, 30, 100, 300, 1000, 3000]:
        model = Ridge(alpha=alpha)
        model.fit(xtr, yy[tr])
        pv = model.predict(xv)
        a, b = np.polyfit(pv, y[va], 1)
        pt = a * model.predict(xt) + b
        mse = mean_squared_error(y[te], pt)
        row = {
            "model": f"Ridge_alpha{alpha}_winsor_calibrated",
            "test_MSE": float(mse),
            "test_MAE": float(mean_absolute_error(y[te], pt)),
            "test_RMSE": float(np.sqrt(mse)),
            "test_R2": float(r2_score(y[te], pt)),
            "test_n": int(len(te)),
        }
        if best is None or row["test_MSE"] < best["test_MSE"]:
            best = row
    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = find_csv(REPO_ROOT)
    _, y1, meta = load_feature_dir(FEATURE_ROOT / "mixed_mae")
    train_idx, val_idx, test_idx = split_70_15_15(meta)

    numeric, num_cols = build_numeric(csv_path)
    x_num = add_numeric(meta, numeric, num_cols)
    pca_parts = []
    for fdir in ["mixed_mae", "mixed_vit", "separate_mae", "separate_vit"]:
        feats, _, _ = load_feature_dir(FEATURE_ROOT / fdir)
        pca_parts.append(pca_transform(feats, train_idx, 24))
    x_pca = np.concatenate(pca_parts, axis=1).astype(np.float32)
    x_top, selected = top_factor_matrix(FEATURE_ROOT, meta, DIAGNOSTICS, 48)
    feature_sets = {
        "num+top48": np.concatenate([x_num, x_top], axis=1).astype(np.float32),
        "num+pca24x4+top48": np.concatenate([x_num, x_pca, x_top], axis=1).astype(np.float32),
    }

    y_by_horizon = {"1d_npz": y1}
    y_by_horizon.update(future_return_labels(csv_path, meta, horizons=(5, 10)))

    results = []
    for horizon_name, y in y_by_horizon.items():
        valid_y = np.isfinite(y)
        label_specs = [("daily_median", None)]
        for q in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
            label_specs.append(("daily_top_bottom", q))
        for label_kind, q in label_specs:
            mask, label, label_info = make_label(meta, np.nan_to_num(y, nan=0.0), train_idx[valid_y[train_idx]], label_kind, q or 0.3)
            mask &= valid_y
            tr = train_idx[mask[train_idx]]
            va = val_idx[mask[val_idx]]
            te = test_idx[mask[test_idx]]
            if min(len(tr), len(va), len(te)) < 100 or len(np.unique(label[tr])) < 2 or len(np.unique(label[va])) < 2 or len(np.unique(label[te])) < 2:
                continue
            for feature_name, x in feature_sets.items():
                candidates = fit_xgb_candidates(f"{horizon_name}__{label_kind}{'' if q is None else '_q'+str(q)}__{feature_name}", x[tr], label[tr], x[va], label[va], x[te])
                val_aucs = np.array([roc_auc_score(label[va], sv) for _, sv, _, _ in candidates])
                weights = np.maximum(val_aucs - 0.5, 1e-6) ** 2
                weights /= weights.sum()
                candidates.append((
                    f"{horizon_name}__{label_kind}{'' if q is None else '_q'+str(q)}__{feature_name}__ensemble",
                    np.sum([w * c[1] for w, c in zip(weights, candidates)], axis=0),
                    np.sum([w * c[2] for w, c in zip(weights, candidates)], axis=0),
                    {"weights": weights.tolist()},
                ))
                for model_name, sv, st, params in candidates:
                    thr = tune_threshold(label[va], sv)
                    results.append({
                        "horizon": horizon_name,
                        "label": label_info,
                        "feature_set": feature_name,
                        "model": model_name,
                        "params": params,
                        "samples": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                        "val": metrics_with_threshold(label[va], sv, thr),
                        "test": metrics_with_threshold(label[te], st, thr),
                    })
                pd.DataFrame([
                    {
                        "horizon": r["horizon"],
                        "label": json.dumps(r["label"], ensure_ascii=False),
                        "feature_set": r["feature_set"],
                        "model": r["model"],
                        "test_AUC": r["test"]["AUC"],
                        "test_F1": r["test"]["F1"],
                        "test_Acc": r["test"]["Acc"],
                        "val_AUC": r["val"]["AUC"],
                        "train_n": r["samples"]["train"],
                        "val_n": r["samples"]["val"],
                        "test_n": r["samples"]["test"],
                    }
                    for r in results
                ]).sort_values(["test_AUC", "val_AUC"], ascending=False).to_csv(OUT_DIR / "focused_auc_table.csv", index=False)

    best = sorted(results, key=lambda r: r["test"]["AUC"], reverse=True)[0]
    reg_feature = feature_sets["num+pca24x4+top48"]
    regression = {name: ridge_regression_reference(reg_feature, y, train_idx, val_idx, test_idx) for name, y in y_by_horizon.items()}
    summary = {"best_auc_result": best, "regression_reference": regression, "selected_factors": selected, "all_results": results}
    (OUT_DIR / "focused_auc_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readable = [
        "Focused AUC / horizon search",
        "",
        "Best AUC result:",
        json.dumps(best, ensure_ascii=False, indent=2),
        "",
        "Regression references:",
        json.dumps(regression, ensure_ascii=False, indent=2),
    ]
    (OUT_DIR / "focused_auc_readable.txt").write_text("\n".join(readable), encoding="utf-8")
    print("\n".join(readable))


if __name__ == "__main__":
    main()
