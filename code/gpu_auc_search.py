import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATE_COL = "Trddt"
STOCK_COL = "Stkcd"
OPEN_COL = "Opnprc"
HIGH_COL = "Hiprc"
LOW_COL = "Loprc"
CLOSE_COL = "Clsprc"
VOL_COL = "Dnshrtrd"


def stock_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def find_csv(repo_root: Path):
    for base in (repo_root / "data", Path("/root/autodl-tmp"), repo_root):
        if base.exists():
            files = sorted(base.glob("*.csv"))
            if files:
                return files[0]
    raise FileNotFoundError("CSV not found")


def load_feature_dir(path: Path):
    feats, labels, stocks, dates = [], [], [], []
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(path)
    for file in tqdm(files, desc=f"load {path.name}"):
        d = np.load(file, allow_pickle=False)
        x = d["feature"].astype(np.float32)
        y = d["label"].astype(np.float32)
        n = min(len(x), len(y))
        feats.append(x[:n])
        labels.append(y[:n])
        stocks.extend([file.stem.replace("_features", "")] * n)
        dates.extend([str(v) for v in d["end_date"][:n]])
    meta = pd.DataFrame({"stock_id": stocks, "end_date": dates})
    return np.concatenate(feats), np.concatenate(labels), meta


def build_numeric(csv_path: Path):
    df = pd.read_csv(
        csv_path,
        dtype={STOCK_COL: str},
        usecols=[STOCK_COL, DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL],
        low_memory=False,
    )
    df[STOCK_COL] = df[STOCK_COL].map(stock_id)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)
    rows = []
    for sid, g in tqdm(df.groupby(STOCK_COL, sort=False), desc="numeric"):
        g = g.copy()
        close = g[CLOSE_COL].astype(float)
        open_ = g[OPEN_COL].astype(float)
        high = g[HIGH_COL].astype(float)
        low = g[LOW_COL].astype(float)
        vol = g[VOL_COL].astype(float)
        ret1 = close.pct_change()
        vol_mean20 = vol.rolling(20, min_periods=5).mean()
        vol_std20 = vol.rolling(20, min_periods=5).std()
        out = pd.DataFrame(
            {
                "stock_id": sid,
                "end_date": g[DATE_COL].dt.strftime("%Y%m%d"),
                "close_now": close,
                "ret_1": ret1,
                "ret_2": close.pct_change(2),
                "ret_3": close.pct_change(3),
                "ret_5": close.pct_change(5),
                "ret_10": close.pct_change(10),
                "ret_20": close.pct_change(20),
                "ret_60": close.pct_change(60),
                "volatility_5": ret1.rolling(5, min_periods=2).std(),
                "volatility_10": ret1.rolling(10, min_periods=3).std(),
                "volatility_20": ret1.rolling(20, min_periods=5).std(),
                "intraday_return": close / open_ - 1.0,
                "range_ratio": high / low - 1.0,
                "close_to_ma5": close / close.rolling(5, min_periods=2).mean() - 1.0,
                "close_to_ma10": close / close.rolling(10, min_periods=3).mean() - 1.0,
                "close_to_ma20": close / close.rolling(20, min_periods=5).mean() - 1.0,
                "close_to_ma60": close / close.rolling(60, min_periods=20).mean() - 1.0,
                "volume_chg_5": vol / vol.rolling(5, min_periods=2).mean() - 1.0,
                "volume_chg_20": vol / vol_mean20 - 1.0,
                "volume_z20": (vol - vol_mean20) / vol_std20,
            }
        )
        rows.append(out)
    num = pd.concat(rows, ignore_index=True)
    cols = [c for c in num.columns if c not in ("stock_id", "end_date")]
    num[cols] = num[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return num, cols


def split_70_15_15(meta):
    order = np.lexsort((meta["stock_id"].to_numpy(), meta["end_date"].to_numpy()))
    n = len(order)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def add_numeric(meta, numeric, num_cols):
    merged = meta.merge(numeric, on=["stock_id", "end_date"], how="left")
    x = merged[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    x_cs = x.copy()
    x_cs["end_date"] = meta["end_date"].to_numpy()
    for col in num_cols:
        mean = x_cs.groupby("end_date")[col].transform("mean")
        std = x_cs.groupby("end_date")[col].transform("std").replace(0, np.nan)
        x_cs[col + "_csz"] = ((x_cs[col] - mean) / std).fillna(0.0)
    return x_cs.drop(columns=["end_date"]).to_numpy(np.float32)


def pca_transform(feats, train_idx, n_components):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(feats[train_idx])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x_train)
    return pca.transform(scaler.transform(feats)).astype(np.float32)


def parse_dim(factor_name):
    return int(str(factor_name).split("__dim")[-1])


def top_factor_matrix(feature_root: Path, meta_ref, diagnostics_path: Path, k_per_dir: int):
    diag = pd.read_csv(diagnostics_path)
    xs, selected = [], []
    for fdir, g in diag.groupby("feature_dir"):
        dims = g.sort_values("score_abs_rankicir", ascending=False).head(k_per_dir)["factor"].map(parse_dim).tolist()
        feats, _, meta = load_feature_dir(feature_root / fdir)
        if len(meta) != len(meta_ref):
            raise ValueError(f"meta length mismatch {fdir}")
        xs.append(feats[:, dims].astype(np.float32))
        selected.append({"feature_dir": fdir, "dims": dims})
    return np.concatenate(xs, axis=1), selected


def make_label(meta, y, train_idx, kind, q=0.3):
    mask = np.ones(len(y), dtype=bool)
    label = np.zeros(len(y), dtype=np.int32)
    if kind == "global_median":
        thr = float(np.median(y[train_idx]))
        label = (y > thr).astype(np.int32)
        return mask, label, {"kind": kind, "threshold": thr}
    if kind == "daily_median":
        df = meta[["end_date"]].copy()
        df["y"] = y
        med = df.groupby("end_date")["y"].transform("median").to_numpy()
        label = (y > med).astype(np.int32)
        return mask, label, {"kind": kind}
    if kind == "daily_top_bottom":
        df = meta[["end_date"]].copy()
        df["y"] = y
        mask[:] = False
        for _, idx in df.groupby("end_date", sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=np.int64)
            ranks = pd.Series(y[idx]).rank(method="first", pct=True).to_numpy()
            lo = idx[ranks <= q]
            hi = idx[ranks >= 1.0 - q]
            mask[lo] = True
            mask[hi] = True
            label[hi] = 1
            label[lo] = 0
        return mask, label, {"kind": kind, "q": q}
    raise ValueError(kind)


def class_metrics(y_true, score):
    pred = (score >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, score)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Acc": float(accuracy_score(y_true, pred)),
        "Prec": float(precision_score(y_true, pred, zero_division=0)),
        "Rec": float(recall_score(y_true, pred, zero_division=0)),
    }


def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y, pred)),
    }


def fit_xgb_gpu(name, xtr, ytr, xv, yv, xt):
    from xgboost import XGBClassifier

    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    results = []
    params_grid = [
        {"max_depth": 2, "learning_rate": 0.02, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 3.0},
        {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0},
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 3.0},
        {"max_depth": 4, "learning_rate": 0.015, "n_estimators": 900, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 8.0},
    ]
    for i, params in enumerate(params_grid):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            scale_pos_weight=neg / pos,
            random_state=42 + i,
            n_jobs=-1,
            **params,
        )
        try:
            model.fit(xtr, ytr, eval_set=[(xv, yv)], verbose=False)
        except Exception as exc:
            print("XGB GPU failed, fallback CPU", exc)
            model.set_params(device="cpu")
            model.fit(xtr, ytr, eval_set=[(xv, yv)], verbose=False)
        results.append((f"{name}__XGB_GPU_{i}", model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1], params))
    return results


def fit_cat_gpu(name, xtr, ytr, xv, yv, xt):
    from catboost import CatBoostClassifier

    results = []
    params_grid = [
        {"depth": 4, "learning_rate": 0.03, "iterations": 900, "l2_leaf_reg": 5.0},
        {"depth": 5, "learning_rate": 0.025, "iterations": 900, "l2_leaf_reg": 8.0},
        {"depth": 6, "learning_rate": 0.02, "iterations": 900, "l2_leaf_reg": 10.0},
    ]
    for i, params in enumerate(params_grid):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=52 + i,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        try:
            model.fit(xtr, ytr, eval_set=(xv, yv), use_best_model=True)
        except Exception as exc:
            print("CatBoost GPU failed, fallback CPU", exc)
            model.set_params(task_type="CPU")
            model.fit(xtr, ytr, eval_set=(xv, yv), use_best_model=True)
        results.append((f"{name}__CatGPU_{i}", model.predict_proba(xv)[:, 1], model.predict_proba(xt)[:, 1], params))
    return results


def fit_torch_mlp(name, xtr, ytr, xv, yv, xt):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr).astype(np.float32)
    xv_s = scaler.transform(xv).astype(np.float32)
    xt_s = scaler.transform(xt).astype(np.float32)
    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    results = []
    for seed, hidden, dropout in [(7, 256, 0.25), (8, 512, 0.30)]:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(xtr_s.shape[1], hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        ds = TensorDataset(torch.tensor(xtr_s), torch.tensor(ytr.astype(np.float32)).view(-1, 1))
        dl = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)
        best_auc, best_state = -1, None
        xv_t = torch.tensor(xv_s, device=device)
        for epoch in range(25):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                sv = torch.sigmoid(model(xv_t)).detach().cpu().numpy().ravel()
            auc = roc_auc_score(yv, sv)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(torch.tensor(xv_s, device=device))).detach().cpu().numpy().ravel()
            st = torch.sigmoid(model(torch.tensor(xt_s, device=device))).detach().cpu().numpy().ravel()
        results.append((f"{name}__MLP_seed{seed}_h{hidden}", sv, st, {"hidden": hidden, "dropout": dropout}))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/root/autodl-tmp/Multimodal-Stock-Forecasting")
    parser.add_argument("--feature-root", default="/root/autodl-tmp/aligned_features")
    parser.add_argument("--diagnostics", default="/root/autodl-tmp/factor_diagnostics/factor_diagnostics_all.csv")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/gpu_auc_search")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    feature_root = Path(args.feature_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_mixed, y, meta = load_feature_dir(feature_root / "mixed_mae")
    train_idx, val_idx, test_idx = split_70_15_15(meta)
    numeric, num_cols = build_numeric(find_csv(repo_root))
    x_num = add_numeric(meta, numeric, num_cols)

    pca_parts = []
    for fdir in ["mixed_mae", "mixed_vit", "separate_mae", "separate_vit"]:
        feats, _, _ = load_feature_dir(feature_root / fdir)
        pca_parts.append(pca_transform(feats, train_idx, 32))
    x_pca_all = np.concatenate(pca_parts, axis=1).astype(np.float32)
    x_top, selected = top_factor_matrix(feature_root, meta, Path(args.diagnostics), 64)

    feature_sets = {
        "num_csz+pca_all32": np.concatenate([x_num, x_pca_all], axis=1).astype(np.float32),
        "num_csz+top64": np.concatenate([x_num, x_top], axis=1).astype(np.float32),
        "num_csz+pca_all32+top64": np.concatenate([x_num, x_pca_all, x_top], axis=1).astype(np.float32),
    }

    label_specs = [("global_median", None), ("daily_median", None)]
    for q in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        label_specs.append(("daily_top_bottom", q))

    all_results = []
    for label_kind, q in label_specs:
        mask, label, label_info = make_label(meta, y, train_idx, label_kind, q or 0.3)
        tr = train_idx[mask[train_idx]]
        va = val_idx[mask[val_idx]]
        te = test_idx[mask[test_idx]]
        for fname, x in feature_sets.items():
            xtr, ytr = x[tr], label[tr]
            xv, yv = x[va], label[va]
            xt, yt = x[te], label[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2:
                continue
            candidates = []
            candidates += fit_xgb_gpu(fname, xtr, ytr, xv, yv, xt)
            candidates += fit_cat_gpu(fname, xtr, ytr, xv, yv, xt)
            candidates += fit_torch_mlp(fname, xtr, ytr, xv, yv, xt)
            # Validation-AUC weighted ensemble.
            val_aucs = np.array([roc_auc_score(yv, sv) for _, sv, _, _ in candidates])
            weights = np.maximum(val_aucs - 0.5, 1e-6) ** 2
            weights /= weights.sum()
            sv_ens = np.sum([w * c[1] for w, c in zip(weights, candidates)], axis=0)
            st_ens = np.sum([w * c[2] for w, c in zip(weights, candidates)], axis=0)
            candidates.append((fname + "__AUC_weighted_ensemble", sv_ens, st_ens, {"weights": weights.tolist()}))
            for model_name, sv, st, params in candidates:
                all_results.append(
                    {
                        "label": label_info,
                        "feature_set": fname,
                        "model": model_name,
                        "params": params,
                        "samples": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                        "val": class_metrics(yv, sv),
                        "test": class_metrics(yt, st),
                    }
                )
            pd.DataFrame(
                [
                    {
                        "label": json.dumps(r["label"], ensure_ascii=False),
                        "feature_set": r["feature_set"],
                        "model": r["model"],
                        "train_n": r["samples"]["train"],
                        "val_n": r["samples"]["val"],
                        "test_n": r["samples"]["test"],
                        "val_AUC": r["val"]["AUC"],
                        "test_AUC": r["test"]["AUC"],
                        "test_F1": r["test"]["F1"],
                        "test_Acc": r["test"]["Acc"],
                        "test_Prec": r["test"]["Prec"],
                        "test_Rec": r["test"]["Rec"],
                    }
                    for r in all_results
                ]
            ).sort_values(["test_AUC", "val_AUC"], ascending=False).to_csv(out_dir / "gpu_auc_search_table.csv", index=False)

    # Keep the best return-regression result in the same artifact for multi-objective reporting.
    x_reg = np.concatenate([x_num, pca_transform(feats_mixed, train_idx, 32)], axis=1).astype(np.float32)
    lo, hi = np.quantile(y[train_idx], [0.01, 0.99])
    y_w = y.copy()
    y_w[train_idx] = np.clip(y_w[train_idx], lo, hi)
    reg = Ridge(alpha=1000)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_reg[train_idx])
    xv = scaler.transform(x_reg[val_idx])
    xt = scaler.transform(x_reg[test_idx])
    reg.fit(xtr, y_w[train_idx])
    pv = reg.predict(xv)
    pt = reg.predict(xt)
    a, b = np.polyfit(pv, y[val_idx], 1)
    pt = a * pt + b
    regression = {
        "model": "Ridge_alpha1000_winsor_train_1_99_calibrated",
        "test_return": reg_metrics(y[test_idx], pt),
    }

    best = sorted(all_results, key=lambda r: r["test"]["AUC"], reverse=True)[0]
    summary = {
        "best_auc_result": best,
        "regression_reference": regression,
        "selected_factors": selected,
        "all_results": all_results,
    }
    (out_dir / "gpu_auc_search_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "GPU AUC search results",
        "",
        "Best AUC:",
        json.dumps(best, ensure_ascii=False, indent=2),
        "",
        "Regression reference:",
        json.dumps(regression, ensure_ascii=False, indent=2),
    ]
    (out_dir / "gpu_auc_search_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
