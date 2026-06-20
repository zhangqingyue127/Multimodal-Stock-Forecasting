import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DATE_COL = "Trddt"
STOCK_COL = "Stkcd"
OPEN_COL = "Opnprc"
HIGH_COL = "Hiprc"
LOW_COL = "Loprc"
CLOSE_COL = "Clsprc"
VOL_COL = "Dnshrtrd"

BASELINE = {
    "model": "RidgeFusion+calibrated",
    "return": {"MSE": 0.003080710070207715, "MAE": 0.03736747056245804, "R2": 0.0006667971611022949},
    "classification": {"AUC": 0.5235833688562794, "F1": 0.4729583993024483},
    "ranking": {"RankIC": -0.011592435650527477, "TopK_Win": 0.49978867173194885, "Spread": 0.0019309308845549822},
}


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
        out = pd.DataFrame(
            {
                "stock_id": sid,
                "end_date": g[DATE_COL].dt.strftime("%Y%m%d"),
                "close_now": close,
                "ret_1": ret1,
                "ret_2": close.pct_change(2),
                "ret_5": close.pct_change(5),
                "ret_10": close.pct_change(10),
                "ret_20": close.pct_change(20),
                "volatility_5": ret1.rolling(5, min_periods=2).std(),
                "volatility_20": ret1.rolling(20, min_periods=5).std(),
                "intraday_return": close / open_ - 1.0,
                "range_ratio": high / low - 1.0,
                "close_to_ma5": close / close.rolling(5, min_periods=2).mean() - 1.0,
                "close_to_ma10": close / close.rolling(10, min_periods=3).mean() - 1.0,
                "close_to_ma20": close / close.rolling(20, min_periods=5).mean() - 1.0,
                "volume_chg_5": vol / vol.rolling(5, min_periods=2).mean() - 1.0,
                "volume_chg_20": vol / vol.rolling(20, min_periods=5).mean() - 1.0,
                "volume_z20": (vol - vol.rolling(20, min_periods=5).mean()) / vol.rolling(20, min_periods=5).std(),
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


def make_groups(meta_subset):
    sizes = meta_subset.groupby("end_date", sort=False).size().astype(int).tolist()
    return sizes


def sort_by_date(meta, idx):
    sub = meta.iloc[idx].reset_index(drop=False).rename(columns={"index": "orig_idx"})
    order = np.lexsort((sub["stock_id"].to_numpy(), sub["end_date"].to_numpy()))
    sorted_orig = sub.iloc[order]["orig_idx"].to_numpy()
    sorted_meta = meta.iloc[sorted_orig].reset_index(drop=True)
    return sorted_orig, sorted_meta


def cross_sectional_rank_labels(meta, y, bins=10):
    df = meta[["end_date"]].copy()
    df["y"] = y
    labels = np.zeros(len(df), dtype=np.int32)
    for _, pos in df.groupby("end_date", sort=False).groups.items():
        idx = np.asarray(list(pos), dtype=np.int64)
        ranks = pd.Series(df["y"].to_numpy()[idx]).rank(method="first", pct=True).to_numpy()
        labels[idx] = np.clip(np.floor(ranks * bins), 0, bins - 1).astype(np.int32)
    return labels


def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y, pred)),
    }


def class_metrics(y_true, score, threshold):
    label = (y_true > threshold).astype(int)
    pred = (score > threshold).astype(int)
    try:
        auc = roc_auc_score(label, score)
    except Exception:
        auc = float("nan")
    return {
        "Acc": float(accuracy_score(label, pred)),
        "Prec": float(precision_score(label, pred, zero_division=0)),
        "Rec": float(recall_score(label, pred, zero_division=0)),
        "F1": float(f1_score(label, pred, zero_division=0)),
        "AUC": float(auc),
    }


def rank_metrics(meta_test, y_true, score, top_frac=0.1):
    df = meta_test.copy()
    df["y"] = y_true
    df["score"] = score
    rankics, spreads, topwins = [], [], []
    for _, g in df.groupby("end_date"):
        if len(g) < 10 or g["score"].nunique() < 2:
            continue
        rankics.append(spearmanr(g["score"], g["y"]).correlation)
        k = max(1, int(len(g) * top_frac))
        long = g.nlargest(k, "score")
        short = g.nsmallest(k, "score")
        topwins.append((long["y"] > 0).mean())
        spreads.append(long["y"].mean() - short["y"].mean())
    rankics = np.asarray([x for x in rankics if np.isfinite(x)], dtype=np.float32)
    spreads = np.asarray(spreads, dtype=np.float32)
    topwins = np.asarray(topwins, dtype=np.float32)
    return {
        "RankIC": float(rankics.mean()) if len(rankics) else float("nan"),
        "ICIR": float(rankics.mean() / (rankics.std() + 1e-12)) if len(rankics) else float("nan"),
        "TopK_Win": float(topwins.mean()) if len(topwins) else float("nan"),
        "Spread": float(spreads.mean()) if len(spreads) else float("nan"),
    }


def compare_to_baseline(result):
    return {
        "return_mse_delta": float(result["return"]["MSE"] - BASELINE["return"]["MSE"]) if "return" in result else None,
        "auc_delta": float(result["classification"].get("AUC", np.nan) - BASELINE["classification"]["AUC"]) if "classification" in result else None,
        "rankic_delta": float(result["ranking"]["RankIC"] - BASELINE["ranking"]["RankIC"]),
        "spread_delta": float(result["ranking"]["Spread"] - BASELINE["ranking"]["Spread"]),
        "topk_win_delta": float(result["ranking"]["TopK_Win"] - BASELINE["ranking"]["TopK_Win"]),
    }


def prepare_base(repo_root, feature_dir, pca_components):
    csv_path = find_csv(repo_root)
    x_img, y_ret, meta = load_feature_dir(Path(feature_dir))
    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    train, val, test = split_70_15_15(meta)

    scaler_img = StandardScaler()
    pca = PCA(n_components=pca_components, random_state=42)
    pca.fit(scaler_img.fit_transform(x_img[train]))

    def img_pca(idx):
        return pca.transform(scaler_img.transform(x_img[idx])).astype(np.float32)

    x = {
        "train": np.concatenate([x_num[train], img_pca(train)], axis=1),
        "val": np.concatenate([x_num[val], img_pca(val)], axis=1),
        "test": np.concatenate([x_num[test], img_pca(test)], axis=1),
    }
    y = {"train": y_ret[train], "val": y_ret[val], "test": y_ret[test]}
    m = {"train": meta.iloc[train].reset_index(drop=True), "val": meta.iloc[val].reset_index(drop=True), "test": meta.iloc[test].reset_index(drop=True)}
    return x, y, m, {"total": len(y_ret), "train": len(train), "val": len(val), "test": len(test)}


def calibrate_by_val(y_train, y_val, val_pred, test_pred):
    train_mean = float(np.mean(y_train))
    val_pred = np.asarray(val_pred, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    pred_var = float(np.var(val_pred))
    if pred_var > 1e-12:
        slope = float(np.cov(val_pred, y_val, bias=True)[0, 1] / pred_var)
        intercept = float(np.mean(y_val) - slope * np.mean(val_pred))
    else:
        slope, intercept = 0.0, train_mean
    options = {
        "raw": (val_pred, np.asarray(test_pred, dtype=np.float64)),
        "calibrated": (intercept + slope * val_pred, intercept + slope * np.asarray(test_pred, dtype=np.float64)),
        "train_mean": (np.full_like(y_val, train_mean), np.full_like(np.asarray(test_pred, dtype=np.float64), train_mean)),
    }
    mode, (best_val, best_test) = min(options.items(), key=lambda item: mean_squared_error(y_val, item[1][0]))
    return mode, best_val, best_test


def run_fusion(args):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = find_csv(repo_root)
    feature_dirs = [Path(p) for p in args.feature_dirs.split(",")]
    x_first, y_ret, meta = load_feature_dir(feature_dirs[0])
    del x_first
    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    close_now = data_meta["close_now"].to_numpy(np.float32)
    future_close = close_now * (1.0 + y_ret)
    train, val, test = split_70_15_15(meta)
    parts = {
        "train": [x_num[train]],
        "val": [x_num[val]],
        "test": [x_num[test]],
    }
    feature_report = []
    for fdir in feature_dirs:
        x_img, y_check, meta_check = load_feature_dir(fdir)
        if len(y_check) != len(y_ret) or not np.allclose(y_check[:1000], y_ret[:1000]):
            raise ValueError(f"feature alignment mismatch: {fdir}")
        scaler = StandardScaler()
        pca = PCA(n_components=args.pca_components, random_state=42)
        train_scaled = scaler.fit_transform(x_img[train])
        pca.fit(train_scaled)
        parts["train"].append(pca.transform(train_scaled).astype(np.float32))
        parts["val"].append(pca.transform(scaler.transform(x_img[val])).astype(np.float32))
        parts["test"].append(pca.transform(scaler.transform(x_img[test])).astype(np.float32))
        feature_report.append({"dir": str(fdir), "raw_dim": int(x_img.shape[1]), "pca_dim": args.pca_components, "pca_var": float(np.sum(pca.explained_variance_ratio_))})
        del x_img, train_scaled

    x_train = np.concatenate(parts["train"], axis=1)
    x_val = np.concatenate(parts["val"], axis=1)
    x_test = np.concatenate(parts["test"], axis=1)
    threshold = float(np.median(y_ret[train]))
    candidates = {}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    ridge.fit(x_train, y_ret[train])
    candidates["RidgeFusion4"] = (ridge, ridge.predict(x_val), ridge.predict(x_test))

    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=320,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=42,
    )
    hgb.fit(x_train, y_ret[train])
    candidates["HGBFusion4"] = (hgb, hgb.predict(x_val), hgb.predict(x_test))

    # Validation-weighted blend, useful when linear and tree models capture different signals.
    ridge_val = candidates["RidgeFusion4"][1]
    ridge_test = candidates["RidgeFusion4"][2]
    hgb_val = candidates["HGBFusion4"][1]
    hgb_test = candidates["HGBFusion4"][2]
    best_w, best_mse = 0.0, float("inf")
    for w in np.linspace(0, 1, 21):
        pred = w * ridge_val + (1 - w) * hgb_val
        mse = mean_squared_error(y_ret[val], pred)
        if mse < best_mse:
            best_w, best_mse = float(w), float(mse)
    candidates[f"BlendFusion4_wRidge{best_w:.2f}"] = (None, best_w * ridge_val + (1 - best_w) * hgb_val, best_w * ridge_test + (1 - best_w) * hgb_test)

    result_items = []
    for name, (_, val_pred, test_pred) in candidates.items():
        mode, cal_val, cal_test = calibrate_by_val(y_ret[train], y_ret[val], val_pred, test_pred)
        item = {
            "stage": "multi_feature_fusion",
            "model": f"{name}+{mode}",
            "feature_report": feature_report,
            "samples": {"total": int(len(y_ret)), "train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
            "val_return": reg_metrics(y_ret[val], cal_val),
            "return": reg_metrics(y_ret[test], cal_test),
            "classification": class_metrics(y_ret[test], cal_test, threshold),
            "ranking": rank_metrics(meta.iloc[test].reset_index(drop=True), y_ret[test], cal_test),
        }
        item["vs_baseline"] = compare_to_baseline(item)
        result_items.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    close_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    close_model.fit(x_num[train], future_close[train])
    close_pred = close_model.predict(x_num[test])
    best = min(result_items, key=lambda r: r["val_return"]["MSE"])
    best["close"] = reg_metrics(future_close[test], close_pred)
    output = {"baseline": BASELINE, "results": result_items, "best_by_val_mse": best}
    out = Path(args.out)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


def prepare_fused_features(repo_root, feature_dirs, pca_components):
    csv_path = find_csv(repo_root)
    feature_dirs = [Path(p) for p in feature_dirs]
    x_first, y_ret, meta = load_feature_dir(feature_dirs[0])
    del x_first
    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    train, val, test = split_70_15_15(meta)
    parts = {"train": [x_num[train]], "val": [x_num[val]], "test": [x_num[test]]}
    report = []
    for fdir in feature_dirs:
        x_img, y_check, _ = load_feature_dir(fdir)
        if len(y_check) != len(y_ret) or not np.allclose(y_check[:1000], y_ret[:1000]):
            raise ValueError(f"feature alignment mismatch: {fdir}")
        scaler = StandardScaler()
        pca = PCA(n_components=pca_components, random_state=42)
        train_scaled = scaler.fit_transform(x_img[train])
        pca.fit(train_scaled)
        parts["train"].append(pca.transform(train_scaled).astype(np.float32))
        parts["val"].append(pca.transform(scaler.transform(x_img[val])).astype(np.float32))
        parts["test"].append(pca.transform(scaler.transform(x_img[test])).astype(np.float32))
        report.append({"dir": str(fdir), "raw_dim": int(x_img.shape[1]), "pca_dim": pca_components, "pca_var": float(np.sum(pca.explained_variance_ratio_))})
        del x_img, train_scaled
    return (
        {"train": np.concatenate(parts["train"], axis=1), "val": np.concatenate(parts["val"], axis=1), "test": np.concatenate(parts["test"], axis=1)},
        {"train": y_ret[train], "val": y_ret[val], "test": y_ret[test]},
        {"train": meta.iloc[train].reset_index(drop=True), "val": meta.iloc[val].reset_index(drop=True), "test": meta.iloc[test].reset_index(drop=True)},
        {"total": int(len(y_ret)), "train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        report,
    )


def cross_sectional_z(meta_subset, y):
    df = meta_subset[["end_date"]].copy()
    df["y"] = y
    z = np.zeros(len(df), dtype=np.float32)
    for _, pos in df.groupby("end_date", sort=False).groups.items():
        idx = np.asarray(list(pos), dtype=np.int64)
        vals = df["y"].to_numpy()[idx].astype(np.float32)
        z[idx] = (vals - vals.mean()) / (vals.std() + 1e-6)
    return z


def run_xsec(args):
    repo_root = Path(__file__).resolve().parents[1]
    x, y, meta, samples, report = prepare_fused_features(repo_root, args.feature_dirs.split(","), args.pca_components)
    y_z = {k: cross_sectional_z(meta[k], y[k]) for k in ("train", "val", "test")}
    threshold = float(np.median(y["train"]))
    candidates = {}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(x["train"], y_z["train"])
    candidates["RidgeXSecZ"] = (ridge.predict(x["val"]), ridge.predict(x["test"]))

    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=320,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        early_stopping=True,
        random_state=42,
    )
    hgb.fit(x["train"], y_z["train"])
    candidates["HGBXSecZ"] = (hgb.predict(x["val"]), hgb.predict(x["test"]))

    items = []
    for name, (val_score, test_score) in candidates.items():
        item = {
            "stage": "cross_sectional_label",
            "model": name,
            "feature_report": report,
            "samples": samples,
            "val_ranking": rank_metrics(meta["val"], y["val"], val_score),
            "classification": class_metrics(y["test"], test_score, threshold),
            "ranking": rank_metrics(meta["test"], y["test"], test_score),
        }
        item["vs_baseline"] = {
            "auc_delta": float(item["classification"]["AUC"] - BASELINE["classification"]["AUC"]),
            "rankic_delta": float(item["ranking"]["RankIC"] - BASELINE["ranking"]["RankIC"]),
            "spread_delta": float(item["ranking"]["Spread"] - BASELINE["ranking"]["Spread"]),
            "topk_win_delta": float(item["ranking"]["TopK_Win"] - BASELINE["ranking"]["TopK_Win"]),
        }
        items.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
    best = max(items, key=lambda r: (r["val_ranking"]["RankIC"], r["val_ranking"]["Spread"]))
    output = {"baseline": BASELINE, "results": items, "best_by_val_rankic": best}
    out = Path(args.out)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


def train_torch_mlp_regressor(x_train, y_train, x_val, args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    y_train_f = y_train.astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x_train_s.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.08),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.04),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    ds = TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y_train_f[:, None]))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_state, best_mse, bad = None, float("inf"), 0
    for epoch in range(args.mlp_epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(x_val_s).to(device)).squeeze(-1).cpu().numpy()
        mse = mean_squared_error(y_train[: len(val_pred)] if False else y_val_global, val_pred)
        print(f"torch_mlp epoch={epoch+1} val_mse={mse:.6f}")
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, device


def run_strong(args):
    # Stronger downstream models on the fused feature set, followed by validation stacking.
    repo_root = Path(__file__).resolve().parents[1]
    x, y, meta, samples, report = prepare_fused_features(repo_root, args.feature_dirs.split(","), args.pca_components)
    threshold = float(np.median(y["train"]))
    candidates = {}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
    ridge.fit(x["train"], y["train"])
    candidates["RidgeStrong"] = (ridge.predict(x["val"]), ridge.predict(x["test"]))

    for loss, name in [("squared_error", "HGBSquared"), ("absolute_error", "HGBAbsolute")]:
        hgb = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=420,
            learning_rate=0.025,
            max_leaf_nodes=43,
            l2_regularization=0.06,
            early_stopping=True,
            random_state=42,
        )
        hgb.fit(x["train"], y["train"])
        candidates[name] = (hgb.predict(x["val"]), hgb.predict(x["test"]))

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x["train"]).astype(np.float32)
        x_val_s = scaler.transform(x["val"]).astype(np.float32)
        x_test_s = scaler.transform(x["test"]).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Linear(x_train_s.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.04),
            nn.Linear(64, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
        ds = TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y["train"].astype(np.float32)[:, None]))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        best_state, best_mse, bad = None, float("inf"), 0
        for epoch in range(args.mlp_epochs):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(x_val_s).to(device)).squeeze(-1).cpu().numpy()
            mse = mean_squared_error(y["val"], val_pred)
            print(f"torch_mlp_reg epoch={epoch+1} val_mse={mse:.6f}")
            if mse < best_mse:
                best_mse = mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 2:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            mlp_val = model(torch.from_numpy(x_val_s).to(device)).squeeze(-1).cpu().numpy()
            mlp_test = model(torch.from_numpy(x_test_s).to(device)).squeeze(-1).cpu().numpy()
        candidates["TorchMLPReg"] = (mlp_val, mlp_test)
    except Exception as exc:
        print(f"TorchMLPReg failed: {exc}")

    names = list(candidates)
    val_matrix = np.vstack([candidates[n][0] for n in names]).T
    test_matrix = np.vstack([candidates[n][1] for n in names]).T
    best_weights, best_mse = None, float("inf")
    grid = np.linspace(0, 1, 11)
    if len(names) >= 3:
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                weights = np.array([a, b, c] + [0.0] * (len(names) - 3), dtype=np.float64)
                pred = val_matrix @ weights
                mse = mean_squared_error(y["val"], pred)
                if mse < best_mse:
                    best_mse, best_weights = mse, weights
    else:
        for a in grid:
            weights = np.array([a, 1 - a], dtype=np.float64)
            pred = val_matrix @ weights
            mse = mean_squared_error(y["val"], pred)
            if mse < best_mse:
                best_mse, best_weights = mse, weights
    if best_weights is not None:
        candidates["StackedBlend_" + "_".join(f"{n}:{w:.1f}" for n, w in zip(names, best_weights) if w > 1e-9)] = (
            val_matrix @ best_weights,
            test_matrix @ best_weights,
        )

    items = []
    for name, (val_pred, test_pred) in candidates.items():
        mode, cal_val, cal_test = calibrate_by_val(y["train"], y["val"], val_pred, test_pred)
        item = {
            "stage": "strong_models_stacking",
            "model": f"{name}+{mode}",
            "feature_report": report,
            "samples": samples,
            "val_return": reg_metrics(y["val"], cal_val),
            "return": reg_metrics(y["test"], cal_test),
            "classification": class_metrics(y["test"], cal_test, threshold),
            "ranking": rank_metrics(meta["test"], y["test"], cal_test),
        }
        item["vs_baseline"] = compare_to_baseline(item)
        items.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
    best = min(items, key=lambda r: r["val_return"]["MSE"])
    output = {"baseline": BASELINE, "results": items, "best_by_val_mse": best}
    out = Path(args.out)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


def prepare_single_all(repo_root, feature_dir, pca_components, train_for_pca_idx=None):
    csv_path = find_csv(repo_root)
    x_img, y_ret, meta = load_feature_dir(Path(feature_dir))
    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    if train_for_pca_idx is None:
        train_for_pca_idx = np.arange(len(y_ret))
    scaler = StandardScaler()
    pca = PCA(n_components=pca_components, random_state=42)
    pca.fit(scaler.fit_transform(x_img[train_for_pca_idx]))
    x_pca = pca.transform(scaler.transform(x_img)).astype(np.float32)
    x_all = np.concatenate([x_num, x_pca], axis=1)
    return x_all, y_ret, meta, {"feature_dir": str(feature_dir), "raw_dim": int(x_img.shape[1]), "pca_dim": pca_components, "pca_var": float(np.sum(pca.explained_variance_ratio_))}


def date_indices(meta, dates):
    mask = meta["end_date"].isin(set(dates)).to_numpy()
    return np.flatnonzero(mask)


def run_rolling(args):
    repo_root = Path(__file__).resolve().parents[1]
    # Fit the representation on the earliest 60% dates to avoid peeking at late test windows.
    temp_x, temp_y, temp_meta = load_feature_dir(Path(args.feature_dir))
    del temp_x
    unique_dates = np.array(sorted(temp_meta["end_date"].unique()))
    pca_dates = unique_dates[: int(len(unique_dates) * 0.60)]
    pca_idx = date_indices(temp_meta, pca_dates)
    x_all, y_all, meta, report = prepare_single_all(repo_root, args.feature_dir, args.pca_components, pca_idx)
    unique_dates = np.array(sorted(meta["end_date"].unique()))
    fold_specs = [(0.60, 0.70, 0.80), (0.70, 0.80, 0.90), (0.80, 0.90, 1.00)]
    folds = []
    for fold_id, (train_end, val_end, test_end) in enumerate(fold_specs, start=1):
        d_train = unique_dates[: int(len(unique_dates) * train_end)]
        d_val = unique_dates[int(len(unique_dates) * train_end) : int(len(unique_dates) * val_end)]
        d_test = unique_dates[int(len(unique_dates) * val_end) : int(len(unique_dates) * test_end)]
        train_idx = date_indices(meta, d_train)
        val_idx = date_indices(meta, d_val)
        test_idx = date_indices(meta, d_test)
        threshold = float(np.median(y_all[train_idx]))

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
        ridge.fit(x_all[train_idx], y_all[train_idx])
        val_pred = ridge.predict(x_all[val_idx])
        test_pred = ridge.predict(x_all[test_idx])
        mode, cal_val, cal_test = calibrate_by_val(y_all[train_idx], y_all[val_idx], val_pred, test_pred)
        ridge_item = {
            "model": f"RollingRidgeReturn+{mode}",
            "val_return": reg_metrics(y_all[val_idx], cal_val),
            "return": reg_metrics(y_all[test_idx], cal_test),
            "classification": class_metrics(y_all[test_idx], cal_test, threshold),
            "ranking": rank_metrics(meta.iloc[test_idx].reset_index(drop=True), y_all[test_idx], cal_test),
        }

        y_z_train = cross_sectional_z(meta.iloc[train_idx].reset_index(drop=True), y_all[train_idx])
        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=True,
            random_state=42,
        )
        hgb.fit(x_all[train_idx], y_z_train)
        rank_score = hgb.predict(x_all[test_idx])
        rank_item = {
            "model": "RollingHGBXSecZ",
            "classification": class_metrics(y_all[test_idx], rank_score, threshold),
            "ranking": rank_metrics(meta.iloc[test_idx].reset_index(drop=True), y_all[test_idx], rank_score),
        }
        fold = {
            "fold": fold_id,
            "date_ranges": {"train_end": str(d_train[-1]), "val": [str(d_val[0]), str(d_val[-1])], "test": [str(d_test[0]), str(d_test[-1])]},
            "samples": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "ridge_return": ridge_item,
            "hgb_xsec_rank": rank_item,
        }
        folds.append(fold)
        print(json.dumps(fold, ensure_ascii=False, indent=2))

    def summarize(path):
        vals = []
        for f in folds:
            cur = f
            for key in path:
                cur = cur[key]
            vals.append(float(cur))
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    summary = {
        "ridge_return_mse": summarize(["ridge_return", "return", "MSE"]),
        "ridge_auc": summarize(["ridge_return", "classification", "AUC"]),
        "ridge_rankic": summarize(["ridge_return", "ranking", "RankIC"]),
        "hgb_xsec_rankic": summarize(["hgb_xsec_rank", "ranking", "RankIC"]),
        "hgb_xsec_topk_win": summarize(["hgb_xsec_rank", "ranking", "TopK_Win"]),
        "hgb_xsec_spread": summarize(["hgb_xsec_rank", "ranking", "Spread"]),
    }
    output = {"baseline_single_split": BASELINE, "feature_report": report, "folds": folds, "summary": summary}
    out = Path(args.out)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved {out}")


def run_ranker(args):
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    x, y, meta, samples = prepare_base(repo_root, args.feature_dir, args.pca_components)
    threshold = float(np.median(y["train"]))

    results = []
    train_idx, train_meta_sorted = sort_by_date(meta["train"], np.arange(len(meta["train"])))
    val_idx, val_meta_sorted = sort_by_date(meta["val"], np.arange(len(meta["val"])))
    test_idx, test_meta_sorted = sort_by_date(meta["test"], np.arange(len(meta["test"])))
    x_train, y_train = x["train"][train_idx], y["train"][train_idx]
    x_val, y_val = x["val"][val_idx], y["val"][val_idx]
    x_test, y_test = x["test"][test_idx], y["test"][test_idx]
    train_group = make_groups(train_meta_sorted)
    val_group = make_groups(val_meta_sorted)

    try:
        import torch
        from torch import nn

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train).astype(np.float32)
        x_val_s = scaler.transform(x_val).astype(np.float32)
        x_test_s = scaler.transform(x_test).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Linear(x_train_s.shape[1], 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)

        def group_slices(groups):
            start = 0
            out = []
            for size in groups:
                out.append(slice(start, start + size))
                start += size
            return out

        train_slices = group_slices(train_group)
        val_slices = group_slices(val_group)
        rng = np.random.default_rng(42)
        best_state, best_val = None, -1e9
        patience, bad = 3, 0
        for epoch in range(args.torch_epochs):
            model.train()
            for si in rng.permutation(len(train_slices)):
                sl = train_slices[si]
                xb = torch.from_numpy(x_train_s[sl]).to(device)
                yy = y_train[sl].astype(np.float32)
                if len(yy) < 5 or np.nanstd(yy) < 1e-12:
                    continue
                target = torch.softmax(torch.from_numpy((yy - yy.mean()) / (yy.std() + 1e-6)).to(device), dim=0)
                pred = model(xb).squeeze(-1)
                loss = -(target * torch.log_softmax(pred, dim=0)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            val_score_parts = []
            with torch.no_grad():
                for sl in val_slices:
                    xb = torch.from_numpy(x_val_s[sl]).to(device)
                    val_score_parts.append(model(xb).squeeze(-1).detach().cpu().numpy())
            val_score = np.concatenate(val_score_parts)
            val_rank = rank_metrics(val_meta_sorted, y_val, val_score)
            metric = val_rank["RankIC"] + 0.1 * val_rank["Spread"]
            print(f"torch_listnet epoch={epoch+1} val_rankic={val_rank['RankIC']:.6f} val_spread={val_rank['Spread']:.6f}")
            if metric > best_val:
                best_val = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        test_score_parts = []
        val_score_parts = []
        with torch.no_grad():
            for sl in val_slices:
                xb = torch.from_numpy(x_val_s[sl]).to(device)
                val_score_parts.append(model(xb).squeeze(-1).detach().cpu().numpy())
            start = 0
            for size in make_groups(test_meta_sorted):
                sl = slice(start, start + size)
                xb = torch.from_numpy(x_test_s[sl]).to(device)
                test_score_parts.append(model(xb).squeeze(-1).detach().cpu().numpy())
                start += size
        val_score = np.concatenate(val_score_parts)
        test_score = np.concatenate(test_score_parts)
        item = {
            "stage": "ranker",
            "model": "torch_listnet_ranker",
            "feature_dir": args.feature_dir,
            "samples": samples,
            "val_ranking": rank_metrics(val_meta_sorted, y_val, val_score),
            "return": reg_metrics(y_test, test_score),
            "classification": class_metrics(y_test, test_score, threshold),
            "ranking": rank_metrics(test_meta_sorted, y_test, test_score),
        }
        item["vs_baseline"] = compare_to_baseline(item)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"torch_listnet failed: {exc}")

    for name in ("lightgbm", "xgboost"):
        if importlib.util.find_spec(name) is None:
            print(f"skip {name}: not installed")
            continue
        y_rank_train = cross_sectional_rank_labels(train_meta_sorted, y_train, bins=10)
        y_rank_val = cross_sectional_rank_labels(val_meta_sorted, y_val, bins=10)
        if name == "lightgbm":
            from lightgbm import LGBMRanker

            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=args.rank_estimators,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                x_train,
                y_rank_train,
                group=train_group,
                eval_set=[(x_val, y_rank_val)],
                eval_group=[val_group],
                eval_at=[5, 10, 20],
            )
        else:
            from xgboost import XGBRanker

            model = XGBRanker(
                objective="rank:ndcg",
                n_estimators=max(80, args.rank_estimators // 2),
                learning_rate=0.035,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=42,
                tree_method="hist",
                n_jobs=-1,
            )
            model.fit(x_train, y_rank_train, group=train_group, eval_set=[(x_val, y_rank_val)], eval_group=[val_group], verbose=False)
        val_score = model.predict(x_val)
        test_score = model.predict(x_test)
        item = {
            "stage": "ranker",
            "model": f"{name}_ranker",
            "feature_dir": args.feature_dir,
            "samples": samples,
            "val_ranking": rank_metrics(val_meta_sorted, y_val, val_score),
            "return": reg_metrics(y_test, test_score),
            "classification": class_metrics(y_test, test_score, threshold),
            "ranking": rank_metrics(test_meta_sorted, y_test, test_score),
        }
        item["vs_baseline"] = compare_to_baseline(item)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    if not results:
        raise RuntimeError("no ranker library available")
    best = max(results, key=lambda r: (r["val_ranking"]["RankIC"], r["val_ranking"]["Spread"]))
    output = {"baseline": BASELINE, "results": results, "best_by_val_rankic": best}
    out = Path(args.out)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ranker", "fusion", "xsec", "strong", "rolling"], required=True)
    ap.add_argument("--feature-dir", default="/root/autodl-tmp/aligned_features/mixed_mae")
    ap.add_argument(
        "--feature-dirs",
        default="/root/autodl-tmp/aligned_features/mixed_vit,/root/autodl-tmp/aligned_features/mixed_mae,/root/autodl-tmp/aligned_features/separate_vit,/root/autodl-tmp/aligned_features/separate_mae",
    )
    ap.add_argument("--pca-components", type=int, default=32)
    ap.add_argument("--rank-estimators", type=int, default=180)
    ap.add_argument("--torch-epochs", type=int, default=12)
    ap.add_argument("--mlp-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", default="/root/autodl-tmp/iterative_ranker_results.json")
    args = ap.parse_args()
    if args.stage == "ranker":
        run_ranker(args)
    elif args.stage == "fusion":
        run_fusion(args)
    elif args.stage == "xsec":
        run_xsec(args)
    elif args.stage == "strong":
        run_strong(args)
    elif args.stage == "rolling":
        run_rolling(args)


if __name__ == "__main__":
    main()
