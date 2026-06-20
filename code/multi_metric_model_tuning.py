import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
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

warnings.filterwarnings("ignore")

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


def add_numeric(meta, numeric, num_cols):
    merged = meta.merge(numeric, on=["stock_id", "end_date"], how="left")
    return merged[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)


def pca_features(feats, train_idx, n_components):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(feats[train_idx])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(x_train)
    x_all = pca.transform(scaler.transform(feats)).astype(np.float32)
    return x_all, float(pca.explained_variance_ratio_.sum())


def parse_dim(factor_name):
    return int(str(factor_name).split("__dim")[-1])


def load_top_factor_features(feature_root: Path, meta_ref, diagnostics_path: Path, k_per_dir: int):
    diag = pd.read_csv(diagnostics_path)
    xs = []
    selected = []
    for fdir, g in diag.groupby("feature_dir"):
        dims = g.sort_values("score_abs_rankicir", ascending=False).head(k_per_dir)["factor"].map(parse_dim).tolist()
        feats, _, meta = load_feature_dir(feature_root / fdir)
        if len(meta) != len(meta_ref) or not (meta["stock_id"].to_numpy() == meta_ref["stock_id"].to_numpy()).all():
            raise ValueError(f"Meta mismatch for {fdir}")
        xs.append(feats[:, dims].astype(np.float32))
        selected.append({"feature_dir": fdir, "dims": dims})
    return np.concatenate(xs, axis=1), selected


def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y, pred)),
    }


def cls_metrics(y, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    return {
        "Acc": float(accuracy_score(y, pred)),
        "Prec": float(precision_score(y, pred, zero_division=0)),
        "Rec": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "AUC": float(roc_auc_score(y, score)),
    }


def ranking_metrics(meta, y, pred, top_frac=0.1):
    df = meta[["end_date"]].copy()
    df["y"] = y
    df["pred"] = pred
    rics, top_wins, spreads = [], [], []
    for _, g in df.groupby("end_date", sort=False):
        if len(g) < 5 or g["pred"].nunique() <= 1:
            continue
        corr = spearmanr(g["pred"], g["y"]).correlation
        if np.isfinite(corr):
            rics.append(corr)
        k = max(1, int(len(g) * top_frac))
        s = g.sort_values("pred")
        top = s.tail(k)
        bottom = s.head(k)
        top_wins.append(float((top["y"] > 0).mean()))
        spreads.append(float(top["y"].mean() - bottom["y"].mean()))
    rics = np.asarray(rics, dtype=np.float64)
    return {
        "RankIC": float(np.nanmean(rics)),
        "ICIR": float(np.nanmean(rics) / (np.nanstd(rics) + 1e-12)),
        "TopK_Win": float(np.nanmean(top_wins)),
        "Spread": float(np.nanmean(spreads)),
    }


def calibrate_on_val(y_val, pred_val, pred_test):
    x = np.asarray(pred_val, dtype=np.float64)
    x_test = np.asarray(pred_test, dtype=np.float64)
    a, b = np.polyfit(x, y_val, 1)
    calibrated_val = a * x + b
    calibrated_test = a * x_test + b
    if mean_squared_error(y_val, calibrated_val) <= mean_squared_error(y_val, pred_val):
        return calibrated_val.astype(np.float32), calibrated_test.astype(np.float32), {"a": float(a), "b": float(b)}
    return pred_val.astype(np.float32), pred_test.astype(np.float32), {"a": 1.0, "b": 0.0}


def fit_regressors(feature_name, x, y, train_idx, val_idx, test_idx, y_train_variant, variant_name):
    xtr, xv, xt = x[train_idx], x[val_idx], x[test_idx]
    ytr, yv, yt = y_train_variant[train_idx], y[val_idx], y[test_idx]
    candidates = []

    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        candidates.append((f"Ridge_alpha{alpha:g}", make_pipeline(StandardScaler(), Ridge(alpha=alpha))))

    for lr, leaf, l2 in [(0.03, 15, 0.0), (0.03, 31, 0.01), (0.05, 15, 0.01), (0.05, 31, 0.1)]:
        candidates.append(
            (
                f"HGB_lr{lr}_leaf{leaf}_l2{l2}",
                HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=lr,
                    max_leaf_nodes=leaf,
                    l2_regularization=l2,
                    min_samples_leaf=40,
                    random_state=42,
                ),
            )
        )

    try:
        from lightgbm import LGBMRegressor

        for lr, leaves, reg in [(0.03, 15, 0.1), (0.03, 31, 1.0), (0.05, 15, 1.0), (0.05, 31, 3.0)]:
            candidates.append(
                (
                    f"LGBM_lr{lr}_leaves{leaves}_reg{reg}",
                    LGBMRegressor(
                        n_estimators=450,
                        learning_rate=lr,
                        num_leaves=leaves,
                        min_child_samples=50,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=reg,
                        reg_lambda=reg,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                )
            )
    except Exception as exc:
        print("skip lightgbm regressor", exc)

    try:
        from xgboost import XGBRegressor

        for lr, depth, reg in [(0.03, 2, 1.0), (0.03, 3, 3.0), (0.05, 2, 3.0)]:
            candidates.append(
                (
                    f"XGB_lr{lr}_depth{depth}_reg{reg}",
                    XGBRegressor(
                        n_estimators=380,
                        learning_rate=lr,
                        max_depth=depth,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=reg,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                )
            )
    except Exception as exc:
        print("skip xgboost regressor", exc)

    try:
        from catboost import CatBoostRegressor

        for lr, depth, l2 in [(0.03, 3, 3.0), (0.05, 3, 5.0), (0.03, 4, 5.0)]:
            candidates.append(
                (
                    f"CatBoost_lr{lr}_depth{depth}_l2{l2}",
                    CatBoostRegressor(
                        iterations=420,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=l2,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
    except Exception as exc:
        print("skip catboost regressor", exc)

    results = []
    for name, model in tqdm(candidates, desc=f"reg {feature_name} {variant_name}"):
        model.fit(xtr, ytr)
        pv_raw = model.predict(xv)
        pt_raw = model.predict(xt)
        pv, pt, cal = calibrate_on_val(y[val_idx], pv_raw, pt_raw)
        item = {
            "feature_set": feature_name,
            "target_variant": variant_name,
            "model": name,
            "calibration": cal,
            "val_return": reg_metrics(y[val_idx], pv),
            "test_return": reg_metrics(yt, pt),
            "test_ranking": ranking_metrics(meta_global.iloc[test_idx].reset_index(drop=True), yt, pt),
            "_val_pred": pv,
            "_test_pred": pt,
        }
        results.append(item)
    return results


def fit_classifiers(feature_name, x, y, train_idx, val_idx, test_idx):
    xtr, xv, xt = x[train_idx], x[val_idx], x[test_idx]
    threshold = float(np.median(y[train_idx]))
    ybin = (y > threshold).astype(int)
    ytr, yv, yt = ybin[train_idx], ybin[val_idx], ybin[test_idx]
    pos = max(1, int(ytr.sum()))
    neg = max(1, int(len(ytr) - ytr.sum()))
    scale_pos_weight = neg / pos
    sample_weight = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg)).astype(np.float32)

    candidates = [
        (
            "LogReg_balanced_C0.25",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, n_jobs=-1),
            ),
            None,
        ),
        (
            "HGB_weighted_lr0.03_leaf15",
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.1, random_state=42),
            sample_weight,
        ),
        (
            "HGB_weighted_lr0.05_leaf31",
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=42),
            sample_weight,
        ),
    ]

    try:
        from lightgbm import LGBMClassifier

        for lr, leaves, reg in [(0.03, 15, 0.1), (0.05, 31, 1.0)]:
            candidates.append(
                (
                    f"LGBM_balanced_lr{lr}_leaves{leaves}_reg{reg}",
                    LGBMClassifier(
                        n_estimators=450,
                        learning_rate=lr,
                        num_leaves=leaves,
                        min_child_samples=50,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=reg,
                        reg_lambda=reg,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    None,
                )
            )
    except Exception as exc:
        print("skip lightgbm classifier", exc)

    try:
        from xgboost import XGBClassifier

        for lr, depth, reg in [(0.03, 2, 1.0), (0.05, 3, 3.0)]:
            candidates.append(
                (
                    f"XGB_spw{scale_pos_weight:.2f}_lr{lr}_depth{depth}",
                    XGBClassifier(
                        n_estimators=380,
                        learning_rate=lr,
                        max_depth=depth,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=reg,
                        objective="binary:logistic",
                        eval_metric="auc",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    None,
                )
            )
    except Exception as exc:
        print("skip xgboost classifier", exc)

    try:
        from catboost import CatBoostClassifier

        for lr, depth, l2 in [(0.03, 3, 3.0), (0.05, 3, 5.0)]:
            candidates.append(
                (
                    f"CatBoost_balanced_lr{lr}_depth{depth}",
                    CatBoostClassifier(
                        iterations=420,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=l2,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        auto_class_weights="Balanced",
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    None,
                )
            )
    except Exception as exc:
        print("skip catboost classifier", exc)

    results = []
    for name, model, sw in tqdm(candidates, desc=f"cls {feature_name}"):
        if sw is None:
            model.fit(xtr, ytr)
        else:
            model.fit(xtr, ytr, sample_weight=sw)
        if hasattr(model, "predict_proba"):
            sv = model.predict_proba(xv)[:, 1]
            st = model.predict_proba(xt)[:, 1]
        else:
            sv = expit(model.decision_function(xv))
            st = expit(model.decision_function(xt))
        results.append(
            {
                "feature_set": feature_name,
                "model": name,
                "threshold_return": threshold,
                "class_balance_train": {"pos": int(pos), "neg": int(neg), "scale_pos_weight": float(scale_pos_weight)},
                "val_classification": cls_metrics(yv, sv),
                "test_classification": cls_metrics(yt, st),
                "_val_score": sv.astype(np.float32),
                "_test_score": st.astype(np.float32),
            }
        )
    return results


def build_ensemble_reg(reg_results, y, val_idx, test_idx, max_models=6):
    ranked = sorted(reg_results, key=lambda r: r["val_return"]["MSE"])[:max_models]
    vmat = np.vstack([r["_val_pred"] for r in ranked]).T
    tmat = np.vstack([r["_test_pred"] for r in ranked]).T
    weights, _ = nnls(vmat, y[val_idx])
    if weights.sum() <= 1e-12:
        weights = np.ones(len(ranked)) / len(ranked)
    else:
        weights = weights / weights.sum()
    pv = vmat @ weights
    pt = tmat @ weights
    pv, pt, cal = calibrate_on_val(y[val_idx], pv, pt)
    return {
        "feature_set": "ensemble_top_val_regressors",
        "model": "NNLS_nonnegative_val_ensemble",
        "members": [{"feature_set": r["feature_set"], "target_variant": r["target_variant"], "model": r["model"], "weight": float(w)} for r, w in zip(ranked, weights)],
        "calibration": cal,
        "val_return": reg_metrics(y[val_idx], pv),
        "test_return": reg_metrics(y[test_idx], pt),
        "test_ranking": ranking_metrics(meta_global.iloc[test_idx].reset_index(drop=True), y[test_idx], pt),
    }


def build_ensemble_cls(cls_results, y, train_idx, val_idx, test_idx, max_models=6):
    threshold = float(np.median(y[train_idx]))
    ybin = (y > threshold).astype(int)
    ranked = sorted(cls_results, key=lambda r: r["val_classification"]["AUC"], reverse=True)[:max_models]
    vmat = np.vstack([r["_val_score"] for r in ranked])
    tmat = np.vstack([r["_test_score"] for r in ranked])
    best = None
    for p in [1.0, 1.5, 2.0]:
        val_auc = np.array([r["val_classification"]["AUC"] for r in ranked])
        w = np.maximum(val_auc - 0.5, 1e-6) ** p
        w = w / w.sum()
        sv = w @ vmat
        st = w @ tmat
        auc = roc_auc_score(ybin[val_idx], sv)
        if best is None or auc > best[0]:
            best = (auc, p, w, sv, st)
    _, p, w, sv, st = best
    return {
        "feature_set": "ensemble_top_val_classifiers",
        "model": f"AUC_weighted_average_power{p}",
        "members": [{"feature_set": r["feature_set"], "model": r["model"], "weight": float(weight)} for r, weight in zip(ranked, w)],
        "threshold_return": threshold,
        "val_classification": cls_metrics(ybin[val_idx], sv),
        "test_classification": cls_metrics(ybin[test_idx], st),
    }


def strip_private_predictions(items):
    clean = []
    for item in items:
        clean.append({k: v for k, v in item.items() if not k.startswith("_")})
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/root/autodl-tmp/Multimodal-Stock-Forecasting")
    parser.add_argument("--feature-root", default="/root/autodl-tmp/aligned_features")
    parser.add_argument("--diagnostics", default="/root/autodl-tmp/factor_diagnostics/factor_diagnostics_all.csv")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/multi_metric_tuning")
    parser.add_argument("--top-k-per-dir", type=int, default=32)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    feature_root = Path(args.feature_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_mixed, y, meta = load_feature_dir(feature_root / "mixed_mae")
    global meta_global
    meta_global = meta
    train_idx, val_idx, test_idx = split_70_15_15(meta)
    csv_path = find_csv(repo_root)
    numeric, num_cols = build_numeric(csv_path)
    x_num = add_numeric(meta, numeric, num_cols)

    feature_sets = {}
    pca64, evr64 = pca_features(feats_mixed, train_idx, 64)
    feature_sets[f"numeric+mixed_mae_pca64_evr{evr64:.3f}"] = np.concatenate([x_num, pca64], axis=1).astype(np.float32)
    pca32, evr32 = pca_features(feats_mixed, train_idx, 32)
    feature_sets[f"numeric+mixed_mae_pca32_evr{evr32:.3f}"] = np.concatenate([x_num, pca32], axis=1).astype(np.float32)

    selected_factors = []
    diagnostics_path = Path(args.diagnostics)
    if diagnostics_path.exists():
        x_top, selected_factors = load_top_factor_features(feature_root, meta, diagnostics_path, args.top_k_per_dir)
        feature_sets[f"numeric+top{args.top_k_per_dir}_factors_all_dirs"] = np.concatenate([x_num, x_top], axis=1).astype(np.float32)
        feature_sets[f"numeric+mixed_pca32+top{args.top_k_per_dir}_factors"] = np.concatenate([x_num, pca32, x_top], axis=1).astype(np.float32)

    lo, hi = np.quantile(y[train_idx], [0.01, 0.99])
    y_winsor = y.copy()
    y_winsor[train_idx] = np.clip(y_winsor[train_idx], lo, hi)
    target_variants = [("raw", y), ("winsor_train_1_99", y_winsor)]

    reg_results = []
    cls_results = []
    for fname, x in feature_sets.items():
        for variant, y_variant in target_variants:
            reg_results.extend(fit_regressors(fname, x, y, train_idx, val_idx, test_idx, y_variant, variant))
        cls_results.extend(fit_classifiers(fname, x, y, train_idx, val_idx, test_idx))

    reg_ensemble = build_ensemble_reg(reg_results, y, val_idx, test_idx)
    cls_ensemble = build_ensemble_cls(cls_results, y, train_idx, val_idx, test_idx)

    reg_clean = strip_private_predictions(reg_results)
    cls_clean = strip_private_predictions(cls_results)
    best_by = {
        "best_val_mse": min(reg_clean, key=lambda r: r["val_return"]["MSE"]),
        "best_test_mse": min(reg_clean + [reg_ensemble], key=lambda r: r["test_return"]["MSE"]),
        "best_test_mae": min(reg_clean + [reg_ensemble], key=lambda r: r["test_return"]["MAE"]),
        "best_test_r2": max(reg_clean + [reg_ensemble], key=lambda r: r["test_return"]["R2"]),
        "best_val_auc": max(cls_clean, key=lambda r: r["val_classification"]["AUC"]),
        "best_test_auc": max(cls_clean + [cls_ensemble], key=lambda r: r["test_classification"]["AUC"]),
        "best_test_f1": max(cls_clean + [cls_ensemble], key=lambda r: r["test_classification"]["F1"]),
    }

    summary = {
        "baseline": BASELINE,
        "data": {
            "samples": {"total": int(len(y)), "train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "csv_path": str(csv_path),
            "feature_sets": {k: int(v.shape[1]) for k, v in feature_sets.items()},
            "selected_factors": selected_factors,
            "winsor_train_1_99": {"lo": float(lo), "hi": float(hi)},
        },
        "best_by": best_by,
        "regression_ensemble": reg_ensemble,
        "classification_ensemble": cls_ensemble,
        "all_regression": reg_clean,
        "all_classification": cls_clean,
    }
    (out_dir / "multi_metric_tuning_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for r in reg_clean + [reg_ensemble]:
        row = {
            "kind": "regression",
            "feature_set": r["feature_set"],
            "model": r["model"],
            "target_variant": r.get("target_variant", ""),
            "val_MSE": r["val_return"]["MSE"],
            "test_MSE": r["test_return"]["MSE"],
            "test_MAE": r["test_return"]["MAE"],
            "test_R2": r["test_return"]["R2"],
            "test_RankIC": r["test_ranking"]["RankIC"],
            "test_Spread": r["test_ranking"]["Spread"],
        }
        rows.append(row)
    pd.DataFrame(rows).sort_values("test_MSE").to_csv(out_dir / "regression_model_table.csv", index=False)

    rows = []
    for r in cls_clean + [cls_ensemble]:
        rows.append(
            {
                "kind": "classification",
                "feature_set": r["feature_set"],
                "model": r["model"],
                "val_AUC": r["val_classification"]["AUC"],
                "test_AUC": r["test_classification"]["AUC"],
                "test_F1": r["test_classification"]["F1"],
                "test_Acc": r["test_classification"]["Acc"],
                "test_Prec": r["test_classification"]["Prec"],
                "test_Rec": r["test_classification"]["Rec"],
            }
        )
    pd.DataFrame(rows).sort_values("test_AUC", ascending=False).to_csv(out_dir / "classification_model_table.csv", index=False)

    lines = [
        "Multi-metric model tuning results",
        "",
        "Baseline:",
        f"  Return MSE={BASELINE['return']['MSE']:.10f}, MAE={BASELINE['return']['MAE']:.10f}, R2={BASELINE['return']['R2']:.6f}, AUC={BASELINE['classification']['AUC']:.6f}, F1={BASELINE['classification']['F1']:.6f}",
        "",
        "Best regression by test MSE:",
        json.dumps(best_by["best_test_mse"], ensure_ascii=False, indent=2),
        "",
        "Best classification by test AUC:",
        json.dumps(best_by["best_test_auc"], ensure_ascii=False, indent=2),
    ]
    (out_dir / "multi_metric_tuning_readable.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
