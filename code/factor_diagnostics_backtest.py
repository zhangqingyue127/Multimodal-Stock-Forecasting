import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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


def corr_vec(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denom = np.sqrt((x * x).sum(axis=0) * float((y * y).sum())) + 1e-12
    return (x * y[:, None]).sum(axis=0) / denom


def daily_factor_stats(x_train, y_train, meta_train, min_names):
    ic_list, ric_list = [], []
    for _, pos in tqdm(meta_train.groupby("end_date", sort=False).groups.items(), desc="daily IC"):
        idx = np.asarray(list(pos), dtype=np.int64)
        if len(idx) < 10 or np.nanstd(y_train[idx]) < 1e-12:
            continue
        xd = x_train[idx]
        yd = y_train[idx]
        ic_list.append(corr_vec(xd, yd))
        xr = np.apply_along_axis(rankdata, 0, xd)
        yr = rankdata(yd)
        ric_list.append(corr_vec(xr, yr))
    ic = np.asarray(ic_list, dtype=np.float32)
    ric = np.asarray(ric_list, dtype=np.float32)
    ic_mean = np.nanmean(ic, axis=0)
    ic_std = np.nanstd(ic, axis=0) + 1e-12
    ric_mean = np.nanmean(ric, axis=0)
    ric_std = np.nanstd(ric, axis=0) + 1e-12
    out = pd.DataFrame(
        {
            "factor": min_names,
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "ICIR": ic_mean / ic_std,
            "IC_pos_ratio": (ic > 0).mean(axis=0),
            "RankIC_mean": ric_mean,
            "RankIC_std": ric_std,
            "RankICIR": ric_mean / ric_std,
            "RankIC_pos_ratio": (ric > 0).mean(axis=0),
            "score_abs_rankicir": np.abs(ric_mean / ric_std),
        }
    )
    return out.sort_values("score_abs_rankicir", ascending=False)


def reg_metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y, pred)),
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
    return {
        "RankIC": float(rankics.mean()) if len(rankics) else float("nan"),
        "ICIR": float(rankics.mean() / (rankics.std() + 1e-12)) if len(rankics) else float("nan"),
        "TopK_Win": float(np.mean(topwins)) if topwins else float("nan"),
        "Spread": float(np.mean(spreads)) if spreads else float("nan"),
    }


def backtest_metrics(meta_test, y_true, score, top_frac=0.1):
    df = meta_test.copy()
    df["ret"] = y_true
    df["score"] = score
    rows, prev_long, prev_short = [], None, None
    for date, g in df.groupby("end_date"):
        if len(g) < 10 or g["score"].nunique() < 2:
            continue
        k = max(1, int(len(g) * top_frac))
        long = g.nlargest(k, "score")
        short = g.nsmallest(k, "score")
        long_set, short_set = set(long["stock_id"]), set(short["stock_id"])
        turnover = np.nan
        if prev_long is not None:
            long_turn = 1.0 - len(long_set & prev_long) / max(1, len(long_set))
            short_turn = 1.0 - len(short_set & prev_short) / max(1, len(short_set))
            turnover = 0.5 * (long_turn + short_turn)
        prev_long, prev_short = long_set, short_set
        rows.append(
            {
                "date": date,
                "long": float(long["ret"].mean()),
                "short": float(short["ret"].mean()),
                "long_short": float(long["ret"].mean() - short["ret"].mean()),
                "turnover": turnover,
            }
        )
    daily = pd.DataFrame(rows)
    def perf(col):
        r = daily[col].to_numpy(np.float64)
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        ann_ret = float(equity[-1] ** (252.0 / len(r)) - 1.0) if len(r) else float("nan")
        ann_vol = float(np.std(r) * np.sqrt(252.0)) if len(r) else float("nan")
        sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252.0)) if len(r) else float("nan")
        return {"annual_return": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe, "max_drawdown": float(dd.min()) if len(r) else float("nan")}
    return {
        "days": int(len(daily)),
        "long": perf("long"),
        "short": perf("short"),
        "long_short": perf("long_short"),
        "avg_turnover": float(daily["turnover"].dropna().mean()) if len(daily) > 1 else float("nan"),
        "daily_spread_mean": float(daily["long_short"].mean()) if len(daily) else float("nan"),
    }, daily


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
    opts = {
        "raw": (val_pred, np.asarray(test_pred, dtype=np.float64)),
        "calibrated": (intercept + slope * val_pred, intercept + slope * np.asarray(test_pred, dtype=np.float64)),
        "train_mean": (np.full_like(y_val, train_mean), np.full_like(np.asarray(test_pred, dtype=np.float64), train_mean)),
    }
    return min(opts.items(), key=lambda item: mean_squared_error(y_val, item[1][0]))


def cross_sectional_z(meta_subset, y):
    df = meta_subset[["end_date"]].copy()
    df["y"] = y
    z = np.zeros(len(df), dtype=np.float32)
    for _, pos in df.groupby("end_date", sort=False).groups.items():
        idx = np.asarray(list(pos), dtype=np.int64)
        vals = df["y"].to_numpy()[idx].astype(np.float32)
        z[idx] = (vals - vals.mean()) / (vals.std() + 1e-6)
    return z


def fit_predict_models(x_train, y_train, x_val, y_val, x_test):
    results = {}
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
    ridge.fit(x_train, y_train)
    results["Ridge"] = (ridge.predict(x_val), ridge.predict(x_test))
    hgb = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.08, early_stopping=True, random_state=42)
    hgb.fit(x_train, y_train)
    results["HGB"] = (hgb.predict(x_val), hgb.predict(x_test))
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=260, learning_rate=0.035, max_depth=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=42)
        xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        results["XGBRegressor"] = (xgb.predict(x_val), xgb.predict(x_test))
    except Exception as exc:
        print(f"XGBRegressor failed: {exc}")
    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(n_estimators=260, learning_rate=0.035, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1)
        lgbm.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        results["LGBMRegressor"] = (lgbm.predict(x_val), lgbm.predict(x_test))
    except Exception as exc:
        print(f"LGBMRegressor failed: {exc}")
    try:
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(iterations=260, learning_rate=0.035, depth=5, loss_function="RMSE", random_seed=42, verbose=False, allow_writing_files=False)
        cb.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
        results["CatBoostRegressor"] = (cb.predict(x_val), cb.predict(x_test))
    except Exception as exc:
        print(f"CatBoostRegressor failed: {exc}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dirs", default="/root/autodl-tmp/aligned_features/mixed_vit,/root/autodl-tmp/aligned_features/mixed_mae,/root/autodl-tmp/aligned_features/separate_vit,/root/autodl-tmp/aligned_features/separate_mae")
    ap.add_argument("--top-ks", default="32,64,128")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/factor_diagnostics")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    feature_dirs = [Path(p) for p in args.feature_dirs.split(",")]
    top_ks = [int(x) for x in args.top_ks.split(",")]

    x0, y, meta = load_feature_dir(feature_dirs[0])
    del x0
    train, val, test = split_70_15_15(meta)
    meta_train = meta.iloc[train].reset_index(drop=True)
    meta_val = meta.iloc[val].reset_index(drop=True)
    meta_test = meta.iloc[test].reset_index(drop=True)
    csv_path = find_csv(repo_root)
    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)

    all_diag = []
    selected = {k: [] for k in top_ks}
    cache_selected = {}
    for fdir in feature_dirs:
        x, y_check, _ = load_feature_dir(fdir)
        if len(y_check) != len(y) or not np.allclose(y_check[:1000], y[:1000]):
            raise ValueError(f"feature alignment mismatch: {fdir}")
        names = [f"{fdir.name}__dim{i}" for i in range(x.shape[1])]
        diag = daily_factor_stats(x[train], y[train], meta_train, names)
        diag.insert(0, "feature_dir", fdir.name)
        diag.to_csv(out_dir / f"factor_diagnostics_{fdir.name}.csv", index=False)
        all_diag.append(diag)
        max_k = max(top_ks)
        top_indices = [int(s.split("__dim")[-1]) for s in diag.head(max_k)["factor"]]
        cache_selected[fdir.name] = {
            "indices": top_indices,
            "train": x[train][:, top_indices],
            "val": x[val][:, top_indices],
            "test": x[test][:, top_indices],
            "diag": diag.head(max_k).to_dict(orient="records"),
        }
        for k in top_ks:
            selected[k].append((fdir.name, top_indices[:k]))
        del x
    all_diag_df = pd.concat(all_diag, ignore_index=True).sort_values("score_abs_rankicir", ascending=False)
    all_diag_df.to_csv(out_dir / "factor_diagnostics_all.csv", index=False)

    experiments = []
    for k in top_ks:
        parts_train = [x_num[train]]
        parts_val = [x_num[val]]
        parts_test = [x_num[test]]
        for name, _ in selected[k]:
            parts_train.append(cache_selected[name]["train"][:, :k])
            parts_val.append(cache_selected[name]["val"][:, :k])
            parts_test.append(cache_selected[name]["test"][:, :k])
        x_train = np.concatenate(parts_train, axis=1)
        x_val = np.concatenate(parts_val, axis=1)
        x_test = np.concatenate(parts_test, axis=1)
        raw_models = fit_predict_models(x_train, y[train], x_val, y[val], x_test)
        y_z_train = cross_sectional_z(meta_train, y[train])
        rank_models = fit_predict_models(x_train, y_z_train, x_val, cross_sectional_z(meta_val, y[val]), x_test)
        for model_name, (val_pred, test_pred) in raw_models.items():
            mode, (cal_val, cal_test) = calibrate_by_val(y[train], y[val], val_pred, test_pred)
            threshold = float(np.median(y[train]))
            try:
                auc = float(roc_auc_score((y[test] > threshold).astype(int), cal_test))
            except Exception:
                auc = float("nan")
            bt, daily = backtest_metrics(meta_test, y[test], cal_test)
            daily.to_csv(out_dir / f"backtest_top{k}_{model_name}_{mode}.csv", index=False)
            item = {
                "stage": "top_factor_raw_return",
                "top_k_per_feature_dir": k,
                "model": f"{model_name}+{mode}",
                "return": reg_metrics(y[test], cal_test),
                "AUC": auc,
                "ranking": rank_metrics(meta_test, y[test], cal_test),
                "backtest": bt,
            }
            experiments.append(item)
            print(json.dumps(item, ensure_ascii=False, indent=2))
        for model_name, (_, test_score) in rank_models.items():
            bt, daily = backtest_metrics(meta_test, y[test], test_score)
            daily.to_csv(out_dir / f"backtest_top{k}_{model_name}_xsecz.csv", index=False)
            item = {
                "stage": "top_factor_xsec_z",
                "top_k_per_feature_dir": k,
                "model": f"{model_name}_XSecZ",
                "ranking": rank_metrics(meta_test, y[test], test_score),
                "backtest": bt,
            }
            experiments.append(item)
            print(json.dumps(item, ensure_ascii=False, indent=2))

    best_return = min([e for e in experiments if "return" in e], key=lambda e: e["return"]["MSE"])
    best_rankic = max(experiments, key=lambda e: e["ranking"]["RankIC"])
    best_sharpe = max(experiments, key=lambda e: e["backtest"]["long_short"]["sharpe"])
    summary = {
        "samples": {"total": int(len(y)), "train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "top_factor_table": str(out_dir / "factor_diagnostics_all.csv"),
        "top_20_factors": all_diag_df.head(20).to_dict(orient="records"),
        "experiments": experiments,
        "best_return_mse": best_return,
        "best_rankic": best_rankic,
        "best_long_short_sharpe": best_sharpe,
    }
    (out_dir / "factor_selection_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", out_dir / "factor_selection_backtest_summary.json")


if __name__ == "__main__":
    main()
