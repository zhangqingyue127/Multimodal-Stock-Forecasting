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
    recall_score,
    roc_auc_score,
    r2_score,
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
    for file in tqdm(files, desc=f"读取NPZ {path.name}"):
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
    for sid, g in tqdm(df.groupby(STOCK_COL, sort=False), desc="构建数值时序特征"):
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
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    return train, val, test


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", default="/root/autodl-tmp/aligned_features/mixed_mae")
    ap.add_argument("--out", default="/root/autodl-tmp/reference_like_improved_results.txt")
    ap.add_argument("--pca-components", type=int, default=32)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = find_csv(repo_root)
    print(f"📂 加载特征: {args.feature_dir}")
    x_img, y_ret, meta = load_feature_dir(Path(args.feature_dir))
    print(f"✅ 原始样本数: {len(y_ret)}")

    num, num_cols = build_numeric(csv_path)
    data_meta = meta.merge(num, on=["stock_id", "end_date"], how="left")
    x_num = data_meta[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    close_now = data_meta["close_now"].to_numpy(np.float32)
    future_close = close_now * (1.0 + y_ret)

    train, val, test = split_70_15_15(meta)
    print(f"时间切分 -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    threshold = float(np.median(y_ret[train]))
    print(f"分类阈值(训练集收益率中位数): {threshold:.6f} | 测试正类比例: {float((y_ret[test] > threshold).mean()):.2%}")

    # PCA is fit on train only, then used as compact image representation.
    pca = PCA(n_components=args.pca_components, random_state=42)
    scaler_img = StandardScaler()
    x_img_train_scaled = scaler_img.fit_transform(x_img[train])
    pca.fit(x_img_train_scaled)
    def img_pca(idx):
        return pca.transform(scaler_img.transform(x_img[idx])).astype(np.float32)

    x_train = np.concatenate([x_num[train], img_pca(train)], axis=1)
    x_val = np.concatenate([x_num[val], img_pca(val)], axis=1)
    x_test = np.concatenate([x_num[test], img_pca(test)], axis=1)

    # Try a small validation tournament and select by val MSE.
    candidates = {}
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    ridge.fit(x_train, y_ret[train])
    candidates["RidgeFusion"] = (ridge, ridge.predict(x_val))

    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=42,
    )
    hgb.fit(x_train, y_ret[train])
    candidates["HGBFusion"] = (hgb, hgb.predict(x_val))

    best_name, (best_model, best_val_pred) = min(
        candidates.items(), key=lambda item: mean_squared_error(y_ret[val], item[1][1])
    )
    raw_val_mse = mean_squared_error(y_ret[val], best_val_pred)
    print(f"选择模型: {best_name} | Val MSE: {raw_val_mse:.6f}")

    raw_test_score = best_model.predict(x_test)
    train_mean = float(np.mean(y_ret[train]))
    val_pred = np.asarray(best_val_pred, dtype=np.float64)
    y_val = y_ret[val].astype(np.float64)
    pred_var = float(np.var(val_pred))
    if pred_var > 1e-12:
        slope = float(np.cov(val_pred, y_val, bias=True)[0, 1] / pred_var)
        intercept = float(np.mean(y_val) - slope * np.mean(val_pred))
    else:
        slope, intercept = 0.0, train_mean
    val_options = {
        "raw": val_pred,
        "calibrated": intercept + slope * val_pred,
        "train_mean": np.full_like(y_val, train_mean),
    }
    score_mode, score_val_pred = min(
        val_options.items(), key=lambda item: mean_squared_error(y_val, item[1])
    )
    if score_mode == "calibrated":
        test_score = intercept + slope * raw_test_score
    elif score_mode == "train_mean":
        test_score = np.full_like(raw_test_score, train_mean, dtype=np.float64)
    else:
        test_score = raw_test_score
    print(f"收益率预测校准: {score_mode} | Val MSE: {mean_squared_error(y_val, score_val_pred):.6f}")
    close_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    close_model.fit(x_num[train], future_close[train])
    close_pred = close_model.predict(x_num[test])

    clf = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=42,
    )
    clf.fit(x_train, (y_ret[train] > threshold).astype(int))
    cls_score = clf.predict_proba(x_test)[:, 1]
    cls_label = (y_ret[test] > threshold).astype(int)
    cls_pred = (cls_score >= 0.5).astype(int)
    cls = {
        "Acc": float(accuracy_score(cls_label, cls_pred)),
        "Prec": float(precision_score(cls_label, cls_pred, zero_division=0)),
        "Rec": float(recall_score(cls_label, cls_pred, zero_division=0)),
        "F1": float(f1_score(cls_label, cls_pred, zero_division=0)),
        "AUC": float(roc_auc_score(cls_label, cls_score)),
    }

    results = {
        "model": f"{best_name}+{score_mode}",
        "feature_dir": args.feature_dir,
        "samples": {"total": int(len(y_ret)), "train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "close": reg_metrics(future_close[test], close_pred),
        "return": reg_metrics(y_ret[test], test_score),
        "classification": cls,
        "ranking": rank_metrics(meta.iloc[test].reset_index(drop=True), y_ret[test], test_score),
        "reference_file_return_mse": 0.0031112903,
        "reference_file_auc": 0.5195918199,
        "reference_file_rankic": -0.0171990457,
    }
    lines = []
    lines.append(f"📂 加载特征: {args.feature_dir}")
    lines.append(f"✅ 原始样本数: {len(y_ret)}")
    lines.append(f"时间切分 -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    lines.append(f"分类阈值(训练集收益率中位数): {threshold:.6f}")
    lines.append(f"🎯 准备训练 {best_name} (Numeric + Image PCA{args.pca_components}) 模型")
    lines.append("")
    lines.append("📊 测试集结果")
    lines.append(f"1️⃣ 收盘价: {results['close']}")
    lines.append(f"2️⃣ 收益率: {results['return']}")
    lines.append(f"3️⃣ 分类: {results['classification']}")
    lines.append(f"4️⃣ 排序: {results['ranking']}")
    lines.append("")
    lines.append("对照文件关键指标:")
    lines.append("收益率 MSE=0.0031112903 | AUC=0.5195918199 | RankIC=-0.0171990457")
    lines.append("本次结果 JSON:")
    lines.append(json.dumps(results, ensure_ascii=False, indent=2))
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")
    Path(str(args.out) + ".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
