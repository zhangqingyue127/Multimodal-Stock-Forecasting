"""三层评估：特征诊断、冻结特征简单探针，以及数值/图像/融合消融实验。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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


def find_csv(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    for base in (repo_root / "data", Path("/root/autodl-tmp"), repo_root):
        if base.exists():
            files = sorted(base.glob("*.csv"))
            if files:
                return files[0]
    raise FileNotFoundError("No CSV file found.")


def load_feature_dir(path: Path, max_stocks=None):
    files = sorted(path.glob("*.npz"))
    if max_stocks is not None:
        files = files[:max_stocks]
    if not files:
        raise FileNotFoundError(path)

    features, labels, stock_ids, dates = [], [], [], []
    for file in tqdm(files, desc=f"Loading {path.name}"):
        data = np.load(file, allow_pickle=False)
        feat = np.asarray(data["feature"], dtype=np.float32)
        label = np.asarray(data["label"], dtype=np.float32)
        n = min(len(feat), len(label))
        features.append(feat[:n])
        labels.append(label[:n])
        stock = file.stem.replace("_features", "")
        stock_ids.extend([stock] * n)
        if "end_date" in data:
            dates.extend([str(x) for x in data["end_date"][:n]])
        else:
            dates.extend([str(i) for i in range(n)])
    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = pd.DataFrame({"stock_id": stock_ids, "end_date": dates})
    return x, y, meta


def stable_sample(n, max_samples, seed):
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    return idx


def pearson_ic_by_feature(x, y):
    y = y.astype(np.float64)
    yc = y - y.mean()
    xc = x.astype(np.float64) - x.mean(axis=0)
    denom = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
    corr = np.divide((xc * yc[:, None]).sum(axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 0)
    return corr


def rank_ic_for_dims(x, y, dims):
    yr = rankdata(y)
    out = []
    for dim in dims:
        xr = rankdata(x[:, dim])
        c = np.corrcoef(xr, yr)[0, 1]
        out.append(float(c) if np.isfinite(c) else 0.0)
    return np.asarray(out, dtype=np.float32)


def layer1_diagnostics(name, x, y, out_dir, seed, max_diag_samples=20000):
    """第一层只检查特征分布、相似度、IC与PCA结构，不证明预测收益。"""
    idx = stable_sample(len(y), max_diag_samples, seed)
    xs, ys = x[idx], y[idx]
    rng = np.random.default_rng(seed)
    sim_idx = rng.choice(len(ys), size=min(2000, len(ys)), replace=False)
    z = xs[sim_idx].astype(np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    sim = z @ z.T
    mask = ~np.eye(len(sim_idx), dtype=bool)

    ic = pearson_ic_by_feature(xs, ys)
    top_dims = np.argsort(np.abs(ic))[-min(50, len(ic)) :]
    ric = rank_ic_for_dims(xs, ys, top_dims)

    pca = PCA(n_components=2, random_state=seed)
    pc = pca.fit_transform(StandardScaler().fit_transform(xs[: min(10000, len(xs))]))
    y_plot = ys[: len(pc)]
    q = pd.qcut(y_plot, q=5, labels=False, duplicates="drop")
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(pc[:, 0], pc[:, 1], c=q, s=5, cmap="viridis", alpha=0.55)
    plt.colorbar(sc, label="future return quantile")
    plt.title(f"PCA diagnostic: {name}")
    plt.tight_layout()
    pca_path = out_dir / f"pca_{name}.png"
    plt.savefig(pca_path, dpi=160)
    plt.close()

    return {
        "samples_used": int(len(ys)),
        "feature_dim": int(x.shape[1]),
        "feature_variance": float(np.var(xs)),
        "mean_pairwise_cosine_similarity": float(sim[mask].mean()),
        "mean_abs_ic": float(np.mean(np.abs(ic))),
        "max_abs_ic": float(np.max(np.abs(ic))),
        "top50_mean_abs_rank_ic": float(np.mean(np.abs(ric))),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_plot": str(pca_path),
        "interpretation_limit": "diagnostic only; these metrics do not prove profitable stock prediction.",
    }


def regression_metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "directional_accuracy_from_regression": float((np.sign(pred) == np.sign(y_true)).mean()),
    }


def chronological_split(meta, test_size=0.2):
    dates = pd.to_datetime(meta["end_date"], format="%Y%m%d", errors="coerce")
    valid_dates = np.sort(dates.dropna().unique())
    if len(valid_dates) < 5:
        return None
    cutoff = valid_dates[int(len(valid_dates) * (1 - test_size))]
    train = dates < cutoff
    test = dates >= cutoff
    if train.sum() == 0 or test.sum() == 0:
        return None
    return train.to_numpy(), test.to_numpy(), str(pd.Timestamp(cutoff).date())


def fit_regressor(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    return model.predict(x_test)


def layer2_simple_models(x, y, meta, seed, max_samples, max_mlp_train=12000):
    """第二层冻结视觉特征，用简单模型检验图像表征是否含预测信号。"""
    idx = stable_sample(len(y), max_samples, seed)
    x, y, meta = x[idx], y[idx], meta.iloc[idx].reset_index(drop=True)

    split = chronological_split(meta)
    split_name = "chronological"
    if split is None:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        train_mask = np.zeros(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        train_mask[tr] = True
        test_mask[te] = True
        cutoff = None
        split_name = "random"
    else:
        train_mask, test_mask, cutoff = split

    x_train, x_test = x[train_mask], x[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    results = {
        "samples_used": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "split": split_name,
        "cutoff_date": cutoff,
        "models": {},
    }

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=1e-3, l1_ratio=0.1, max_iter=1000, random_state=seed)),
    }
    for model_name, model in models.items():
        pred = fit_regressor(model, x_train, x_test, y_train)
        results["models"][model_name] = regression_metrics(y_test, pred)

    rng = np.random.default_rng(seed)
    mlp_train_idx = np.arange(len(y_train))
    if len(mlp_train_idx) > max_mlp_train:
        mlp_train_idx = rng.choice(mlp_train_idx, size=max_mlp_train, replace=False)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=40,
            early_stopping=True,
            random_state=seed,
        ),
    )
    mlp.fit(x_train[mlp_train_idx], y_train[mlp_train_idx])
    mlp_pred = mlp.predict(x_test)
    results["models"]["mlp_regressor_subsample"] = {
        **regression_metrics(y_test, mlp_pred),
        "train_samples_used": int(len(mlp_train_idx)),
        "note": "MLP is run as a small nonlinear probe to avoid making the evaluation dominated by probe training time.",
    }

    y_train_cls = (y_train > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.3, random_state=seed))
    clf.fit(x_train, y_train_cls)
    cls_pred = clf.predict(x_test)
    results["direction_classifier"] = {
        "model": "logistic_regression",
        "accuracy": float(accuracy_score(y_test_cls, cls_pred)),
        "positive_rate_test": float(y_test_cls.mean()),
    }
    return results


def build_numeric_features(csv_path: Path):
    df = pd.read_csv(
        csv_path,
        dtype={STOCK_COL: str},
        usecols=[STOCK_COL, DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL],
        low_memory=False,
    )
    df[STOCK_COL] = df[STOCK_COL].map(stock_id)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)
    parts = []
    for sid, g in tqdm(df.groupby(STOCK_COL, sort=False), desc="Building numeric features"):
        g = g.copy()
        close = g[CLOSE_COL].astype(float)
        open_ = g[OPEN_COL].astype(float)
        high = g[HIGH_COL].astype(float)
        low = g[LOW_COL].astype(float)
        vol = g[VOL_COL].astype(float)
        ret1 = close.pct_change()
        out = pd.DataFrame({
            "stock_id": sid,
            "end_date": g[DATE_COL].dt.strftime("%Y%m%d"),
            "ret_1": ret1,
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "volatility_20": ret1.rolling(20, min_periods=5).std(),
            "intraday_return": close / open_ - 1.0,
            "range_ratio": high / low - 1.0,
            "close_to_ma5": close / close.rolling(5, min_periods=2).mean() - 1.0,
            "close_to_ma20": close / close.rolling(20, min_periods=5).mean() - 1.0,
            "volume_chg_5": vol / vol.rolling(5, min_periods=2).mean() - 1.0,
            "volume_z20": (vol - vol.rolling(20, min_periods=5).mean()) / vol.rolling(20, min_periods=5).std(),
        })
        parts.append(out)
    num = pd.concat(parts, ignore_index=True)
    feature_cols = [c for c in num.columns if c not in ("stock_id", "end_date")]
    num[feature_cols] = num[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return num, feature_cols


def layer3_ablation(feature_sets, numeric_df, numeric_cols, seed, max_samples):
    """第三层比较数值、图像和两者融合，判断图像模态是否提供增量。"""
    results = {}
    for name, (x_img, y, meta) in feature_sets.items():
        idx = stable_sample(len(y), max_samples, seed)
        x_img_s = x_img[idx]
        y_s = y[idx]
        meta_s = meta.iloc[idx].reset_index(drop=True)
        joined = meta_s.merge(numeric_df, on=["stock_id", "end_date"], how="left")
        x_num = joined[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

        split = chronological_split(meta_s)
        if split is None:
            tr, te = train_test_split(np.arange(len(y_s)), test_size=0.2, random_state=seed)
            train_mask = np.zeros(len(y_s), dtype=bool)
            test_mask = np.zeros(len(y_s), dtype=bool)
            train_mask[tr] = True
            test_mask[te] = True
            cutoff = None
        else:
            train_mask, test_mask, cutoff = split

        # 三个输入组使用相同模型和切分，差异只来自模态组合。
        experiments = {
            "numeric_only": x_num,
            "image_only": x_img_s,
            "numeric_plus_image": np.concatenate([x_num, x_img_s], axis=1),
        }
        results[name] = {
            "samples_used": int(len(y_s)),
            "cutoff_date": cutoff,
            "note": "No text modality found in this repository, so text ablations are not run.",
            "experiments": {},
        }
        for exp_name, x_exp in experiments.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_exp[train_mask], y_s[train_mask])
            pred = model.predict(x_exp[test_mask])
            results[name]["experiments"][exp_name] = regression_metrics(y_s[test_mask], pred)
    return results


def main():
    """主流程：加载四类特征 -> 依次执行三层评估 -> 汇总JSON结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default="/root/autodl-tmp/aligned_features")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/three_layer_eval")
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stocks", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = find_csv(repo_root, args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 同时读取mixed/separate与ViT/MAE四类视觉表征。
    feature_root = Path(args.feature_root)
    feature_sets = {}
    for name in ("mixed_vit", "mixed_mae", "separate_vit", "separate_mae"):
        x, y, meta = load_feature_dir(feature_root / name, args.max_stocks)
        feature_sets[name] = (x, y, meta)

    # 2. 第一层：方差、余弦相似度、IC和PCA等诊断性指标。
    layer1 = {}
    for name, (x, y, _) in feature_sets.items():
        layer1[name] = layer1_diagnostics(name, x, y, out_dir, args.seed)

    # 3. 第二层：Ridge、ElasticNet和小型MLP等冻结特征探针。
    layer2 = {}
    for name, (x, y, meta) in feature_sets.items():
        layer2[name] = layer2_simple_models(x, y, meta, args.seed, args.max_samples)

    # 4. 第三层：数值单模态、图像单模态和多模态融合消融。
    numeric_df, numeric_cols = build_numeric_features(csv_path)
    layer3 = layer3_ablation(feature_sets, numeric_df, numeric_cols, args.seed, args.max_samples)

    results = {
        "scope": {
            "csv": str(csv_path),
            "feature_root": str(feature_root),
            "max_samples_for_modeling": args.max_samples,
            "layers": {
                "layer1": "diagnostic feature statistics only",
                "layer2": "frozen image encoder + simple probe models",
                "layer3": "downstream ablation with numeric time-series features and image features; no text modality available",
            },
        },
        "layer1_diagnostics": layer1,
        "layer2_linear_probe_simple_models": layer2,
        "layer3_downstream_ablation": layer3,
        "allowed_conclusion": "Use layer 1 only for feature diagnostics. Use layer 2/3 to discuss predictive signal. Only layer 3 ablations can support claims about image modality adding value over numeric baselines.",
    }
    out_path = out_dir / "three_layer_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
