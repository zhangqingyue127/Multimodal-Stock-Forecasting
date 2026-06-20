import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_feature_dir(path: Path, max_stocks: int | None):
    files = sorted(path.glob("*.npz"))
    if max_stocks is not None:
        files = files[:max_stocks]
    if not files:
        raise FileNotFoundError(f"No feature files found in {path}")

    features = []
    labels = []
    stocks = []
    for file in tqdm(files, desc=f"Loading {path.name}"):
        data = np.load(file, allow_pickle=False)
        feat = np.asarray(data["feature"], dtype=np.float32)
        label = np.asarray(data["label"], dtype=np.float32)
        n = min(len(feat), len(label))
        features.append(feat[:n])
        labels.append(label[:n])
        stocks.extend([file.stem.replace("_features", "")] * n)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), np.asarray(stocks)


def sample_data(x, y, max_samples, seed):
    if max_samples is None or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_samples, replace=False)
    idx.sort()
    return x[idx], y[idx]


def evaluate(x, y, max_samples, seed):
    x, y = sample_data(x, y, max_samples, seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = make_pipeline(
        StandardScaler(with_mean=True),
        Ridge(alpha=1.0, random_state=seed),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    directional_accuracy = float((np.sign(pred) == np.sign(y_test)).mean())
    return {
        "samples_used": int(len(y)),
        "feature_dim": int(x.shape[1]),
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
    }


def winner(vit, mae):
    if vit["mse"] < mae["mse"]:
        return "vit"
    if mae["mse"] < vit["mse"]:
        return "mae"
    return "tie"


def main():
    parser = argparse.ArgumentParser(description="Compare ViT vs MAE feature quality for mixed and separate figures.")
    parser.add_argument("--feature-root", default="/root/autodl-tmp/aligned_features")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="/root/autodl-tmp/aligned_features/comparison_results.json")
    args = parser.parse_args()

    root = Path(args.feature_root)
    results = {}
    for source in ("mixed", "separate"):
        results[source] = {}
        for encoder in ("vit", "mae"):
            feat_dir = root / f"{source}_{encoder}"
            x, y, _ = load_feature_dir(feat_dir, args.max_stocks)
            results[source][encoder] = evaluate(x, y, args.max_samples, args.seed)
        results[source]["winner_by_mse"] = winner(results[source]["vit"], results[source]["mae"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
