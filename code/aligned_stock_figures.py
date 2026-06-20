import argparse
import gc
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


DATE_COL = "Trddt"
STOCK_COL = "Stkcd"
OPEN_COL = "Opnprc"
HIGH_COL = "Hiprc"
LOW_COL = "Loprc"
CLOSE_COL = "Clsprc"
VOL_COL = "Dnshrtrd"

COMPONENT_NAMES = ("candle", "volume", "ma")
MA_WINDOWS = (5, 10, 20)


def find_csv(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    candidates = [
        repo_root / "data",
        Path("/root/autodl-tmp"),
        repo_root,
    ]
    for base in candidates:
        if base.exists():
            files = sorted(base.glob("*.csv"))
            if files:
                return files[0]
    raise FileNotFoundError("No CSV file found. Pass --csv explicitly.")


def stock_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def normalize(values):
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    scale = max(hi - lo, 1e-6)
    return np.nan_to_num((arr - lo) / scale, nan=0.0, posinf=1.0, neginf=0.0)


def blank(size: int) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def draw_line(img, values, color, y0=4, y1=None, width=1):
    values = np.asarray(values, dtype=np.float32)
    size = img.shape[0]
    y1 = size - 5 if y1 is None else y1
    h = max(y1 - y0, 1)
    xs = np.linspace(0, size - 1, len(values)).astype(np.int32)
    ys = (y0 + (1.0 - values) * h).astype(np.int32)
    for i in range(1, len(xs)):
        x_a, x_b = xs[i - 1], xs[i]
        y_a, y_b = ys[i - 1], ys[i]
        steps = max(abs(x_b - x_a), abs(y_b - y_a), 1)
        x_line = np.linspace(x_a, x_b, steps + 1).astype(np.int32)
        y_line = np.linspace(y_a, y_b, steps + 1).astype(np.int32)
        for dx in range(-width, width + 1):
            yy = np.clip(y_line + dx, 0, size - 1)
            img[yy, x_line] = color
    return img


def draw_candles(img, o, h, l, c, y0=4, y1=None):
    size = img.shape[0]
    y1 = size - 5 if y1 is None else y1
    n = len(o)
    step = size / n
    body_w = max(int(step * 0.55), 1)
    height = max(y1 - y0, 1)
    for i in range(n):
        up = c[i] >= o[i]
        color = np.array([210, 30, 30] if up else [20, 150, 60], dtype=np.uint8)
        x = int(i * step + step / 2)
        yh = int(y0 + (1 - h[i]) * height)
        yl = int(y0 + (1 - l[i]) * height)
        yo = int(y0 + (1 - o[i]) * height)
        yc = int(y0 + (1 - c[i]) * height)
        xa = max(x - 1, 0)
        xb = min(x + 2, size)
        img[min(yh, yl) : max(yh, yl) + 1, xa:xb] = color
        ya, yb = sorted((yo, yc))
        if yb - ya < 2:
            yb = min(ya + 2, size - 1)
        x1 = max(x - body_w // 2, 0)
        x2 = min(x + body_w // 2 + 1, size)
        img[ya : yb + 1, x1:x2] = color
    return img


def draw_volume(img, volume, up_mask, y0=4, y1=None):
    size = img.shape[0]
    y1 = size - 5 if y1 is None else y1
    n = len(volume)
    step = size / n
    bar_w = max(int(step * 0.55), 1)
    height = max(y1 - y0, 1)
    for i, v in enumerate(volume):
        color = np.array([210, 30, 30] if up_mask[i] else [20, 150, 60], dtype=np.uint8)
        x = int(i * step + step / 2)
        x1 = max(x - bar_w // 2, 0)
        x2 = min(x + bar_w // 2 + 1, size)
        top = int(y1 - v * height)
        img[top:y1, x1:x2] = color
    return img


def moving_average(values, window):
    s = pd.Series(values, dtype="float32")
    return s.rolling(window, min_periods=1).mean().to_numpy(dtype=np.float32)


def render_components(prices, size):
    o_raw = np.asarray(prices["open"], dtype=np.float32)
    h_raw = np.asarray(prices["high"], dtype=np.float32)
    l_raw = np.asarray(prices["low"], dtype=np.float32)
    c_raw = np.asarray(prices["close"], dtype=np.float32)
    v = normalize(prices["volume"])

    price_min = float(np.nanmin(l_raw))
    price_max = float(np.nanmax(h_raw))
    price_scale = max(price_max - price_min, 1e-6)
    o = (o_raw - price_min) / price_scale
    h = (h_raw - price_min) / price_scale
    l = (l_raw - price_min) / price_scale
    c = (c_raw - price_min) / price_scale
    up_mask = c_raw >= o_raw

    candle = draw_candles(blank(size), o, h, l, c)
    volume = draw_volume(blank(size), v, up_mask)

    ma_img = blank(size)
    ma_colors = ([40, 95, 210], [230, 150, 20], [130, 60, 180])
    for ma, color in zip(MA_WINDOWS, ma_colors):
        ma_values = (moving_average(c_raw, ma) - price_min) / price_scale
        draw_line(ma_img, np.clip(ma_values, 0, 1), np.array(color, dtype=np.uint8), width=1)

    mixed = blank(size)
    volume_top = int(size * 0.78)
    draw_candles(mixed, o, h, l, c, y0=4, y1=volume_top - 3)
    for ma, color in zip(MA_WINDOWS, ma_colors):
        ma_values = (moving_average(c_raw, ma) - price_min) / price_scale
        draw_line(mixed, np.clip(ma_values, 0, 1), np.array(color, dtype=np.uint8), y0=4, y1=volume_top - 3, width=1)
    draw_volume(mixed, v, up_mask, y0=volume_top + 3, y1=size - 4)

    return {"mixed": mixed, "candle": candle, "volume": volume, "ma": ma_img}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        dtype={
            STOCK_COL: str,
            OPEN_COL: np.float32,
            HIGH_COL: np.float32,
            LOW_COL: np.float32,
            CLOSE_COL: np.float32,
            VOL_COL: np.float32,
        },
        usecols=[DATE_COL, STOCK_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL],
        low_memory=False,
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df[STOCK_COL] = df[STOCK_COL].map(stock_id)
    return df.sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)


def future_returns(close, horizon):
    close = np.asarray(close, dtype=np.float32)
    out = np.full(len(close), np.nan, dtype=np.float32)
    for i in range(len(close) - horizon):
        out[i] = close[i + horizon] / close[i] - 1.0
    return out


def process_stock(sid, group, mixed_dir, separate_dir, window, horizon, size, overwrite):
    mixed_path = mixed_dir / f"{sid}.npz"
    separate_path = separate_dir / f"{sid}.npz"
    if not overwrite and mixed_path.exists() and separate_path.exists():
        return sid, 0, "skipped"

    group = group.sort_values(DATE_COL).reset_index(drop=True)
    labels_by_end = future_returns(group[CLOSE_COL].to_numpy(dtype=np.float32), horizon)
    valid_ends = [end for end in range(window - 1, len(group) - horizon) if np.isfinite(labels_by_end[end])]
    if not valid_ends:
        return sid, 0, "empty"

    arrays = {name: [] for name in ("mixed", *COMPONENT_NAMES)}
    labels = []
    end_dates = []
    for end in valid_ends:
        win = group.iloc[end - window + 1 : end + 1]
        prices = {
            "open": win[OPEN_COL].to_numpy(dtype=np.float32),
            "high": win[HIGH_COL].to_numpy(dtype=np.float32),
            "low": win[LOW_COL].to_numpy(dtype=np.float32),
            "close": win[CLOSE_COL].to_numpy(dtype=np.float32),
            "volume": win[VOL_COL].to_numpy(dtype=np.float32),
        }
        rendered = render_components(prices, size)
        for name, image in rendered.items():
            arrays[name].append(image)
        labels.append(labels_by_end[end])
        end_dates.append(group.loc[end, DATE_COL].strftime("%Y%m%d"))

    label_arr = np.asarray(labels, dtype=np.float32)
    date_arr = np.asarray(end_dates)
    component_arr = np.asarray(COMPONENT_NAMES)

    np.savez_compressed(
        mixed_path,
        mixed=np.asarray(arrays["mixed"], dtype=np.uint8),
        label=label_arr,
        end_date=date_arr,
        component_names=component_arr,
    )
    np.savez_compressed(
        separate_path,
        candle=np.asarray(arrays["candle"], dtype=np.uint8),
        volume=np.asarray(arrays["volume"], dtype=np.uint8),
        ma=np.asarray(arrays["ma"], dtype=np.uint8),
        label=label_arr,
        end_date=date_arr,
        component_names=component_arr,
    )
    return sid, len(label_arr), "written"


def main():
    parser = argparse.ArgumentParser(description="Generate aligned mixed and separate stock chart images.")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out-root", default="/root/autodl-tmp/aligned_figures")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = find_csv(repo_root, args.csv)
    out_root = Path(args.out_root)
    mixed_dir = out_root / "mixed"
    separate_dir = out_root / "separate"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    separate_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {csv_path}")
    print(f"Mixed output: {mixed_dir}")
    print(f"Separate output: {separate_dir}")
    print(f"Components: {', '.join(COMPONENT_NAMES)}")

    df = load_data(csv_path)
    groups = list(df.groupby(STOCK_COL, sort=True))
    if args.max_stocks is not None:
        groups = groups[: args.max_stocks]
    del df
    gc.collect()

    summary = {"written": 0, "skipped": 0, "empty": 0, "samples": 0}
    if args.workers <= 1:
        for sid, group in tqdm(groups, desc="Generating aligned figures"):
            _, count, status = process_stock(
                sid,
                group,
                mixed_dir,
                separate_dir,
                args.window,
                args.horizon,
                args.image_size,
                args.overwrite,
            )
            summary[status] += 1
            summary["samples"] += count
            gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_stock,
                    sid,
                    group,
                    mixed_dir,
                    separate_dir,
                    args.window,
                    args.horizon,
                    args.image_size,
                    args.overwrite,
                )
                for sid, group in groups
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating aligned figures"):
                _, count, status = future.result()
                summary[status] += 1
                summary["samples"] += count

    manifest = {
        "csv": str(csv_path),
        "window": args.window,
        "horizon": args.horizon,
        "image_size": args.image_size,
        "component_names": list(COMPONENT_NAMES),
        "mixed_rule": "mixed = candle + moving averages + volume",
        "separate_rule": "one separate image per mixed component: candle, volume, ma",
        "summary": summary,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
