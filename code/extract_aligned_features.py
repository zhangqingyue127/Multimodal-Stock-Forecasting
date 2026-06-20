import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm


MODEL_NAMES = {
    "vit": "vit_base_patch16_224",
    "mae": "vit_base_patch16_224.mae",
}
COMPONENT_NAMES = ("candle", "volume", "ma")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(encoder: str, device: torch.device, checkpoint: str | None):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import timm

    model_name = MODEL_NAMES[encoder]
    if checkpoint:
        model = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=224)
        state = load_file(checkpoint) if checkpoint.endswith(".safetensors") else torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224)
    model.to(device)
    model.eval()
    return model


def preprocess(images: np.ndarray, device: torch.device):
    x = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = MEAN.to(device)
    std = STD.to(device)
    return (x - mean) / std


@torch.inference_mode()
def extract_batches(model, images: np.ndarray, batch_size: int, device: torch.device):
    feats = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start : start + batch_size], device)
        feat = model(batch)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feats.append(feat.detach().cpu().numpy().astype(np.float32))
        del batch, feat
    return np.concatenate(feats, axis=0)


def read_components(data):
    if "component_names" in data:
        names = [str(x) for x in data["component_names"].tolist()]
    else:
        names = list(COMPONENT_NAMES)
    return [name for name in names if name in data]


def process_file(path: Path, out_path: Path, source: str, model, batch_size: int, device: torch.device, aggregate: str):
    data = np.load(path, allow_pickle=False)
    labels = np.asarray(data["label"], dtype=np.float32)
    dates = data["end_date"] if "end_date" in data else np.arange(len(labels)).astype(str)

    if source == "mixed":
        feature = extract_batches(model, np.asarray(data["mixed"], dtype=np.uint8), batch_size, device)
        component_names = np.asarray(["mixed"])
    else:
        names = read_components(data)
        component_feats = []
        for name in names:
            feats = extract_batches(model, np.asarray(data[name], dtype=np.uint8), batch_size, device)
            component_feats.append(feats)
            gc.collect()
        stacked = np.stack(component_feats, axis=1)
        if aggregate == "concat":
            feature = stacked.reshape(stacked.shape[0], -1)
        elif aggregate == "mean":
            feature = stacked.mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")
        component_names = np.asarray(names)

    np.savez_compressed(
        out_path,
        feature=feature.astype(np.float32),
        label=labels,
        end_date=dates,
        stock_id=path.stem,
        source=source,
        encoder=out_path.parent.name,
        component_names=component_names,
        aggregate=aggregate if source == "separate" else "single",
    )


def main():
    parser = argparse.ArgumentParser(description="Extract ViT/MAE features from aligned stock figures.")
    parser.add_argument("--fig-root", default="/root/autodl-tmp/aligned_figures")
    parser.add_argument("--out-root", default="/root/autodl-tmp/aligned_features")
    parser.add_argument("--source", choices=["mixed", "separate"], required=True)
    parser.add_argument("--encoder", choices=["vit", "mae"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--aggregate", choices=["concat", "mean"], default="concat")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Source: {args.source} | Encoder: {args.encoder}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    fig_dir = Path(args.fig_root) / args.source
    out_dir = Path(args.out_root) / f"{args.source}_{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(fig_dir.glob("*.npz"))
    if args.max_stocks is not None:
        files = files[: args.max_stocks]
    if not files:
        raise FileNotFoundError(f"No figure npz files found in {fig_dir}")

    model = load_model(args.encoder, device, args.checkpoint)
    written = skipped = 0
    for path in tqdm(files, desc=f"Extracting {args.source}-{args.encoder}"):
        out_path = out_dir / f"{path.stem}_features.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        process_file(path, out_path, args.source, model, args.batch_size, device, args.aggregate)
        written += 1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Done. Written={written}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
