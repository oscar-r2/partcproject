
######################### Script to convert a PyTorch to ONNX #########################
# Luciano Ost
# Export a PyTorch checkpoint (.pt / .pth) to ONNX.

# Works with:
#    - A checkpoint dict that contains {"model_state": state_dict, "classes": [...]}
#    - A raw state_dict saved directly via torch.save(model.state_dict(), ...)

# Default model architecture matches the synthetic vehicle project:
# 3*(Conv+ReLU+Pool) + Flatten + Linear + ReLU + Dropout + Linear
# I did not test it with other models.
#######################################################################################

import argparse
from pathlib import Path
import torch
import torch.nn as nn


class SimpleCNN_Dropout(nn.Module):
    def __init__(self, num_classes=4, input_size=64):
        super().__init__()
        fm = input_size // 8  # after 3 pools
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * fm * fm, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SimpleCNN_NoDropout(nn.Module):
    def __init__(self, num_classes=4, input_size=64):
        super().__init__()
        fm = input_size // 8
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * fm * fm, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("classes")
    return ckpt, None


def infer_num_classes(state, classes, override):
    if override is not None:
        return override
    if classes is not None:
        return len(classes)
    for key in ("classifier.4.bias", "classifier.3.bias"):
        if key in state:
            return int(state[key].numel())
    raise ValueError("Could not infer num_classes. Provide --num_classes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt/.pth checkpoint")
    ap.add_argument("--onnx_out", default="model.onnx", help="Output ONNX filename")
    ap.add_argument("--input_size", type=int, default=64, help="Input image size (H=W)")
    ap.add_argument("--num_classes", type=int, default=None, help="Override number of classes")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset")
    ap.add_argument("--dynamic_batch", action="store_true", help="Export with dynamic batch axis")
    args = ap.parse_args()

    state, classes = load_checkpoint(Path(args.ckpt))
    num_classes = infer_num_classes(state, classes, args.num_classes)

    use_dropout = ("classifier.4.weight" in state) or ("classifier.4.bias" in state)
    Model = SimpleCNN_Dropout if use_dropout else SimpleCNN_NoDropout
    model = Model(num_classes=num_classes, input_size=args.input_size)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        args.onnx_out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX to: {args.onnx_out}")
    print(f"Model: {'Dropout' if use_dropout else 'NoDropout'} | classes={num_classes} | input={args.input_size}x{args.input_size}")


if __name__ == "__main__":
    main()
