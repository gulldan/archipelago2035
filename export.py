#!/usr/bin/env python3
"""
export.py — Convert PyTorch .pt (Ultralytics YOLO *.pt) to ONNX for inference.

Usage:
  uv run python export.py --pt path/to/best.pt --onnx out/best.onnx --imgsz 1536 --opset 12 --fp16 0

Notes:
  - Requires ultralytics installed only for export step (not needed for inference).
  - If --fp16=1, will try to export fp16 ONNX (may fall back to fp32).
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser("YOLO PT->ONNX exporter")
    ap.add_argument("--pt", type=str, required=True, help="Path to .pt model (Ultralytics)")
    ap.add_argument("--onnx", type=str, required=True, help="Output .onnx path")
    ap.add_argument("--imgsz", type=int, default=1536, help="Export image size (square)")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset")
    ap.add_argument("--dynamic", type=int, default=1, help="Dynamic shapes (0/1)")
    ap.add_argument("--simplify", type=int, default=1, help="Simplify ONNX (0/1)")
    ap.add_argument("--fp16", type=int, default=0, help="Export fp16 (0/1)")
    args = ap.parse_args()

    from ultralytics import YOLO  # import locally to avoid runtime dep

    pt = Path(args.pt)
    assert pt.exists(), f".pt not found: {pt}"
    model = YOLO(str(pt))
    print(f"Loading: {pt}")

    print("Exporting ONNX...")
    onnx_path = model.export(
        format="onnx",
        imgsz=int(args.imgsz),
        opset=int(args.opset),
        dynamic=bool(args.dynamic),
        simplify=bool(args.simplify),
        half=bool(args.fp16),
    )
    out = Path(args.onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes to default filename; copy/rename to requested path
    src = Path(onnx_path)
    if src.resolve().as_posix() != out.resolve().as_posix():
        out.write_bytes(src.read_bytes())
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


