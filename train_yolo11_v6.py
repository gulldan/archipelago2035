#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_yolo11_v6.py — SAR/UAV training (fixed tile sampling, UAV augs, final phase)

Запуск (пример):
python train_yolo11_v6.py train \
  --data ./dataset/yolo_dataset/dataset.yaml \
  --model l \
  --epochs 60 \
  --batch 16 \
  --imgsz 1792 \
  --tile-train 1 --tile 1024 --stride 512 \
  --pos-frac 0.9 --max-neg 4 --workers 16 \
  --cover-thr 0.30 --final-epochs 10 --rand-phase 1
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

# =========================== utils ===========================


def _enable_h100_perf():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name} ({total:.1f} GB)"
    return "CPU"


def load_image_cv(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images_from_dir(dir_path: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        [str(p) for p in Path(dir_path).rglob("*") if p.suffix.lower() in exts]
    )


def xywhn_to_xyxy(x: np.ndarray) -> np.ndarray:
    xc, yc, w, h = x.T
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywhn(x: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = x.T
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return np.stack([xc, yc, w, h], axis=1)


def _tile_coords_with_offset(
    H: int, W: int, tile: int, stride: int, offx: int, offy: int
):
    # сетка с фазовым сдвигом (для train)
    xs = list(range(offx, max(W - tile + 1, 1), stride))
    ys = list(range(offy, max(H - tile + 1, 1), stride))
    if 0 not in xs:
        xs = [0] + xs
    if 0 not in ys:
        ys = [0] + ys
    last_x = max(W - tile, 0)
    last_y = max(H - tile, 0)
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return xs, ys


def tile_coords(H: int, W: int, tile: int, overlap: float):
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile + 1, 1), stride)) + [max(W - tile, 0)]
    ys = list(range(0, max(H - tile + 1, 1), stride)) + [max(H - tile, 0)]
    return xs, ys


# =========================== Dataset (fixed + random phase) ===========================


class SARTileDataset(Dataset):
    """
    Train/Val тайлы под UAV:
    - Позитивный тайл: центр GT внутри ИЛИ доля покрытия площади GT тайлом >= cover_thr.
    - Для каждого GT гарантируем тайл (по сетке со stride).
    - Train: можно рандомизировать фазу сетки (rand_phase=True) → меньше переобучения на стыки.
    - Val: только позитивные тайлы (без фона), фиксированная сетка.
    """

    def __init__(
        self,
        data_yaml: str,
        split: str = "train",
        imgsz: int = 1280,
        tile: int = 1024,
        stride: int = 512,
        pos_frac: float = 0.9,
        max_neg_per_image: int = 4,
        min_box_size: float = 2.0,
        transforms: A.Compose | None = None,
        cover_thr: float = 0.30,
        rand_phase: bool = True,
    ):
        super().__init__()
        self.imgsz = imgsz
        self.tile = tile
        self.stride = stride
        self.pos_frac = pos_frac
        self.max_neg_per_image = max_neg_per_image
        self.min_box_size = min_box_size
        self.transforms = transforms
        self.cover_thr = cover_thr
        self.rand_phase = rand_phase if split == "train" else False

        with open(data_yaml) as f:
            y = yaml.safe_load(f)
        root = Path(y.get("path", "."))

        train_key = "train" if "train" in y else "train/images"
        val_key = "val" if "val" in y else "val/images"
        if split == "train":
            img_dir = root / (
                y[train_key] if isinstance(y[train_key], str) else y["train"]["images"]
            )
        else:
            img_dir = root / (
                y[val_key] if isinstance(y[val_key], str) else y["val"]["images"]
            )

        if isinstance(y.get(split, None), dict):
            lbl_dir = root / y[split]["labels"]
        else:
            lbl_dir = Path(str(img_dir).replace("images", "labels"))

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples = []

        def _read_hw(img_path: Path) -> tuple[int, int]:
            try:
                with Image.open(str(img_path)) as img:
                    return img.height, img.width
            except Exception:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not load image from {img_path}")
                h, w = img.shape[:2]
                return h, w

        def _read_yolo_labels(label_path: Path) -> np.ndarray:
            if not label_path.exists() or label_path.stat().st_size == 0:
                return np.zeros((0, 5), dtype=np.float32)
            try:
                arr = np.loadtxt(str(label_path), ndmin=2, dtype=np.float32)
                if arr.shape[1] != 5:
                    return np.zeros((0, 5), dtype=np.float32)
                return arr
            except Exception:
                return np.zeros((0, 5), dtype=np.float32)

        def _center_in_tile(box_xyxy, x0, y0, tile):
            x1, y1, x2, y2 = box_xyxy
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return (x0 <= cx < x0 + tile) and (y0 <= cy < y0 + tile)

        def _cover_ratio(tile_rect, box):
            x0, y0, x1, y1 = tile_rect
            bx1, by1, bx2, by2 = box
            xx1 = max(x0, bx1)
            yy1 = max(y0, by1)
            xx2 = min(x1, bx2)
            yy2 = min(y1, by2)
            iw = max(0.0, xx2 - xx1)
            ih = max(0.0, yy2 - yy1)
            inter = iw * ih
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            return inter / area_b

        rng = np.random.default_rng(42)
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in exts:
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            bboxes = _read_yolo_labels(label_path)

            H, W = _read_hw(img_path)

            # сетка
            if self.rand_phase:
                offx = int(rng.integers(0, self.stride))
                offy = int(rng.integers(0, self.stride))
                xs, ys = _tile_coords_with_offset(
                    H, W, self.tile, self.stride, offx, offy
                )
            else:
                xs, ys = tile_coords(
                    H, W, self.tile, overlap=1 - self.stride / self.tile
                )

            boxes_abs = None
            if len(bboxes):
                xywhn = bboxes[:, 1:5]
                xyxy = xywhn_to_xyxy(xywhn)
                boxes_abs = (xyxy * np.array([W, H, W, H])).astype(np.float32)

            # гарантируем по тайлу для каждого GT (по сетке)
            forced_pos = set()
            if boxes_abs is not None and len(boxes_abs):
                for bx in boxes_abs:
                    cx = 0.5 * (bx[0] + bx[2])
                    cy = 0.5 * (bx[1] + bx[3])
                    gx = int(
                        np.clip(
                            (cx // self.stride) * self.stride, 0, max(W - self.tile, 0)
                        )
                    )
                    gy = int(
                        np.clip(
                            (cy // self.stride) * self.stride, 0, max(H - self.tile, 0)
                        )
                    )
                    forced_pos.add((gx, gy))

            pos_tiles, neg_tiles = [], []
            for y0 in ys:
                for x0 in xs:
                    x1, y1 = x0 + self.tile, y0 + self.tile
                    tile_rect = np.array([x0, y0, x1, y1], dtype=np.float32)

                    has_pos = False
                    if boxes_abs is not None and len(boxes_abs):
                        if any(
                            _center_in_tile(bx, x0, y0, self.tile) for bx in boxes_abs
                        ):
                            has_pos = True
                        else:
                            cov = (
                                np.max(
                                    [_cover_ratio(tile_rect, bx) for bx in boxes_abs]
                                )
                                if len(boxes_abs)
                                else 0.0
                            )
                            has_pos = cov >= self.cover_thr

                    if has_pos or ((x0, y0) in forced_pos):
                        pos_tiles.append((img_path, label_path, (H, W), (x0, y0)))
                    else:
                        neg_tiles.append((img_path, label_path, (H, W), (x0, y0)))

            n_pos = len(pos_tiles)
            if split == "train":
                n_neg = min(
                    len(neg_tiles),
                    max(
                        self.max_neg_per_image,
                        int(n_pos * (1 - self.pos_frac) / max(self.pos_frac, 1e-6)),
                    ),
                )
                if n_neg > 0:
                    neg_tiles = random.sample(neg_tiles, n_neg)
                else:
                    neg_tiles = []
                self.samples.extend(pos_tiles + neg_tiles)
            else:
                self.samples.extend(pos_tiles)

        self._read_yolo_labels = _read_yolo_labels  # reuse in __getitem__

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, lbl_path, (H, W), (x0, y0) = self.samples[index]
        img_full = load_image_cv(str(img_path))
        tile = img_full[y0 : y0 + self.tile, x0 : x0 + self.tile]
        h_t, w_t = tile.shape[:2]

        bboxes = self._read_yolo_labels(lbl_path)
        labels_out = []
        if len(bboxes):
            xywhn = bboxes[:, 1:5]
            cls = bboxes[:, 0:1]
            xyxy = xywhn_to_xyxy(xywhn)
            xyxy_abs = xyxy * np.array([W, H, W, H])

            cx = (xyxy_abs[:, 0] + xyxy_abs[:, 2]) / 2
            cy = (xyxy_abs[:, 1] + xyxy_abs[:, 3]) / 2
            keep = (
                (cx >= x0) & (cx < x0 + self.tile) & (cy >= y0) & (cy < y0 + self.tile)
            )

            if keep.any():
                boxes_keep = xyxy_abs[keep]
                cls = cls[keep]
                boxes_keep[:, [0, 2]] -= x0
                boxes_keep[:, [1, 3]] -= y0

                boxes_keep[:, 0::2] = boxes_keep[:, 0::2].clip(0, w_t)
                boxes_keep[:, 1::2] = boxes_keep[:, 1::2].clip(0, h_t)

                ws = boxes_keep[:, 2] - boxes_keep[:, 0]
                hs = boxes_keep[:, 3] - boxes_keep[:, 1]
                area_keep = (ws >= 2.0) & (hs >= 2.0)
                boxes_keep = boxes_keep[area_keep]
                cls = cls[area_keep]

                if len(boxes_keep):
                    xywh_tile = xyxy_to_xywhn(
                        boxes_keep / np.array([w_t, h_t, w_t, h_t])
                    )
                    labels_out = np.concatenate([cls, xywh_tile], axis=1)

        labels_out = np.array(labels_out, dtype=np.float32)

        if self.transforms:
            transformed = self.transforms(
                image=tile,
                bboxes=labels_out[:, 1:] if len(labels_out) else [],
                class_labels=labels_out[:, 0].tolist() if len(labels_out) else [],
            )
            tile = transformed["image"]
            if len(labels_out):
                b = np.array(transformed["bboxes"], dtype=np.float32)
                c = np.array(transformed["class_labels"], dtype=np.float32).reshape(
                    -1, 1
                )
                labels_out = np.concatenate([c, b], axis=1)
            else:
                labels_out = np.zeros((0, 5), dtype=np.float32)
        else:
            tile = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0

        # channels_last для даталоадера
        tile = tile.contiguous(memory_format=torch.channels_last)
        return tile, torch.from_numpy(labels_out)


# =========================== Augmentations ===========================


def build_transforms(imgsz: int, final_phase: bool = False) -> A.Compose:
    geo_p = 0.6 if not final_phase else 0.25
    blur_p = 0.18 if not final_phase else 0.06

    transforms = [
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(
            min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT
        ),
        # Геометрия под UAV
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.85, 1.18),
            translate_percent=(0.0, 0.08),
            rotate=(-15, 15),
            shear=(-6, 6),
            p=geo_p,
        ),
        # Фотометрия
        A.RandomBrightnessContrast(p=0.35),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=12, p=0.25
        ),
        # Деградации
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=3),
            ],
            p=blur_p,
        ),
        A.ISONoise(p=0.12),
        A.Defocus(p=0.10),
        A.CLAHE(p=0.10),
        # Редкие погодные эффекты
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomFog(p=0.05),
        A.RandomRain(p=0.05),
        A.RandomShadow(p=0.05),
        A.Normalize(),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.0
        ),
    )


def yolo_collate_fn(batch):
    imgs, labels = list(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    new_labels = []
    for i, lab in enumerate(labels):
        if lab.numel() == 0:
            continue
        bi = torch.full((lab.shape[0], 1), i, dtype=lab.dtype)
        new_labels.append(torch.cat([bi, lab], dim=1))
    if len(new_labels):
        labels_out = torch.cat(new_labels, dim=0)
    else:
        labels_out = torch.zeros((0, 6), dtype=torch.float32)
    return imgs, labels_out


def build_tile_dataloaders(
    data_yaml: str,
    imgsz: int,
    tile: int,
    stride: int,
    pos_frac: float,
    max_neg: int,
    batch_size: int,
    workers: int,
    cover_thr: float = 0.30,
    final_phase: bool = False,
    rand_phase: bool = True,
):
    tr_tf = build_transforms(imgsz, final_phase=final_phase)
    train_ds = SARTileDataset(
        data_yaml=data_yaml,
        split="train",
        imgsz=imgsz,
        tile=tile,
        stride=stride,
        pos_frac=pos_frac,
        max_neg_per_image=max_neg,
        transforms=tr_tf,
        cover_thr=cover_thr,
        rand_phase=rand_phase,
    )
    vl_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(
                min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.0
        ),
    )
    val_ds = SARTileDataset(
        data_yaml=data_yaml,
        split="val",
        imgsz=imgsz,
        tile=tile,
        stride=stride,
        pos_frac=1.0,
        max_neg_per_image=0,
        transforms=vl_tf,
        cover_thr=cover_thr,
        rand_phase=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(workers, 16),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=yolo_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(workers, 16),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=yolo_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


def attach_tile_dataloaders(
    model: YOLO,
    data_yaml: str,
    imgsz: int,
    tile: int,
    stride: int,
    pos_frac: float,
    max_neg: int,
    batch_size: int,
    workers: int,
    cover_thr: float = 0.30,
    final_phase_epochs: int = 10,
    rand_phase: bool = True,
):
    state = {"final_on": False, "final_phase_epochs": final_phase_epochs}

    def _on_pretrain_routine_start(trainer):
        tl, vl = build_tile_dataloaders(
            data_yaml,
            imgsz,
            tile,
            stride,
            pos_frac,
            max_neg,
            batch_size,
            workers,
            cover_thr,
            final_phase=False,
            rand_phase=rand_phase,
        )
        print(
            f"[TileTrain] train={len(tl.dataset)} samples, val={len(vl.dataset)} samples"
        )
        trainer.train_loader = tl
        trainer.val_loader = vl

    def _on_train_epoch_start(trainer):
        ep = trainer.epoch
        total = trainer.epochs
        if (not state["final_on"]) and (
            ep >= max(0, total - state["final_phase_epochs"])
        ):
            print(f"[TileTrain] FINAL PHASE at epoch {ep}/{total}")
            tl, _ = build_tile_dataloaders(
                data_yaml,
                imgsz,
                tile,
                stride,
                pos_frac,
                max_neg,
                batch_size,
                workers,
                cover_thr,
                final_phase=True,
                rand_phase=False,
            )
            trainer.train_loader = tl
            state["final_on"] = True

    model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)


# =========================== Train ===========================


def train_yolo(
    data_yaml: str,
    model_size: str = "l",
    epochs: int = 60,
    batch_size: int = 16,
    imgsz: int = 1792,
    project: str = "runs/detect",
    name: str = "sar_yolo",
    tile_train: int = 1,
    tile: int = 1024,
    stride: int = 512,
    pos_frac: float = 0.9,
    max_neg: int = 4,
    workers: int = 16,
    cover_thr: float = 0.30,
    final_phase_epochs: int = 10,
    rand_phase: int = 1,
):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    _enable_h100_perf()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device} / {gpu_info()}")

    model_name = f"yolo11{model_size}.pt"
    print(f"📥 loading {model_name}")
    model = YOLO(model_name)

    # чуть безопаснее по воркерам для hi-res
    if imgsz >= 1792 and workers > 16:
        print(f"⚠️ imgsz={imgsz}: reduce workers {workers} -> 16")
        workers = 16

    # AdamW + TF32/AMP
    base_lr = 0.0018 if model_size in ("l", "x") else 0.0025
    overrides = dict(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=project,
        name=f"{name}_{model_size}_img{imgsz}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        device=device,
        amp=True,
        workers=workers,
        optimizer="AdamW",
        lr0=base_lr,
        lrf=0.01,
        weight_decay=0.01,
        momentum=0.9,
        cos_lr=True,
        warmup_epochs=3.0,
        # Loss weights (1 класс)
        box=7.0,
        cls=0.3,
        dfl=2.0,
        # прочее
        max_det=400,
        iou=0.7,
        patience=80,
        multi_scale=False,
        save_period=5,
        verbose=True,
        plots=True,
    )

    # Включаем кастомные лоадеры
    if tile_train:
        attach_tile_dataloaders(
            model,
            data_yaml,
            imgsz,
            tile,
            stride,
            pos_frac,
            max_neg,
            batch_size,
            workers,
            cover_thr=cover_thr,
            final_phase_epochs=final_phase_epochs,
            rand_phase=bool(rand_phase),
        )

    # компиляция может ускорить (PyTorch 2.x)
    try:
        model.model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    results = model.train(**overrides)

    exp_dir = Path(results.save_dir)
    best = exp_dir / "weights" / "best.pt"
    if best.exists():
        import shutil

        shutil.copy2(best, "best.pt")
        print("✅ best.pt copied to ./best.pt")

    return {"exp_dir": str(exp_dir), "best": str(best)}


# =========================== CLI ===========================


def main():
    parser = argparse.ArgumentParser(
        "SAR YOLOv11 training (fixed tiling + UAV augs + final phase)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument(
        "--model", type=str, default="l", choices=["n", "s", "m", "l", "x"]
    )
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--imgsz", type=int, default=1792)
    p_train.add_argument("--project", type=str, default="runs/detect")
    p_train.add_argument("--name", type=str, default="sar_yolo")

    # tile-train
    p_train.add_argument("--tile-train", type=int, default=1)
    p_train.add_argument("--tile", type=int, default=1024)
    p_train.add_argument("--stride", type=int, default=512)
    p_train.add_argument("--pos-frac", type=float, default=0.9)
    p_train.add_argument("--max-neg", type=int, default=4)
    p_train.add_argument("--workers", type=int, default=16)
    p_train.add_argument("--cover-thr", type=float, default=0.30)
    p_train.add_argument("--final-epochs", type=int, default=10)
    p_train.add_argument(
        "--rand-phase",
        type=int,
        default=1,
        help="Randomize tile grid phase on train (0/1)",
    )

    args = parser.parse_args()

    if args.cmd == "train":
        info = train_yolo(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            tile_train=args.tile_train,
            tile=args.tile,
            stride=args.stride,
            pos_frac=args.pos_frac,
            max_neg=args.max_neg,
            workers=args.workers,
            cover_thr=args.cover_thr,
            final_phase_epochs=args.final_epochs,
            rand_phase=args.rand_phase,
        )
        print("TRAIN DONE:", info)


if __name__ == "__main__":
    main()
