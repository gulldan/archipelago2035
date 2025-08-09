#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ---- Albumentations (для tile‑train)
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from metric import df_to_bytes, open_df_as_bytes

# ---- Метрика организаторов
from metric import evaluate as metric_evaluate

# ===========================
# SUBMISSION INFERENCE CONSTS (подставь лучшие после grid)
# ===========================
MODEL_PATH = "best.pt"
IMG_SIZE = 1280
CONF_THR = 0.10
IOU_NMS_THR = 0.70
SLICE_SIZE = 1024
OVERLAP = 0.25
WBF_IOU_THR = 0.55

DEFAULT_THRESHOLDS = np.round(np.arange(0.3, 1.0, 0.07), 2)
DEFAULT_BETA = 1.0

COLUMNS = [
    "image_id",
    "label",
    "xc",
    "yc",
    "w",
    "h",
    "w_img",
    "h_img",
    "score",
    "time_spent",
]


# ===========================
# HELPERS
# ===========================
def gpu_info():
    """Return human‑readable information about the first CUDA device or CPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name} ({total:.1f} GB)"
    return "CPU"


def load_image_cv(path: str):
    """Load an image in RGB order using OpenCV.

    Raises
    ------
    ValueError if the image cannot be loaded.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images_from_dir(dir_path: str) -> list[str]:
    """Recursively list all image files under `dir_path`."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        [str(p) for p in Path(dir_path).rglob("*") if p.suffix.lower() in exts]
    )


def xywhn_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Convert relative xywh (normalized to [0,1]) to absolute xyxy coordinates."""
    xc, yc, w, h = x.T
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywhn(x: np.ndarray) -> np.ndarray:
    """Convert absolute xyxy to relative xywh (normalized to [0,1])."""
    x1, y1, x2, y2 = x.T
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return np.stack([xc, yc, w, h], axis=1)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1,y1,x2,y2] format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def wbf_light(
    boxes_xywhn: np.ndarray, scores: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray]:
    """A lightweight weighted box fusion for a single image.

    Boxes and scores must be sorted in descending order of scores. Boxes are
    expected in relative xywh format with values in [0,1].
    """
    if len(boxes_xywhn) == 0:
        return boxes_xywhn, scores
    boxes_xyxy = xywhn_to_xyxy(boxes_xywhn)
    order = scores.argsort()[::-1]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]

    fused_boxes, fused_scores = [], []
    used = np.zeros(len(boxes_xyxy), dtype=bool)
    for i in range(len(boxes_xyxy)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(boxes_xyxy)):
            if used[j]:
                continue
            if iou_xyxy(boxes_xyxy[i], boxes_xyxy[j]) >= iou_thr:
                cluster.append(j)
                used[j] = True
        cb = boxes_xyxy[cluster]
        cs = scores[cluster]
        w = cs / (cs.sum() + 1e-12)
        fused = (cb * w[:, None]).sum(axis=0)
        fused_boxes.append(fused)
        fused_scores.append(cs.max())
    fused_boxes = np.array(fused_boxes, dtype=np.float32)
    fused_scores = np.array(fused_scores, dtype=np.float32)
    return xyxy_to_xywhn(fused_boxes), fused_scores


def tile_coords(H: int, W: int, tile: int, overlap: float):
    """Compute the top‑left coordinates of tiles covering an image of size (H,W)."""
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile + 1, 1), stride)) + [max(W - tile, 0)]
    ys = list(range(0, max(H - tile + 1, 1), stride)) + [max(H - tile, 0)]
    return xs, ys


def iou_rect(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two axis‑aligned rectangles given as [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-12)


# ===========================
# TILE TRAIN DATASET
# ===========================


class SARTileDataset(Dataset):
    """Dataset that yields image tiles and corresponding labels for training.

    Images are sliced into overlapping tiles. Tiles containing at least one
    ground‑truth box with an IoU above `pos_iou_thr` are treated as positives;
    others are negatives. A configurable fraction of negatives is included in
    the training set, and a small number of negatives is optionally included
    in the validation set.
    """

    def __init__(
        self,
        data_yaml: str,
        split: str = "train",
        imgsz: int = 1280,
        tile: int = 1024,
        stride: int = 512,
        pos_frac: float = 0.5,
        max_neg_per_image: int = 4,
        min_box_size: float = 1.0,
        transforms: A.Compose | None = None,
        pos_iou_thr: float = 0.2,
    ):
        super().__init__()

        self.imgsz = imgsz
        self.tile = tile
        self.stride = stride
        self.pos_frac = pos_frac
        self.max_neg_per_image = max_neg_per_image
        self.min_box_size = min_box_size
        self.transforms = transforms
        self.pos_iou_thr = pos_iou_thr

        # Load dataset description
        with open(data_yaml) as f:
            y = yaml.safe_load(f)

        root = Path(y.get("path", "."))
        # support both Ultralytics v8 and custom split formats
        train_key = "train" if "train" in y else "train/images"
        val_key = "val" if "val" in y else "val/images"
        if split == "train":
            img_dir = (
                root / y[train_key]
                if isinstance(y[train_key], str)
                else root / y["train"]["images"]
            )
        else:
            img_dir = (
                root / y[val_key]
                if isinstance(y[val_key], str)
                else root / y["val"]["images"]
            )

        if isinstance(y.get(split, None), dict):
            lbl_dir = root / y[split]["labels"]
        else:
            lbl_dir = Path(str(img_dir).replace("images", "labels"))

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples = []
        rng = np.random.default_rng(42)

        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in exts:
                continue
            label_path = lbl_dir / (img_path.stem + ".txt")
            bboxes = self._read_yolo_labels(label_path)

            H, W = self._read_hw(img_path)
            xs, ys = tile_coords(H, W, tile, overlap=1 - self.stride / self.tile)

            # Convert labels to absolute pixel coordinates once
            boxes_abs = None
            if len(bboxes):
                xywhn = bboxes[:, 1:5]
                xyxy = xywhn_to_xyxy(xywhn)
                boxes_abs = (xyxy * np.array([W, H, W, H])).astype(np.float32)

            pos_tiles, neg_tiles = [], []
            for y0 in ys:
                for x0 in xs:
                    x1, y1 = x0 + self.tile, y0 + self.tile
                    tile_rect = np.array([x0, y0, x1, y1], dtype=np.float32)
                    has_pos = False
                    if boxes_abs is not None and len(boxes_abs):
                        # compute IoU for each box and determine if any exceed threshold
                        ious = []
                        for bx in boxes_abs:
                            ious.append(iou_rect(tile_rect, bx))
                        has_pos = (np.max(ious) >= self.pos_iou_thr) if ious else False
                    # assign tile to positive or negative list
                    if has_pos:
                        pos_tiles.append((img_path, label_path, (H, W), (x0, y0)))
                    else:
                        neg_tiles.append((img_path, label_path, (H, W), (x0, y0)))

            n_pos = len(pos_tiles)
            if split == "train":
                # include only a subset of negatives to achieve desired pos_frac
                n_neg = min(
                    len(neg_tiles),
                    int(n_pos * (1 - self.pos_frac) / max(self.pos_frac, 1e-6))
                    + self.max_neg_per_image,
                )
                if n_neg > 0:
                    import random

                    neg_tiles = random.sample(neg_tiles, n_neg)
                else:
                    neg_tiles = []
                self.samples.extend(pos_tiles + neg_tiles)
            else:
                # on validation include some negatives controlled by pos_frac and max_neg_per_image
                if self.pos_frac < 1.0:
                    # sample negatives according to pos_frac for validation
                    n_neg = min(
                        len(neg_tiles),
                        int(n_pos * (1 - self.pos_frac) / max(self.pos_frac, 1e-6))
                        + self.max_neg_per_image,
                    )
                    import random

                    if n_neg > 0:
                        neg_tiles = random.sample(neg_tiles, n_neg)
                    else:
                        neg_tiles = []
                    self.samples.extend(pos_tiles + neg_tiles)
                else:
                    # default behaviour: only positives
                    self.samples.extend(pos_tiles)

    def _read_hw(self, img_path: Path) -> tuple[int, int]:
        """Return height and width of an image without fully loading it."""
        try:
            with Image.open(str(img_path)) as img:
                return img.height, img.width
        except Exception:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image from {img_path}")
            h, w = img.shape[:2]
            return h, w

    def _read_yolo_labels(self, label_path: Path) -> np.ndarray:
        """Read YOLO labels from a txt file. Return empty array if missing or malformed."""
        if not label_path.exists() or label_path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        try:
            with open(label_path) as f:
                lines = f.read().strip()
                if not lines:
                    return np.zeros((0, 5), dtype=np.float32)
            arr = np.loadtxt(str(label_path), ndmin=2, dtype=np.float32)
            if arr.shape[1] != 5:
                return np.zeros((0, 5), dtype=np.float32)
            return arr
        except (ValueError, OSError):
            return np.zeros((0, 5), dtype=np.float32)

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
                area_keep = (ws >= self.min_box_size) & (hs >= self.min_box_size)
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

        return tile, torch.from_numpy(labels_out)


def build_transforms(imgsz: int) -> A.Compose:
    """Build Albumentations transforms tailored for UAV imagery.

    We include moderate photometric augmentations and several geometric
    transforms (rotations/affine) to better simulate drone viewpoints. Noise
    and blur are used to emulate motion or compression artefacts.
    """
    """Construct an Albumentations augmentation pipeline.

    The pipeline combines geometric, photometric and degradation transforms.
    If Albumentations supports Mosaic and MixUp (version >= 1.4), these
    multi-image augmentations will be included to improve small-object
    generalization. Mosaic has probability 0.5 and MixUp 0.15. Otherwise,
    these transforms are skipped.
    """
    # base transforms
    transforms = [
        A.LongestMaxSize(max_size=imgsz),
        A.PadIfNeeded(
            min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT
        ),
        # Geometric augmentations
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.05),
            rotate=(-12, 12),
            shear=(-4, 4),
            p=0.6,
        ),
        # Photometric augmentations
        A.RandomBrightnessContrast(p=0.25),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=18, val_shift_limit=10, p=0.2
        ),
        # Degradations: blur/noise. Use valid parameters to silence warnings.
        A.OneOf(
            [
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=(3, 3)),
            ],
            p=0.15,
        ),
        A.ISONoise(p=0.1),
        A.HorizontalFlip(p=0.5),
    ]

    # Conditionally include Mosaic and MixUp if available
    try:
        # Albumentations ≥ 1.4 introduces Mosaic and MixUp transforms in the API
        if hasattr(A, "Mosaic") and hasattr(A, "MixUp"):
            transforms.insert(0, A.Mosaic(p=0.5))
            transforms.insert(1, A.MixUp(p=0.15))
    except Exception:
        # If any errors occur while checking or adding, skip these transforms
        pass

    transforms.extend(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.0
        ),
    )


def yolo_collate_fn(batch):
    """Custom collate_fn to assemble batches of images and labels for YOLO."""
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
    pos_iou_thr: float = 0.2,
):
    """Construct training and validation dataloaders with tile sampling."""
    tr_tf = build_transforms(imgsz)
    train_ds = SARTileDataset(
        data_yaml=data_yaml,
        split="train",
        imgsz=imgsz,
        tile=tile,
        stride=stride,
        pos_frac=pos_frac,
        max_neg_per_image=max_neg,
        transforms=tr_tf,
        pos_iou_thr=pos_iou_thr,
    )
    # validation transforms: minimal augmentation
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
    # include some negatives on validation by setting pos_frac<1 and providing max_neg_per_image>0
    val_ds = SARTileDataset(
        data_yaml=data_yaml,
        split="val",
        imgsz=imgsz,
        tile=tile,
        stride=stride,
        pos_frac=0.85,
        max_neg_per_image=2,
        transforms=vl_tf,
        pos_iou_thr=pos_iou_thr,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=yolo_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
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
    pos_iou_thr: float = 0.2,
):
    """Attach custom dataloaders to a YOLO trainer via callback."""

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
            pos_iou_thr,
        )
        print(
            f"[TileTrain] Replacing loaders: train={len(tl.dataset)} samples, val={len(vl.dataset)} samples"
        )
        trainer.train_loader = tl
        trainer.val_loader = vl

    model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)


# ===========================
# TRAIN
# ===========================


def train_yolo(
    data_yaml: str,
    model_size: str = "l",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 1280,
    project: str = "runs/detect",
    name: str = "sar_yolo",
    tile_train: int = 1,
    tile: int = 1024,
    stride: int = 512,
    pos_frac: float = 0.5,
    max_neg: int = 6,
    workers: int = 8,
    pos_iou_thr: float = 0.2,
):
    """Train a YOLOv11 model on SAR people detection using tiled training.

    Parameters
    ----------
    data_yaml : str
        Path to a YAML file describing the dataset.
    model_size : str, optional
        One of {"n","s","m","l","x"}, by default "l".
    epochs : int, optional
        Number of training epochs, by default 100.
    batch_size : int, optional
        Batch size, by default 16.
    imgsz : int, optional
        Target input image size for the network, by default 1280.
    project, name : str
        Experiment grouping for Ultralytics.
    tile_train : int
        If 1, enable tiled training with our custom dataloaders.
    tile : int
        Tile size in pixels.
    stride : int
        Tile stride in pixels.
    pos_frac : float
        Fraction of positive tiles per image (controls negative sampling).
    max_neg : int
        Maximum number of negative tiles per image.
    workers : int
        Number of worker threads for data loading.
    pos_iou_thr : float
        IoU threshold between tile and ground truth for positive assignment.
    """
    # set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    import random as _random

    _random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device} / {gpu_info()}")

    model_name = f"yolo11{model_size}.pt"
    print(f"📥 loading {model_name}")
    model = YOLO(model_name)

    # tune training hyper‑parameters. Using SGD with cosine decay, stable momentum, etc.
    # adapt workers for high‑resolution images: too many workers can slow down I/O
    if imgsz >= 1536 and workers > 16:
        print(
            f"⚠️ High‑resolution imgsz={imgsz} detected; reducing workers from {workers} to 16 to avoid I/O overhead."
        )
        workers = 16

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
        # optimizer settings
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        lr0=0.01 if model_size in ("m", "s") else 0.005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        # loss weights
        box=7.5,
        cls=0.7,
        dfl=2.0,
        # detection settings
        max_det=400,
        iou=0.7,
        patience=30,
        multi_scale=False,
        save_period=5,
        verbose=True,
        plots=True,
    )

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
            pos_iou_thr,
        )

    results = model.train(**overrides)

    exp_dir = Path(results.save_dir)
    best = exp_dir / "weights" / "best.pt"
    if best.exists():
        import shutil

        shutil.copy2(best, "best.pt")
        print("✅ best.pt copied to ./best.pt")

    return {"exp_dir": str(exp_dir), "best": str(best)}


# ===========================
# SAHI‑like TILED INFERENCE (+CSV for grid)
# ===========================


@torch.no_grad()
def predict_image_tiled_internal(
    model: YOLO,
    image: np.ndarray,
    slice_size: int,
    overlap: float,
    conf: float,
    iou_nms: float,
    wbf_iou: float,
    imgsz: int,
):
    """Run tiled inference over an image and fuse boxes with WBF."""
    H, W = image.shape[:2]
    xs, ys = tile_coords(H, W, slice_size, overlap)
    all_boxes, all_scores = [], []
    t0 = time.time()

    for y in ys:
        for x in xs:
            tile = image[y : y + slice_size, x : x + slice_size]
            res = model.predict(
                source=tile,
                imgsz=imgsz,
                conf=conf,
                iou=iou_nms,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )[0]
            if res.boxes is None or res.boxes.xywhn.shape[0] == 0:
                continue
            xywhn_tile = res.boxes.xywhn.cpu().numpy()
            scores_tile = res.boxes.conf.cpu().numpy()

            xyxy_tile = xywhn_to_xyxy(xywhn_tile)
            th, tw = tile.shape[:2]
            xyxy_abs = xyxy_tile * np.array([tw, th, tw, th])
            xyxy_abs[:, [0, 2]] += x
            xyxy_abs[:, [1, 3]] += y
            xyxy_full = xyxy_abs / np.array([W, H, W, H])
            xywhn_full = xyxy_to_xywhn(xyxy_full)

            all_boxes.append(xywhn_full)
            all_scores.append(scores_tile)

    dt = time.time() - t0

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), dt

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    boxes, scores = wbf_light(boxes, scores, iou_thr=wbf_iou)
    return boxes, scores, dt


@torch.no_grad()
def predict_image_tiled_tta_internal(
    model: YOLO,
    image: np.ndarray,
    slice_size: int,
    overlap: float,
    conf: float,
    iou_nms: float,
    wbf_iou: float,
    imgsz: int,
):
    """Run tiled inference with simple horizontal flip Test-Time Augmentation (TTA).

    This runs predictions on the original image and on a horizontally flipped
    version, then flips back the boxes and fuses them with weighted box fusion.
    """
    # Original prediction
    boxes, scores, dt0 = predict_image_tiled_internal(
        model, image, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
    )
    # Horizontal flip
    img_hflip = image[:, ::-1].copy()
    boxes_hflip, scores_hflip, dt1 = predict_image_tiled_internal(
        model, img_hflip, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
    )
    # Adjust horizontally flipped boxes back: xc -> 1 - xc
    if boxes_hflip.shape[0] > 0:
        boxes_hflip_adj = boxes_hflip.copy()
        boxes_hflip_adj[:, 0] = 1.0 - boxes_hflip[:, 0]
    else:
        boxes_hflip_adj = boxes_hflip

    # Vertical flip
    img_vflip = image[::-1, :].copy()
    boxes_vflip, scores_vflip, dt2 = predict_image_tiled_internal(
        model, img_vflip, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
    )
    if boxes_vflip.shape[0] > 0:
        boxes_vflip_adj = boxes_vflip.copy()
        boxes_vflip_adj[:, 1] = 1.0 - boxes_vflip[:, 1]
    else:
        boxes_vflip_adj = boxes_vflip

    # 90 degree rotation (counter-clockwise) using numpy.rot90
    img_rot90 = np.rot90(image).copy()
    boxes_rot90, scores_rot90, dt3 = predict_image_tiled_internal(
        model, img_rot90, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
    )
    if boxes_rot90.shape[0] > 0:
        boxes_rot90_adj = boxes_rot90.copy()
        # For 90° ccw: x'<- y, y'<- 1 - x; w'<- h, h'<- w
        xc = boxes_rot90[:, 0]
        yc = boxes_rot90[:, 1]
        w = boxes_rot90[:, 2]
        h = boxes_rot90[:, 3]
        boxes_rot90_adj[:, 0] = yc
        boxes_rot90_adj[:, 1] = 1.0 - xc
        boxes_rot90_adj[:, 2] = h
        boxes_rot90_adj[:, 3] = w
    else:
        boxes_rot90_adj = boxes_rot90

    # Concatenate all predictions
    boxes_list = []
    scores_list = []
    if boxes.shape[0] > 0:
        boxes_list.append(boxes)
        scores_list.append(scores)
    if boxes_hflip_adj.shape[0] > 0:
        boxes_list.append(boxes_hflip_adj)
        scores_list.append(scores_hflip)
    if boxes_vflip_adj.shape[0] > 0:
        boxes_list.append(boxes_vflip_adj)
        scores_list.append(scores_vflip)
    if boxes_rot90_adj.shape[0] > 0:
        boxes_list.append(boxes_rot90_adj)
        scores_list.append(scores_rot90)
    if boxes_list:
        boxes_all = np.vstack(boxes_list)
        scores_all = np.concatenate(scores_list, axis=0)
    else:
        boxes_all = np.zeros((0, 4), dtype=np.float32)
        scores_all = np.zeros((0,), dtype=np.float32)
    # Fuse predictions using weighted box fusion
    boxes_fused, scores_fused = wbf_light(boxes_all, scores_all, iou_thr=wbf_iou)
    return boxes_fused, scores_fused, dt0 + dt1 + dt2 + dt3


def run_tiled_inference_to_df(
    model_path: str,
    image_paths: list[str],
    slice_size=1024,
    overlap=0.25,
    conf=0.10,
    iou_nms=0.7,
    wbf_iou=0.55,
    imgsz=1280,
    tta: bool = False,
) -> pd.DataFrame:
    """Perform tiled inference over a list of images and return results as a DataFrame.

    If `tta` is True, a simple horizontal flip Test-Time Augmentation (TTA) is
    applied during inference and results are fused with weighted box fusion.
    """
    model = YOLO(model_path)
    rows = []
    for img_path in tqdm(image_paths, desc="Infer(SAHI)"):
        img_id = Path(img_path).stem
        img = load_image_cv(img_path)
        H, W = img.shape[:2]
        if tta:
            boxes, scores, dt = predict_image_tiled_tta_internal(
                model, img, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
            )
        else:
            boxes, scores, dt = predict_image_tiled_internal(
                model, img, slice_size, overlap, conf, iou_nms, wbf_iou, imgsz
            )
        if boxes.shape[0] == 0:
            rows.append(
                dict(
                    image_id=img_id,
                    label=0,
                    xc=0.0,
                    yc=0.0,
                    w=0.0,
                    h=0.0,
                    w_img=W,
                    h_img=H,
                    score=0.0,
                    time_spent=dt,
                )
            )
            continue
        for b, s in zip(boxes, scores):
            xc, yc, w, h = b.tolist()
            rows.append(
                dict(
                    image_id=img_id,
                    label=0,
                    xc=xc,
                    yc=yc,
                    w=w,
                    h=h,
                    w_img=W,
                    h_img=H,
                    score=float(s),
                    time_spent=dt,
                )
            )

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


# ===========================
# GRID‑SEARCH
# ===========================


def grid_search_thresholds(
    model_path: str,
    val_images: list[str],
    gt_csv_path: str,
    beta: float = DEFAULT_BETA,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    slice_sizes=(1024, 896, 1152),
    overlaps=(0.20, 0.25, 0.30),
    confs=np.linspace(0.01, 0.35, 9),
    iou_nms_list=(0.5, 0.6, 0.7, 0.8),
    wbf_ious=(0.50, 0.55, 0.60, 0.65),
    imgsz_list=(1280, 1536),
):
    """Grid search inference hyper‑parameters to maximize the evaluation metric."""
    gt_bytes = open_df_as_bytes(gt_csv_path)
    image_ids = [Path(p).stem for p in val_images]
    m = len(set(image_ids))

    best = (-1.0, None)
    log = []
    for sl in slice_sizes:
        for ov in overlaps:
            for conf in confs:
                for iou_nms in iou_nms_list:
                    for wbf_iou in wbf_ious:
                        for imgsz in imgsz_list:
                            df = run_tiled_inference_to_df(
                                model_path=model_path,
                                image_paths=val_images,
                                slice_size=sl,
                                overlap=ov,
                                conf=conf,
                                iou_nms=iou_nms,
                                wbf_iou=wbf_iou,
                                imgsz=imgsz,
                            )
                            pred_bytes = df_to_bytes(df)
                            metric, acc, fp_rate, avg_time = metric_evaluate(
                                predicted_file=pred_bytes,
                                gt_file=gt_bytes,
                                thresholds=thresholds,
                                beta=beta,
                                m=m,
                                parallelize=True,
                            )
                            rec = dict(
                                metric=float(metric),
                                slice_size=sl,
                                overlap=ov,
                                conf=float(conf),
                                iou_nms=float(iou_nms),
                                wbf_iou=float(wbf_iou),
                                imgsz=imgsz,
                                avg_time=float(avg_time),
                            )
                            log.append(rec)
                            print(
                                f"[sl={sl} ov={ov} conf={conf:.3f} nms={iou_nms} wbf={wbf_iou:.2f} imgsz={imgsz}] "
                                f"F{beta:.0f}={metric:.4f}  time={avg_time:.3f}"
                            )

                            if metric > best[0]:
                                best = (metric, rec)

    pd.DataFrame(log).to_csv("grid_search_log.csv", index=False)
    with open("best_thresh.json", "w") as f:
        json.dump(dict(best_metric=best[0], **best[1]), f, indent=2)

    print("\n==== BEST ====")
    print(best)
    return best


# ===========================
# SUBMISSION API
# ===========================

_device = "cuda" if torch.cuda.is_available() else "cpu"
_submission_model = None


def _get_submission_model():
    """Load the model for submission inference and keep it in memory."""
    global _submission_model
    if _submission_model is None:
        _submission_model = YOLO(MODEL_PATH)
        _submission_model.to(_device).eval()
    return _submission_model


@torch.no_grad()
def _infer_tiled_submission(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled inference for the submission API."""
    model = _get_submission_model()
    H, W = image.shape[:2]
    xs, ys = tile_coords(H, W, SLICE_SIZE, OVERLAP)
    all_boxes, all_scores = [], []
    for y in ys:
        for x in xs:
            tile = image[y : y + SLICE_SIZE, x : x + SLICE_SIZE]
            res = model.predict(
                source=tile,
                imgsz=IMG_SIZE,
                conf=CONF_THR,
                iou=IOU_NMS_THR,
                device=0 if _device == "cuda" else _device,
                verbose=False,
            )[0]
            if res.boxes is None or res.boxes.xywhn.shape[0] == 0:
                continue
            xywhn_tile = res.boxes.xywhn.cpu().numpy()
            scores_tile = res.boxes.conf.cpu().numpy()

            xyxy_tile = xywhn_to_xyxy(xywhn_tile)
            th, tw = tile.shape[:2]
            xyxy_abs = xyxy_tile * np.array([tw, th, tw, th])
            xyxy_abs[:, [0, 2]] += x
            xyxy_abs[:, [1, 3]] += y
            xyxy_full = xyxy_abs / np.array([W, H, W, H])
            xywhn_full = xyxy_to_xywhn(xyxy_full)

            all_boxes.append(xywhn_full)
            all_scores.append(scores_tile)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    boxes, scores = wbf_light(boxes, scores, iou_thr=WBF_IOU_THR)
    return boxes, scores


def infer_image_bbox(image: np.ndarray) -> list[dict]:
    """Wrapper for submission API: returns list of detections for one image."""
    dets: list[dict] = []
    boxes, scores = _infer_tiled_submission(image)
    for b, s in zip(boxes, scores):
        xc, yc, w, h = b.tolist()
        dets.append(
            {
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h),
                "label": 0,
                "score": float(s),
            }
        )
    return dets


def predict(images: np.ndarray | list[np.ndarray]) -> list[list[dict]]:
    """Entry point for platform submission: accepts a single image or list of images."""
    if isinstance(images, np.ndarray):
        images = [images]
    return [infer_image_bbox(img) for img in images]


# ===========================
# CLI
# ===========================
def main():
    parser = argparse.ArgumentParser("SAR YOLOv11 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument(
        "--model", type=str, default="l", choices=["n", "s", "m", "l", "x"]
    )
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--imgsz", type=int, default=1280)
    p_train.add_argument("--project", type=str, default="runs/detect")
    p_train.add_argument("--name", type=str, default="sar_yolo")

    # tile‑train
    p_train.add_argument("--tile-train", type=int, default=1)
    p_train.add_argument("--tile", type=int, default=1024)
    p_train.add_argument("--stride", type=int, default=512)
    p_train.add_argument("--pos-frac", type=float, default=0.5)
    p_train.add_argument("--max-neg", type=int, default=6)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--pos-iou-thr", type=float, default=0.2)

    # grid‑search
    p_grid = sub.add_parser("grid")
    p_grid.add_argument("--model", type=str, required=True)
    p_grid.add_argument("--val-images-dir", type=str, required=True)
    p_grid.add_argument("--gt-csv", type=str, required=True)
    p_grid.add_argument("--beta", type=float, default=DEFAULT_BETA)

    # optional: plain inference to csv
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--model", type=str, required=True)
    p_inf.add_argument("--images-dir", type=str, required=True)
    p_inf.add_argument("--out-csv", type=str, default="predictions.csv")
    p_inf.add_argument("--slice-size", type=int, default=1024)
    p_inf.add_argument("--overlap", type=float, default=0.25)
    p_inf.add_argument("--conf", type=float, default=0.10)
    p_inf.add_argument("--iou-nms", type=float, default=0.7)
    p_inf.add_argument("--wbf-iou", type=float, default=0.55)
    p_inf.add_argument("--imgsz", type=int, default=1280)
    p_inf.add_argument(
        "--tta", action="store_true", help="Enable horizontal flip TTA during inference"
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
            pos_iou_thr=args.pos_iou_thr,
        )
        print("TRAIN DONE:", info)

    elif args.cmd == "grid":
        val_images = list_images_from_dir(args.val_images_dir)
        best = grid_search_thresholds(
            model_path=args.model,
            val_images=val_images,
            gt_csv_path=args.gt_csv,
            beta=args.beta,
        )
        print("BEST:", best)

    elif args.cmd == "infer":
        images = list_images_from_dir(args.images_dir)
        df = run_tiled_inference_to_df(
            model_path=args.model,
            image_paths=images,
            slice_size=args.slice_size,
            overlap=args.overlap,
            conf=args.conf,
            iou_nms=args.iou_nms,
            wbf_iou=args.wbf_iou,
            imgsz=args.imgsz,
            tta=args.tta,
        )
        df.to_csv(args.out_csv, index=False)
        print(f"CSV saved to {args.out_csv}, rows={len(df)}")


if __name__ == "__main__":
    main()
