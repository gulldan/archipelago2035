from __future__ import annotations

# pyright: reportMissingImports=false
# ruff: noqa: E402

"""Faster R-CNN ResNet50-FPNv2 — v11 (IterableDataset + Albumentations, turbo-start).

- IterableDataset для train: без предварительной материализации списка тайлов
- Шардинг по DDP-рангам и по DataLoader-воркерам (disjoint подмножества картинок)
- Альбоментации (Albumentations) для качества: геометрия + цвет, bbox-aware
  (используется opencv-python-headless ТОЛЬКО внутри альбументаций; декод JPEG — TurboJPEG)
- Большие исходники (до 8000×8000): работаем через тайлы 1024×1024, аугменты на тайле
- LRU-кэш декодированных изображений per-worker (уменьшает IO/декод)
- RAM-safe: CPU хранит uint8 CHW; нормализация/деление → на GPU (bf16/fp16)
- CUDA prefetcher (асинхронный перенос batch → GPU)
- channels_last, fused/foreach AdamW, pin_memory(+cuda), persistent_workers
- Модель: маленькие якоря (4..64), фиксированный transform size (compile-friendly)

Запуск (пример):
python train_frcnn_uav_v2_fast.py --data ./dataset/yolo_dataset/dataset.yaml \
  --epochs 60 --steps-per-epoch 3500 --batch 100 --imgsz 1280 \
  --tile 1024 --stride 512 --pos-frac 0.9 --max-neg 4 --cover-thr 0.30 \
  --workers 32 --prefetch-factor 4 --pin-memory 1 --persistent-workers 1 \
  --lr 1.6e-3 --amp-dtype bf16 --compile 0 --multiscale 0 --mp-context fork \
  --cache-images 1 --cache-size 256 --use-alb 1

Зависимости (важно):
  pip install albumentations opencv-python-headless pillow-simd turbojpeg
"""

import argparse
import gc
import json
import math
import os
import random
import time
import types
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import cycle, product
from pathlib import Path

# ---------- threads/env ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.ops import box_iou, nms
from torchvision.tv_tensors import BoundingBoxes
from torchvision.tv_tensors import Image as TvImage
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

# optional dynamo (torch.compile)
try:
    import torch._dynamo as dynamo  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional
    dynamo = None

# ============ Albumentations (CPU) ============
try:
    import albumentations as A
    import cv2

    _has_alb = True
except Exception:
    _has_alb = False
    cv2 = None

# ============ fast JPEG decoder (TurboJPEG) ============
try:
    from turbojpeg import TJPF_RGB, TurboJPEG

    _jpeg = TurboJPEG()
    _has_turbo = True
except Exception:
    _jpeg = None
    _has_turbo = False


# ============ misc perf ============
def set_perf() -> None:
    """Enable CUDA perf knobs when GPU is available."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def seed_all(seed: int = 42) -> None:
    """Seed RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp() -> bool:
    """Return True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    """Current DDP rank (0 if not DDP)."""
    return dist.get_rank() if is_ddp() else 0


def ddp_world() -> int:
    """DDP world size (1 if not DDP)."""
    return dist.get_world_size() if is_ddp() else 1


# ============ IO helpers ============
def load_image_fast(path: Path) -> np.ndarray:
    """Fast RGB decode → np.uint8 (H,W,3).

    Uses TurboJPEG for JPG/JPEG, Pillow-SIMD otherwise.
    """
    ext = path.suffix.lower()
    if _has_turbo and ext in (".jpg", ".jpeg"):
        with path.open("rb") as f:
            buf = f.read()
        return _jpeg.decode(buf, pixel_format=TJPF_RGB)  # HWC uint8 RGB
    with Image.open(str(path)) as im:
        return np.array(im.convert("RGB"))


def read_hw_fast(path: Path) -> tuple[int, int]:
    """Read image height and width without full decode (uses TurboJPEG header when possible)."""
    ext = path.suffix.lower()
    if _has_turbo and ext in (".jpg", ".jpeg"):
        with path.open("rb") as f:
            header = _jpeg.decode_header(f.read())
        # header = (width, height, subsample, colorspace)
        return int(header[1]), int(header[0])
    with Image.open(str(path)) as im:
        return im.height, im.width


# ============ math helpers ============
def xywhn_to_xyxy_abs(xywhn: np.ndarray, W: int, H: int) -> np.ndarray:
    if xywhn.size == 0:
        return np.zeros((0, 4), np.float32)
    xc, yc, w, h = xywhn.T
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return np.stack([x1, y1, x2, y2], 1).astype(np.float32)


def tile_grid(
    h: int, w: int, tile: int, stride: int, offx: int = 0, offy: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic grid for tiling a large image with boundaries covered."""
    xs = list(range(offx, max(w - tile + 1, 1), stride))
    ys = list(range(offy, max(h - tile + 1, 1), stride))
    if 0 not in xs:
        xs = [0, *xs]
    if 0 not in ys:
        ys = [0, *ys]
    last_x = max(w - tile, 0)
    last_y = max(h - tile, 0)
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)
    return np.array(sorted(set(xs)), dtype=np.int32), np.array(
        sorted(set(ys)), dtype=np.int32
    )


# ============ worker-local LRU cache for images ============
class _ImageLRU:
    __slots__ = ("cache", "cap")

    def __init__(self, capacity: int = 256) -> None:
        self.cap = int(max(1, capacity))
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        k = str(path)
        if k in self.cache:
            img = self.cache.pop(k)
            self.cache[k] = img
            return img
        img = load_image_fast(path)
        self.cache[k] = img
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return img


# one LRU per worker-process
_IMG_LRU: _ImageLRU | None = None
_IMG_LRU_CAP: int = 256


# ============ datasets ============
@dataclass
class Meta:
    H: int
    W: int
    boxes_abs: np.ndarray


class TileIterableDataset(IterableDataset):
    """Train IterableDataset: генерируем тайлы на лету, без полного списка.
    - DDP-шардинг по рангам
    - worker-шардинг по id воркера
    - Позитивные тайлы: центр-в-тайле + cover>=thr (локальный перебор вокруг бокса)
    - Негативы: ограниченная подвыборка.
    """

    def __init__(
        self,
        data_yaml: str,
        split: str = "train",
        imgsz: int = 1024,
        tile: int = 1024,
        stride: int = 512,
        pos_frac: float = 0.9,
        max_neg_per_image: int = 4,
        cover_thr: float = 0.30,
        min_box_wh: float = 2.0,
        steps_per_epoch: int = 2000,
        use_alb: bool = True,
        cache_images: bool = True,
        cache_size: int = 256,
    ) -> None:
        assert split == "train", "TileIterableDataset предназначен для train"
        super().__init__()
        self.imgsz = imgsz
        self.tile = tile
        self.stride = stride
        self.pos_frac = pos_frac
        self.max_neg_per_image = max_neg_per_image
        self.cover_thr = cover_thr
        self.min_box_wh = min_box_wh
        self.steps_per_epoch = int(steps_per_epoch)
        self.use_alb = bool(use_alb) and _has_alb
        self.cache_images = bool(cache_images)
        self.cache_size = int(cache_size)

        global _IMG_LRU_CAP
        _IMG_LRU_CAP = self.cache_size

        with open(data_yaml) as f:
            y = yaml.safe_load(f)
        root = Path(y.get("path", "."))
        img_dir = root / (
            y["train"] if isinstance(y["train"], str) else y["train"]["images"]
        )
        lbl_dir = root / (
            y["train"].replace("images", "labels")
            if isinstance(y["train"], str)
            else y["train"]["labels"]
        )
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        self.items: list[tuple[Path, Path]] = []
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in exts:
                self.items.append((p, lbl_dir / (p.stem + ".txt")))

        # мета (H,W,boxes) кэшируется
        self.meta: dict[Path, Meta] = {}
        for img_path, lbl_path in self.items:
            H, W = read_hw_fast(img_path)
            yolo = self._read_yolo_fast(lbl_path)
            boxes_abs = xywhn_to_xyxy_abs(yolo[:, 1:5], W, H)
            self.meta[img_path] = Meta(H=int(H), W=int(W), boxes_abs=boxes_abs)

        # Albumentations pipeline (CPU) — bbox-aware
        self.alb = self._build_albumentations(self.imgsz) if self.use_alb else None

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _read_yolo_fast(self, lbl: Path) -> np.ndarray:
        if not lbl.exists() or lbl.stat().st_size == 0:
            return np.zeros((0, 5), np.float32)
        try:
            lines = [
                ln.strip() for ln in lbl.read_text().strip().splitlines() if ln.strip()
            ]
            if not lines:
                return np.zeros((0, 5), np.float32)
            rows = []
            for ln in lines:
                parts = ln.replace("\t", " ").split()
                if len(parts) < 5:
                    continue
                try:
                    cls = float(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    rows.append([cls, x, y, w, h])
                except Exception:
                    continue
            if not rows:
                return np.zeros((0, 5), np.float32)
            arr = np.array(rows, dtype=np.float32)
            # sanitize into [0,1]
            x, y, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            x1 = np.clip(x - w / 2, 0, 1)
            y1 = np.clip(y - h / 2, 0, 1)
            x2 = np.clip(x + w / 2, 0, 1)
            y2 = np.clip(y + h / 2, 0, 1)
            w2 = np.maximum(0, x2 - x1)
            h2 = np.maximum(0, y2 - y1)
            keep = (w2 > 1e-6) & (h2 > 1e-6)
            if not np.any(keep):
                return np.zeros((0, 5), np.float32)
            xc = x1 + w2 / 2
            yc = y1 + h2 / 2
            return np.stack([np.zeros_like(xc), xc, yc, w2, h2], 1).astype(np.float32)
        except Exception:
            return np.zeros((0, 5), np.float32)

    def _build_albumentations(self, imgsz: int):
        # аккуратно с маленькими объектами: ограниченные повороты/масштабы, min_visibility=0.1
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(0.1, 0.1),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                        A.HueSaturationValue(
                            hue_shift_limit=4, sat_shift_limit=8, val_shift_limit=6
                        ),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3),
                        A.GaussianBlur(blur_limit=3),
                    ],
                    p=0.2,
                ),
                A.Affine(
                    scale=(0.9, 1.2),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-12, 12),
                    interpolation=cv2.INTER_LINEAR,
                    p=0.35,
                ),
                A.Resize(imgsz, imgsz, interpolation=cv2.INTER_AREA),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.10
            ),
        )

    def _pos_tiles_for_image(
        self, h: int, w: int, boxes_abs: np.ndarray, xs: np.ndarray, ys: np.ndarray
    ) -> list[tuple[int, int]]:
        tile = self.tile
        pos: set[tuple[int, int]] = set()
        # центр в тайле + гарант-тайл
        for b in boxes_abs:
            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            ix = np.where((xs <= cx) & (xs + tile > cx))[0]
            iy = np.where((ys <= cy) & (ys + tile > cy))[0]
            for i in ix:
                for j in iy:
                    pos.add((int(xs[i]), int(ys[j])))
        if self.cover_thr <= 0.0 or len(boxes_abs) == 0:
            return list(pos)
        thr = float(self.cover_thr)
        for b in boxes_abs:
            bx1, by1, bx2, by2 = map(float, b.tolist())
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            area_b = bw * bh
            if area_b <= 0.0:
                continue
            cand_ix = np.where((xs <= bx2) & (xs + tile >= bx1))[0]
            cand_iy = np.where((ys <= by2) & (ys + tile >= by1))[0]
            if cand_ix.size == 0 or cand_iy.size == 0:
                continue
            x0 = xs[cand_ix].astype(np.float32)
            y0 = ys[cand_iy].astype(np.float32)
            ox = np.minimum(x0[:, None] + tile, bx2) - np.maximum(x0[:, None], bx1)
            oy = np.minimum(y0[None, :] + tile, by2) - np.maximum(y0[None, :], by1)
            ox = np.clip(ox, 0.0, tile)
            oy = np.clip(oy, 0.0, tile)
            inter = ox * oy
            cover = inter / max(area_b, 1e-6)
            mask = cover >= thr
            if mask.any():
                ii, jj = np.where(mask)
                for a, b_ in zip(ii, jj, strict=False):
                    pos.add((int(x0[a]), int(y0[b_])))
        return list(pos)

    def _yield_from_image(
        self, img_path: Path, meta: Meta, rng: np.random.Generator
    ) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        global _IMG_LRU
        if _IMG_LRU is None and self.cache_images:
            _IMG_LRU = _ImageLRU(_IMG_LRU_CAP)

        h, w, boxes_abs = meta.H, meta.W, meta.boxes_abs

        # random phase per epoch
        offx = int(rng.integers(0, self.stride))
        offy = int(rng.integers(0, self.stride))
        xs, ys = tile_grid(h, w, self.tile, self.stride, offx, offy)

        pos_tiles = self._pos_tiles_for_image(h, w, boxes_abs, xs, ys)
        pos_set = set(pos_tiles)
        # негативы — из комплемента
        all_tiles = list(product(xs.tolist(), ys.tolist()))
        neg_tiles = [t for t in all_tiles if t not in pos_set]

        n_pos = len(pos_tiles)
        n_neg = min(
            len(neg_tiles),
            max(
                self.max_neg_per_image,
                int(n_pos * (1 - self.pos_frac) / max(self.pos_frac, 1e-6)),
            ),
        )
        if n_neg > 0:
            neg_tiles = random.sample(neg_tiles, n_neg)
        chosen = pos_tiles + neg_tiles
        rng.shuffle(chosen)

        # декод целой картинки (с кэшем) один раз
        img_full = (
            _IMG_LRU.get(img_path)
            if self.cache_images and _IMG_LRU is not None
            else load_image_fast(img_path)
        )

        for x0, y0 in chosen:
            tile = img_full[
                y0 : y0 + self.tile, x0 : x0 + self.tile, :
            ].copy()  # RGB uint8
            ht, wt = tile.shape[:2]

            # boxes пересчитанные в координаты тайла
            b = boxes_abs
            if len(b):
                cx = 0.5 * (b[:, 0] + b[:, 2])
                cy = 0.5 * (b[:, 1] + b[:, 3])
                m = (
                    (cx >= x0)
                    & (cx < x0 + self.tile)
                    & (cy >= y0)
                    & (cy < y0 + self.tile)
                )
                b = b[m]
            if len(b):
                b = b.copy()
                b[:, [0, 2]] -= x0
                b[:, [1, 3]] -= y0
                b[:, 0::2] = np.clip(b[:, 0::2], 0, wt)
                b[:, 1::2] = np.clip(b[:, 1::2], 0, ht)
                w = b[:, 2] - b[:, 0]
                h = b[:, 3] - b[:, 1]
                keep = (w >= self.min_box_wh) & (h >= self.min_box_wh)
                b = b[keep]
            else:
                b = np.zeros((0, 4), np.float32)

            if self.use_alb and self.alb is not None:
                # Albumentations ожидает BGR; мы храним RGB
                tile_bgr = tile[..., ::-1].copy()
                labels = [1] * len(b)
                try:
                    out = self.alb(
                        image=tile_bgr,
                        bboxes=[tuple(map(float, bb)) for bb in b],
                        labels=labels,
                    )
                    tile_aug_bgr = out["image"]
                    b_aug = (
                        np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
                        if len(out["bboxes"])
                        else np.zeros((0, 4), np.float32)
                    )
                    tile = tile_aug_bgr[..., ::-1].copy()  # back to RGB
                    b = b_aug
                    ht, wt = tile.shape[:2]
                except Exception:
                    # на всякий случай — без аугмент
                    pass
            # если Alb отключён — просто финальный resize до imgsz через Pillow
            elif (ht != self.imgsz) or (wt != self.imgsz):
                tile = np.array(
                    Image.fromarray(tile).resize(
                        (self.imgsz, self.imgsz), resample=Image.Resampling.BILINEAR
                    )
                )
                # рескейлим боксы
                sx = self.imgsz / float(wt)
                sy = self.imgsz / float(ht)
                if len(b):
                    b[:, [0, 2]] *= sx
                    b[:, [1, 3]] *= sy
                ht, wt = self.imgsz, self.imgsz

            # to tensors
            img_t = (
                torch.from_numpy(tile.copy()).permute(2, 0, 1).contiguous()
            )  # CHW uint8
            boxes_t = (
                torch.from_numpy(b)
                if len(b)
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            labels_t = torch.ones((len(boxes_t),), dtype=torch.long)

            # wrap for compatibility with torchvision detection
            img_wrap = TvImage(img_t)  # keep dtype uint8
            boxes_wrap = BoundingBoxes(boxes_t, format="XYXY", canvas_size=(ht, wt))

            # финальная защитная отсечка малых боксов
            bb = boxes_wrap.as_subclass(torch.Tensor).float()
            if bb.numel():
                w = bb[:, 2] - bb[:, 0]
                h = bb[:, 3] - bb[:, 1]
                keep = (w >= self.min_box_wh) & (h >= self.min_box_wh)
                bb = bb[keep]
                labels_t = labels_t[keep]
            else:
                bb = torch.zeros((0, 4), dtype=torch.float32)

            target = {
                "boxes": bb,
                "labels": labels_t,
                "image_id": torch.tensor([0]),
                "iscrowd": torch.zeros((len(bb),), dtype=torch.int64),
            }
            yield img_wrap.contiguous(), target

    def __iter__(self) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        # DDP rank/world
        rank = ddp_rank()
        world = ddp_world()
        # worker shard
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        # раздаём изображения: сначала по DDP-рангам, затем по воркерам (stride)
        imgs = [p for p, _ in self.items]
        imgs = imgs[rank::world] if world > 1 else imgs
        imgs = imgs[worker_id::num_workers] if num_workers > 1 else imgs
        if not imgs:
            return iter(())

        rng = np.random.default_rng(
            12345 + self._epoch * 999 + rank * 37 + worker_id * 17
        )

        steps_left = self.steps_per_epoch
        # цикл по изображениям — кольцевой
        for img_path in cycle(imgs):
            meta = self.meta[img_path]
            for sample in self._yield_from_image(img_path, meta, rng):
                yield sample
                steps_left -= 1
                if steps_left <= 0:
                    return


# Map-style валидационный датасет (позитивные тайлы)
class ValTileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_yaml: str,
        imgsz: int = 1024,
        tile: int = 1024,
        stride: int = 512,
        cover_thr: float = 0.30,
        min_box_wh: float = 2.0,
        cache_images: bool = True,
        cache_size: int = 128,
    ) -> None:
        super().__init__()
        self.imgsz = imgsz
        self.tile = tile
        self.stride = stride
        self.cover_thr = cover_thr
        self.min_box_wh = min_box_wh
        self.cache_images = bool(cache_images)
        self.cache_size = int(cache_size)

        global _IMG_LRU_CAP
        _IMG_LRU_CAP = max(_IMG_LRU_CAP, self.cache_size)

        with open(data_yaml) as f:
            y = yaml.safe_load(f)
        root = Path(y.get("path", "."))
        img_dir = root / (y["val"] if isinstance(y["val"], str) else y["val"]["images"])
        lbl_dir = root / (
            y["val"].replace("images", "labels")
            if isinstance(y["val"], str)
            else y["val"]["labels"]
        )
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        items: list[tuple[Path, Path]] = []
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in exts:
                items.append((p, lbl_dir / (p.stem + ".txt")))

        self.samples: list[tuple[Path, int, int, int, int]] = []  # (img, H,W,x0,y0)
        for img_path, lbl_path in items:
            H, W = read_hw_fast(img_path)
            yolo = self._read_yolo_fast(lbl_path)
            boxes_abs = xywhn_to_xyxy_abs(yolo[:, 1:5], W, H)

            xs, ys = tile_grid(H, W, self.tile, self.stride, 0, 0)
            # только позитивные тайлы
            pos = set()
            for b in boxes_abs:
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
                ix = np.where((xs <= cx) & (xs + self.tile > cx))[0]
                iy = np.where((ys <= cy) & (ys + self.tile > cy))[0]
                for i in ix:
                    for j in iy:
                        pos.add((int(xs[i]), int(ys[j])))
            for x0, y0 in pos:
                self.samples.append((img_path, H, W, x0, y0))

        self.alb = None  # без аугментаций

    def _read_yolo_fast(self, lbl: Path) -> np.ndarray:
        if not lbl.exists() or lbl.stat().st_size == 0:
            return np.zeros((0, 5), np.float32)
        try:
            lines = [
                ln.strip() for ln in lbl.read_text().strip().splitlines() if ln.strip()
            ]
            if not lines:
                return np.zeros((0, 5), np.float32)
            rows = []
            for ln in lines:
                parts = ln.replace("\t", " ").split()
                if len(parts) < 5:
                    continue
                try:
                    cls = float(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    rows.append([cls, x, y, w, h])
                except Exception:
                    continue
            if not rows:
                return np.zeros((0, 5), np.float32)
            arr = np.array(rows, dtype=np.float32)
            x, y, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            x1 = np.clip(x - w / 2, 0, 1)
            y1 = np.clip(y - h / 2, 0, 1)
            x2 = np.clip(x + w / 2, 0, 1)
            y2 = np.clip(y + h / 2, 0, 1)
            w2 = np.maximum(0, x2 - x1)
            h2 = np.maximum(0, y2 - y1)
            keep = (w2 > 1e-6) & (h2 > 1e-6)
            if not np.any(keep):
                return np.zeros((0, 5), np.float32)
            xc = x1 + w2 / 2
            yc = y1 + h2 / 2
            return np.stack([np.zeros_like(xc), xc, yc, w2, h2], 1).astype(np.float32)
        except Exception:
            return np.zeros((0, 5), np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        global _IMG_LRU
        if _IMG_LRU is None and self.cache_images:
            _IMG_LRU = _ImageLRU(_IMG_LRU_CAP)
        img_path, H, W, x0, y0 = self.samples[i]
        img_full = (
            _IMG_LRU.get(img_path)
            if self.cache_images and _IMG_LRU is not None
            else load_image_fast(img_path)
        )
        tile = img_full[y0 : y0 + self.tile, x0 : x0 + self.tile, :].copy()
        ht, wt = tile.shape[:2]

        # resize до imgsz без аугментаций
        if (ht != self.imgsz) or (wt != self.imgsz):
            tile = np.array(
                Image.fromarray(tile).resize(
                    (self.imgsz, self.imgsz), resample=Image.Resampling.BILINEAR
                )
            ).copy()
            wt = ht = self.imgsz

        # боксы — только центр-в-тайле
        boxes = np.zeros((0, 4), np.float32)

        img_t = torch.from_numpy(tile).permute(2, 0, 1).contiguous()
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([i]),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
        return TvImage(img_t), target


# ============ model ============
def create_model(
    num_classes: int = 2,
    rpn_train_topn=3500,
    rpn_post_train=1750,
    rpn_test_topn=3000,
    rpn_post_test=1500,
    fix_transform_size: int = 1280,
    weights: str = "DEFAULT",
):
    tv_weights = None
    if isinstance(weights, str) and weights.upper() == "DEFAULT":
        tv_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=tv_weights)

    # фикс resize внутри torchvision (compile-friendly: одна форма)
    try:
        model.transform.fixed_size = (int(fix_transform_size), int(fix_transform_size))
    except Exception:
        # fallback: clamp min/max
        model.transform.min_size = (int(fix_transform_size),)
        model.transform.max_size = int(fix_transform_size)

    def _torch_choice_stub(self, k) -> int:  # k — tuple размеров
        return 0

    model.transform.torch_choice = types.MethodType(_torch_choice_stub, model.transform)

    # anchors и RPN под малые цели
    sizes = ((4,), (8,), (16,), (32,), (64,))
    aspects = ((0.3, 0.5, 1.0, 2.0),) * 5
    anchor_gen = AnchorGenerator(sizes, aspects)
    in_channels = model.backbone.out_channels
    rpn_head = RPNHead(in_channels, anchor_gen.num_anchors_per_location()[0])
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_gen,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": int(rpn_train_topn), "testing": int(rpn_test_topn)},
        post_nms_top_n={"training": int(rpn_post_train), "testing": int(rpn_post_test)},
        nms_thresh=0.7,
        score_thresh=0.0,
    )
    model.rpn = rpn

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 600
    model.roi_heads.batch_size_per_image = 512

    model = model.to(memory_format=torch.channels_last)
    return model


# ============ evaluation utils ============
@torch.no_grad()
def run_tiled_inference(
    model: torch.nn.Module,
    device: torch.device,
    img_np: np.ndarray,  # HWC uint8 RGB
    tile: int,
    stride: int,
    batch_size: int,
    score_thr: float,
    nms_thr: float,
    max_dets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    H, W = img_np.shape[:2]
    xs, ys = tile_grid(H, W, tile, stride, 0, 0)

    tiles: list[torch.Tensor] = []
    offsets: list[tuple[int, int]] = []
    for y0 in ys.tolist():
        for x0 in xs.tolist():
            t = img_np[y0 : y0 + tile, x0 : x0 + tile, :]
            t_t = torch.from_numpy(t).permute(2, 0, 1).contiguous().float().div_(255.0)
            tiles.append(t_t)
            offsets.append((x0, y0))

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    model.eval()
    for i in range(0, len(tiles), max(1, batch_size)):
        batch = [
            im.to(device, non_blocking=True).contiguous()
            for im in tiles[i : i + batch_size]
        ]
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            preds = model(batch)
        for j, p in enumerate(preds):
            boxes = p["boxes"].detach().float().to("cpu")
            scores = p["scores"].detach().float().to("cpu")
            x0, y0 = offsets[i + j]
            if boxes.numel():
                boxes[:, [0, 2]] += float(x0)
                boxes[:, [1, 3]] += float(y0)
            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))
    boxes = torch.cat(all_boxes, 0)
    scores = torch.cat(all_scores, 0)
    if boxes.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))

    # score filter
    keep = scores >= float(score_thr)
    if keep.any():
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        boxes = boxes[:0]
        scores = scores[:0]

    if boxes.numel():
        keep_idx = nms(boxes, scores, float(nms_thr))
        keep_idx = keep_idx[:max_dets]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
    return boxes, scores


def compute_fbeta(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    beta2 = beta * beta
    denom = (1.0 + beta2) * tp + beta2 * fn + fp
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * tp / denom


def greedy_match_by_iou(
    iou_mat: torch.Tensor, scores: torch.Tensor, thr: float
) -> tuple[int, int, int]:
    # iou_mat: [N_pred, N_gt]
    if iou_mat.numel() == 0:
        n_pred = iou_mat.shape[0]
        n_gt = iou_mat.shape[1]
        tp = 0
        fp = n_pred
        fn = n_gt
        return tp, fp, fn
    Np, Ng = iou_mat.shape
    order = torch.argsort(scores, descending=True)
    gt_used = torch.zeros((Ng,), dtype=torch.bool)
    tp = 0
    for idx in order.tolist():
        if idx >= Np:
            continue
        ious = iou_mat[idx]
        # find best GT not used
        best_iou = 0.0
        best_j = -1
        for j in range(Ng):
            if gt_used[j]:
                continue
            v = float(ious[j])
            if v >= thr and v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
    fp = Np - tp
    fn = int((~gt_used).sum().item())
    return tp, fp, fn


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    device: torch.device,
    data_yaml: str,
    imgsz: int,
    tile: int,
    stride: int,
    batch_size: int,
    score_thr_candidates: list[float],
    nms_thr: float,
    max_dets: int,
    beta: float,
    iou_min: float,
    iou_max: float,
    iou_step: float,
    cache_images: bool = True,
    cache_size: int = 128,
) -> tuple[float, float, dict]:
    with open(data_yaml) as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", "."))
    img_dir = root / (y["val"] if isinstance(y["val"], str) else y["val"]["images"])
    lbl_dir = root / (
        y["val"].replace("images", "labels")
        if isinstance(y["val"], str)
        else y["val"]["labels"]
    )
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    items: list[tuple[Path, Path]] = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in exts:
            items.append((p, lbl_dir / (p.stem + ".txt")))

    # local cache
    img_cache = _ImageLRU(capacity=cache_size) if cache_images else None

    # thresholds list
    ious = []
    t = iou_min
    while t <= (iou_max + 1e-9):
        ious.append(round(t, 5))
        t += iou_step

    # accumulators: for each thr, we will collect tp/fp/fn per score_thr candidate
    stats: dict[float, dict[float, dict[str, int]]] = {
        st: {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in ious}
        for st in score_thr_candidates
    }

    for img_path, lbl_path in items:
        H, W = read_hw_fast(img_path)
        # read gt boxes
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            lines = [
                ln.strip()
                for ln in lbl_path.read_text().strip().splitlines()
                if ln.strip()
            ]
            gt_rows = []
            for ln in lines:
                parts = ln.replace("\t", " ").split()
                if len(parts) < 5:
                    continue
                try:
                    cls = float(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    gt_rows.append([cls, x, y, w, h])
                except Exception:
                    continue
            if gt_rows:
                arr = np.array(gt_rows, dtype=np.float32)
                gt_boxes = xywhn_to_xyxy_abs(arr[:, 1:5], W, H)
            else:
                gt_boxes = np.zeros((0, 4), np.float32)
        else:
            gt_boxes = np.zeros((0, 4), np.float32)

        # get image
        img_np = (
            img_cache.get(img_path)
            if img_cache is not None
            else load_image_fast(img_path)
        )

        # predictions for a single best candidate threshold are not reused across thresholds of score; we'll compute once and then filter
        # so run with min score to get all candidates
        min_st = float(min(score_thr_candidates)) if score_thr_candidates else 0.0
        pred_boxes_all, pred_scores_all = run_tiled_inference(
            model=model,
            device=device,
            img_np=img_np,
            tile=tile,
            stride=stride,
            batch_size=batch_size,
            score_thr=min_st,
            nms_thr=nms_thr,
            max_dets=max_dets,
        )

        # compute once iou matrix with GT
        gt_t = torch.from_numpy(gt_boxes).float()
        # for each score threshold candidate, filter and compute stats per IoU threshold
        for st in score_thr_candidates:
            if pred_boxes_all.numel() == 0:
                # no predictions at all
                for thr in ious:
                    stats[st][thr]["tp"] += 0
                    stats[st][thr]["fp"] += 0
                    stats[st][thr]["fn"] += gt_t.shape[0]
                continue
            keep = pred_scores_all >= float(st)
            pb = pred_boxes_all[keep] if keep.any() else pred_boxes_all[:0]
            ps = pred_scores_all[keep] if keep.any() else pred_scores_all[:0]
            iou_mat = (
                box_iou(pb, gt_t)
                if gt_t.numel() and pb.numel()
                else torch.zeros((pb.shape[0], gt_t.shape[0]))
            )
            for thr in ious:
                tp, fp, fn = greedy_match_by_iou(iou_mat, ps, float(thr))
                stats[st][thr]["tp"] += tp
                stats[st][thr]["fp"] += fp
                stats[st][thr]["fn"] += fn

    # aggregate and choose best score threshold
    best_score = -1.0
    best_thr = score_thr_candidates[0] if score_thr_candidates else 0.0
    per_thr_scores: dict[float, float] = {}
    for st, by_thr in stats.items():
        thr_scores = []
        for thr, c in by_thr.items():
            f = compute_fbeta(c["tp"], c["fp"], c["fn"], beta=beta)
            thr_scores.append(f)
        mean_f = float(np.mean(thr_scores)) if thr_scores else 0.0
        per_thr_scores[st] = mean_f
        if mean_f > best_score:
            best_score = mean_f
            best_thr = st

    return best_score, best_thr, per_thr_scores


# ============ CUDA prefetcher ============
class CUDAPrefetcher:
    def __init__(
        self, loader: DataLoader, device: torch.device, amp_dtype: str, uint8_cpu: bool
    ) -> None:
        self.loader = iter(loader)
        self.device = device
        self.amp_dtype = amp_dtype
        self.uint8_cpu = uint8_cpu
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.next_batch = None
        self.preload()

    def preload(self) -> None:
        try:
            images, targets = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream:
            with torch.cuda.stream(self.stream):
                if self.uint8_cpu:
                    dt = (
                        torch.bfloat16
                        if self.amp_dtype == "bf16"
                        else torch.float16
                        if self.amp_dtype == "fp16"
                        else torch.float32
                    )
                    images = [
                        img.to(device=self.device, non_blocking=True, dtype=dt)
                        .div_(255.0)
                        .contiguous()
                        for img in images
                    ]
                else:
                    images = [
                        img.to(device=self.device, non_blocking=True).contiguous()
                        for img in images
                    ]
                targets = [
                    {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                    for t in targets
                ]
        self.next_batch = (images, targets)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


# ============ train loop ============
def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    imgs, targets = list(zip(*batch, strict=False))
    return list(imgs), list(targets)


def cosine_lr(
    it: int, total: int, base_lr: float, final_lr: float = 1e-6, warmup: int = 800
) -> float:
    if it < warmup:
        return base_lr * (it / max(1, warmup))
    t = (it - warmup) / max(1, total - warmup)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))


def train(args) -> None:
    set_perf()
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DDP
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(ddp_rank())

    # datasets
    train_ds = TileIterableDataset(
        args.data,
        split="train",
        imgsz=args.imgsz,
        tile=args.tile,
        stride=args.stride,
        pos_frac=args.pos_frac,
        max_neg_per_image=args.max_neg,
        cover_thr=args.cover_thr,
        min_box_wh=2.0,
        steps_per_epoch=args.steps_per_epoch,
        use_alb=bool(args.use_alb),
        cache_images=bool(args.cache_images),
        cache_size=args.cache_size,
    )

    pin_kwargs = {"pin_memory": bool(args.pin_memory)}
    if bool(args.pin_memory) and (
        "pin_memory_device" in DataLoader.__init__.__code__.co_varnames
    ):
        pin_kwargs["pin_memory_device"] = "cuda"

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=False,  # IterableDataset нельзя шифлить
        num_workers=min(args.workers, 64),
        prefetch_factor=max(1, args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        collate_fn=collate_fn,
        drop_last=True,
        multiprocessing_context=(
            __import__("multiprocessing").get_context(args.mp_context)
            if args.mp_context
            else None
        ),
        **pin_kwargs,
    )

    # модель
    model = create_model(
        num_classes=2,
        rpn_train_topn=args.rpn_train_topn,
        rpn_post_train=args.rpn_post_train,
        rpn_test_topn=args.rpn_test_topn,
        rpn_post_test=args.rpn_post_test,
        fix_transform_size=args.imgsz,
        weights=args.weights,
    ).to(device)

    # optional checkpoint load (resume or provided state dict)
    if (
        isinstance(args.weights, str)
        and args.weights not in ("DEFAULT", "")
        and os.path.exists(args.weights)
    ):
        ckpt = torch.load(args.weights, map_location="cpu")
        try:
            state = (
                ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            if ddp_rank() == 0:
                pass
        except Exception:
            if ddp_rank() == 0:
                pass

    # compile (опционально)
    if args.compile and (dynamo is not None):
        try:
            dynamo.config.capture_scalar_outputs = True
            model = torch.compile(
                model, mode="max-autotune", fullgraph=False, dynamic=True
            )
        except Exception:
            if ddp_rank() == 0:
                pass

    if is_ddp():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_rank()], find_unused_parameters=False
        )

    # оптимизатор
    opt_kwargs = {"lr": args.lr, "weight_decay": 0.01, "betas": (0.9, 0.999)}
    try:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], **opt_kwargs, fused=True
        )
    except TypeError:
        try:
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                **opt_kwargs,
                foreach=True,
            )
        except TypeError:
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], **opt_kwargs
            )

    amp_dtype = str(args.amp_dtype).lower()
    use_amp = amp_dtype in ("fp16", "bf16")
    autocast_dtype = (
        torch.float16
        if amp_dtype == "fp16"
        else torch.bfloat16
        if amp_dtype == "bf16"
        else torch.float32
    )
    scaler = GradScaler("cuda", enabled=(amp_dtype == "fp16"))

    total_iters = args.steps_per_epoch * args.epochs

    if ddp_rank() == 0:
        exp_dir = Path(
            args.out
            or f"runs/frcnn_uav_v11/exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
    else:
        exp_dir = log_path = None

    best_val = -1.0
    global_step = 0
    last_empty = time.time()

    for ep in range(args.epochs):
        # сообщим датасету номер эпохи (для новой фазы тайлов)
        train_ds.set_epoch(ep)

        if ddp_rank() == 0:
            pass

        model.train()
        pbar = tqdm(total=args.steps_per_epoch, disable=(ddp_rank() != 0), ncols=120)
        running = 0.0

        prefetch = (
            CUDAPrefetcher(train_loader, device, amp_dtype, bool(args.uint8_cpu))
            if device.type == "cuda"
            else iter(train_loader)
        )

        for _ in range(args.steps_per_epoch):
            images, targets = next(prefetch)

            lr = cosine_lr(
                global_step, total_iters, args.lr, final_lr=args.lr * 0.05, warmup=800
            )
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            if amp_dtype == "fp16":
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            loss_val = float(loss.detach())
            running += loss_val
            if ddp_rank() == 0:
                pbar.set_postfix(loss=f"{loss_val:.3f}", lr=f"{lr:.2e}")
                pbar.update(1)

            if (time.time() - last_empty) > 45:
                torch.cuda.empty_cache()
                gc.collect()
                last_empty = time.time()

            del loss, loss_dict
            global_step += 1

        pbar.close()

        # Validation and metric evaluation (F-beta across IoU thresholds)
        if ddp_rank() == 0:
            train_loss = running / max(1, args.steps_per_epoch)
            eval_score = None
            chosen_thr = None
            per_thr_scores = None
            if args.eval_every and ((ep + 1) % args.eval_every == 0):
                st_cands = [
                    float(x)
                    for x in str(args.eval_score_thr_candidates).split(",")
                    if str(x).strip()
                ]
                if not st_cands:
                    st_cands = [0.05, 0.10, 0.15, 0.20]
                eval_score, chosen_thr, per_thr_scores = evaluate_dataset(
                    model=(
                        model.module
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model
                    ),
                    device=device,
                    data_yaml=args.data,
                    imgsz=args.imgsz,
                    tile=args.val_tile,
                    stride=args.val_stride,
                    batch_size=args.eval_batch,
                    score_thr_candidates=st_cands,
                    nms_thr=args.eval_nms,
                    max_dets=args.eval_max_dets,
                    beta=args.beta,
                    iou_min=args.iou_min,
                    iou_max=args.iou_max,
                    iou_step=args.iou_step,
                    cache_images=bool(args.cache_images),
                    cache_size=args.cache_size,
                )
                # save best
                if eval_score is not None and eval_score > best_val:
                    best_val = eval_score
                    exp_p = exp_dir if isinstance(exp_dir, Path) else Path(str(exp_dir))
                    ckpt_path = exp_p / "best.pt"
                    state = (
                        model.module
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model
                    ).state_dict()
                    torch.save(
                        {
                            "model": state,
                            "epoch": ep,
                            "train_loss": train_loss,
                            "fbeta": eval_score,
                            "best_score_thr": chosen_thr,
                        },
                        ckpt_path,
                    )
                    str(ckpt_path)
            # log
            log_obj = {
                "epoch": ep,
                "train_loss": float(train_loss),
                "time": time.time(),
            }
            if eval_score is not None:
                log_obj.update(
                    {
                        "fbeta": float(eval_score),
                        "best_score_thr": float(chosen_thr)
                        if chosen_thr is not None
                        else None,
                        "per_thr_scores": per_thr_scores,
                    }
                )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_obj) + "\n")
            if eval_score is not None:
                pass
            else:
                pass

    if is_ddp():
        dist.barrier()
    if ddp_rank() == 0:
        if best_val >= 0:
            pass
        else:
            pass


# ============ CLI ============
def parse_args():
    ap = argparse.ArgumentParser("Faster R-CNN UAV Training — v11 (iter+alb)")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument(
        "--steps-per-epoch", type=int, default=3000, help="Шагов на эпоху (per-rank)"
    )
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--pos-frac", type=float, default=0.9)
    ap.add_argument("--max-neg", type=int, default=4)
    ap.add_argument("--cover-thr", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--persistent-workers", type=int, default=1)
    ap.add_argument("--pin-memory", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.6e-3)
    ap.add_argument("--out", type=str, default="")
    # AMP / compile
    ap.add_argument(
        "--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"]
    )
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--multiscale", type=int, default=0)
    # RPN top-N
    ap.add_argument("--rpn-train-topn", type=int, default=3500)
    ap.add_argument("--rpn-post-train", type=int, default=1750)
    ap.add_argument("--rpn-test-topn", type=int, default=3000)
    ap.add_argument("--rpn-post-test", type=int, default=1500)
    # memory/cpu
    ap.add_argument("--uint8-cpu", type=int, default=1)
    ap.add_argument(
        "--mp-context",
        type=str,
        default="fork",
        choices=["spawn", "fork", "forkserver"],
    )
    # weights
    ap.add_argument(
        "--weights",
        type=str,
        default="DEFAULT",
        help='"DEFAULT" (torchvision) или путь к .pth/.pt (локально)',
    )
    # caching
    ap.add_argument("--cache-images", type=int, default=1)
    ap.add_argument(
        "--cache-size",
        type=int,
        default=256,
        help="LRU на worker: кол-во целых изображений в кэше",
    )
    # albumentations
    ap.add_argument(
        "--use-alb",
        type=int,
        default=1,
        help="1=Albumentations, 0=без аугментаций (только resize)",
    )
    # evaluation
    ap.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Как часто валидировать (в эпохах); 0=не валидировать",
    )
    ap.add_argument("--val-tile", type=int, default=1024)
    ap.add_argument("--val-stride", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=24)
    ap.add_argument("--eval-max-dets", type=int, default=1000)
    ap.add_argument("--eval-nms", type=float, default=0.50)
    ap.add_argument(
        "--eval-score-thr-candidates", type=str, default="0.05,0.10,0.15,0.20"
    )
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--iou-min", type=float, default=0.30)
    ap.add_argument("--iou-max", type=float, default=0.93)
    ap.add_argument("--iou-step", type=float, default=0.07)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
