import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchvision.ops import nms as tv_nms  # быстрый NMS из torchvision
from ultralytics import YOLO

# =============================== ЛОГИ/ДЕБАГ
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("uav-tiling")

# =============================== Torch perf
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =============================== Модель
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
device_str = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device_str).eval()

try:
    model.model.to(memory_format=torch.channels_last)
    log.info("Using channels_last memory format.")
except Exception as e:
    log.warning(f"channels_last not set: {e}")

USE_HALF = torch.cuda.is_available()
if USE_HALF:
    try:
        model.model.half()
        log.info("Using FP16 weights for inference.")
    except Exception as e:
        log.warning(f"FP16 not enabled: {e}")

# =============================== Параметры тайлинга/батчинга/коарса
TILE_SIZE: int = 1536  # 1024/1280/1536 — под VRAM/метрику
OVERLAP: float = 0.20  # 0.15–0.30
MIN_WH_PIX: int = 4
NMS_IOU: float = 0.55

MAX_TILES_PER_BATCH: int = 4  #
MAX_THREADS_CPU: int = 12  # CPU-параллель для нарезки и декода

USE_COARSE: bool = True
COARSE_SHORT_SIDE: int = 1280  # фикс‑квадрат (ниже letterbox) — важно для cudnn.benchmark
COARSE_CONF: float = 0.10
COARSE_EXPAND: float = 1.8
MAX_TILES: int = 64  # ограничение числа окон

# =============================== Постоянные пула потоков
_TILE_POOL = ThreadPoolExecutor(max_workers=MAX_THREADS_CPU)
_DECODE_POOL = ThreadPoolExecutor(max_workers=MAX_THREADS_CPU)


# =============================== GPU‑таймер
def _gpu_timer():
    if not torch.cuda.is_available():

        def _start():
            return time.perf_counter()

        def _stop(t0):
            return (time.perf_counter() - t0) * 1000.0

        return _start, _stop
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def _start():
        torch.cuda.synchronize()
        start_event.record()
        return start_event

    def _stop(_):
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)  # ms

    return _start, _stop


start_gpu, stop_gpu = _gpu_timer()


# =============================== Утилиты
def _clip(a, lo, hi):
    return max(lo, min(hi, a))


def _grid_windows(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    if ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    return [(x, y, x + tile, y + tile) for y in ys for x in xs]


def _nms_torchvision(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    tb = torch.as_tensor(boxes_xyxy, device="cpu", dtype=torch.float32)
    ts = torch.as_tensor(scores, device="cpu", dtype=torch.float32)
    keep = tv_nms(tb, ts, iou_thr).cpu().numpy().tolist()
    return keep


def _letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    """Мягкое приведение к квадрату size×size (фикс-форма для cudnn.benchmark)."""
    H, W = img.shape[:2]
    scale = size / max(H, W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    # resize bilinear (CPU)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(newH, newW), mode="bilinear", align_corners=False)
    small = t[0].permute(1, 2, 0).byte().numpy()
    # паддинг до size×size
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - newH) // 2
    x0 = (size - newW) // 2
    out[y0 : y0 + newH, x0 : x0 + newW, :] = small
    return out


# =============================== Coarse windows (с фикс‑формой)
def _coarse_windows(img: np.ndarray, short_side: int, conf: float, expand: float, limit: int):
    H, W = img.shape[:2]
    t0 = time.perf_counter()

    small = _letterbox_square(img, short_side)  # фикс 1280×1280 → нет ребенчмарка
    res = model.predict(
        source=[small],
        imgsz=short_side,
        conf=conf,
        iou=0.6,
        device=0,
        verbose=False,
    )

    wins: list[tuple[int, int, int, int]] = []
    for r in res:
        if r.boxes is None:
            continue
        # координаты в small (size×size), потом обратная проекция в исходный кадр
        xywhn = r.boxes.xywhn.detach().float().cpu().numpy()
        # матрица обратно из letterbox: small центрирован в квадрате
        # для size×size масштаб = size / max(H, W); отступы:
        size = short_side
        scale = size / max(H, W)
        pad_x = (size - int(round(W * scale))) // 2
        pad_y = (size - int(round(H * scale))) // 2

        for xc, yc, w, h in xywhn:
            # в пиксели small
            gx_s = xc * size
            gy_s = yc * size
            bw_s = w * size
            bh_s = h * size
            # убрать паддинг и масштаб в исходный
            gx = (gx_s - pad_x) / scale
            gy = (gy_s - pad_y) / scale
            bw = bw_s / scale
            bh = bh_s / scale

            x1 = gx - bw / 2
            y1 = gy - bh / 2
            x2 = gx + bw / 2
            y2 = gy + bh / 2
            # расширение
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ww = (x2 - x1) * expand
            hh = (y2 - y1) * expand
            x1 = _clip(cx - ww / 2, 0, W - 1)
            y1 = _clip(cy - hh / 2, 0, H - 1)
            x2 = _clip(cx + ww / 2, 0, W - 1)
            y2 = _clip(cy + hh / 2, 0, H - 1)
            # квадрат под TILE_SIZE
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x1 = _clip(cx - TILE_SIZE / 2, 0, W - TILE_SIZE)
            y1 = _clip(cy - TILE_SIZE / 2, 0, H - TILE_SIZE)
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            wins.append((int(x1), int(y1), int(x2), int(y2)))

    if wins:
        boxes = np.array(wins, dtype=np.float32)
        scores = np.ones(len(wins), dtype=np.float32)
        keep = _nms_torchvision(boxes, scores, 0.5)
        wins = [wins[i] for i in keep][:limit]

    t_ms = (time.perf_counter() - t0) * 1000.0
    log.debug(f"[coarse] windows={len(wins)} time={t_ms:.1f} ms")
    return wins


# =============================== Батч-инференс тайлов
def _infer_tiles_batch(tiles: list[np.ndarray], conf: float, iou: float, imgsz: int):
    # model.predict принимает список изображений и сам формирует батч
    # (официальный способ пакетного инференса в Ultralytics). :contentReference[oaicite:3]{index=3}
    return model.predict(source=tiles, imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False)


# =============================== Декод результатов тайла → глобальные px
def _decode_results_for_window(r, win, tile_wh, H, W, min_wh_pix):
    x1w, y1w, x2w, y2w = win
    tw, th = tile_wh
    out_xyxy, out_score = [], []
    if r.boxes is None:
        return out_xyxy, out_score
    xywhn = r.boxes.xywhn.detach().float().cpu().numpy()
    scores = r.boxes.conf.detach().float().cpu().numpy().reshape(-1)
    for (xc, yc, w, h), sc in zip(xywhn, scores):
        bw = w * tw
        bh = h * th
        gx = xc * tw + x1w
        gy = yc * th + y1w
        bx1 = _clip(gx - bw / 2, 0, W - 1)
        by1 = _clip(gy - bh / 2, 0, H - 1)
        bx2 = _clip(gx + bw / 2, 0, W - 1)
        by2 = _clip(gy + bh / 2, 0, H - 1)
        if (bx2 - bx1) < min_wh_pix or (by2 - by1) < min_wh_pix:
            continue
        out_xyxy.append([bx1, by1, bx2, by2])
        out_score.append(sc)
    return out_xyxy, out_score


# =============================== Warm‑up
def _warmup_once():
    try:
        img_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        img_coarse = np.zeros((COARSE_SHORT_SIDE, COARSE_SHORT_SIDE, 3), dtype=np.uint8)
        # два прогона на фикс-формах
        _ = model.predict(
            source=[img_tile, img_tile], imgsz=TILE_SIZE, conf=0.01, iou=0.5, device=0, verbose=False
        )
        _ = model.predict(
            source=[img_coarse], imgsz=COARSE_SHORT_SIDE, conf=0.01, iou=0.5, device=0, verbose=False
        )
        log.info("Warm-up done.")
    except Exception as e:
        log.warning(f"Warm-up skipped: {e}")


_warmup_once()  # выполняем при импорте


# =============================== Основной инференс 1 изображения (с детальным профилем)
def infer_image_bbox(
    image: np.ndarray,
    conf: float = 0.16,
    iou: float = 0.65,
    img_size: int = TILE_SIZE,
) -> list[dict]:
    t_all0 = time.perf_counter()
    H, W = image.shape[:2]

    # --- 1) Генерация окон (coarse или сетка)
    t0 = time.perf_counter()
    if USE_COARSE:
        windows = _coarse_windows(image, COARSE_SHORT_SIDE, COARSE_CONF, COARSE_EXPAND, MAX_TILES)
    else:
        windows = []
    if not windows:
        tile = min(TILE_SIZE, min(H, W))
        windows = _grid_windows(H, W, tile=tile, overlap=OVERLAP)
    t1 = time.perf_counter()
    windows_ms = (t1 - t0) * 1000.0

    # --- 2) Извлечение тайлов (CPU‑пул)
    t2 = time.perf_counter()
    # без .copy() — лишний memcpy, берём срезы напрямую
    futs = [_TILE_POOL.submit(lambda w: image[w[1] : w[3], w[0] : w[2], :], win) for win in windows]
    tiles = [f.result() for f in futs]
    t3 = time.perf_counter()
    tiles_ms = (t3 - t2) * 1000.0

    # --- 3) Батч‑инференс по тайлам (GPU‑тайминг)
    all_xyxy, all_scores = [], []
    gpu_total_ms, cpu_decode_ms = 0.0, 0.0
    num_batches = 0

    for b in range(0, len(tiles), MAX_TILES_PER_BATCH):
        batch_tiles = tiles[b : b + MAX_TILES_PER_BATCH]
        batch_wins = windows[b : b + MAX_TILES_PER_BATCH]

        g0 = start_gpu()
        res = _infer_tiles_batch(batch_tiles, conf=conf, iou=iou, imgsz=img_size)
        g_ms = stop_gpu(g0)
        gpu_total_ms += g_ms
        num_batches += 1

        c0 = time.perf_counter()
        futs = [
            _DECODE_POOL.submit(
                _decode_results_for_window, r, w, (t.shape[1], t.shape[0]), H, W, MIN_WH_PIX
            )
            for r, w, t in zip(res, batch_wins, batch_tiles)
        ]
        for f in futs:
            xyxy_list, sc_list = f.result()
            if xyxy_list:
                all_xyxy.extend(xyxy_list)
                all_scores.extend(sc_list)
        cpu_decode_ms += (time.perf_counter() - c0) * 1000.0

    # --- 4) Финальный NMS
    if not all_xyxy:
        t_all1 = time.perf_counter()
        log.info(
            f"[profile] windows={len(windows)} | tiles={len(tiles)} | "
            f"windows_ms={windows_ms:.1f} | tiles_ms={tiles_ms:.1f} | "
            f"gpu_infer={gpu_total_ms:.1f} | decode={cpu_decode_ms:.1f} | nms=0.0 | "
            f"total={(t_all1 - t_all0) * 1000:.1f} ms"
        )
        log.info(f"[profile-extra] batches={num_batches} batch_size={MAX_TILES_PER_BATCH}")
        return []

    n0 = time.perf_counter()
    boxes = np.asarray(all_xyxy, dtype=np.float32)
    scores = np.asarray(all_scores, dtype=np.float32)
    keep = _nms_torchvision(boxes, scores, NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    n1 = time.perf_counter()
    nms_ms = (n1 - n0) * 1000.0

    # --- 5) В формат соревнования (нормированные xywh)
    dets: list[dict] = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        bw = x2 - x1
        bh = y2 - y1
        if bw < MIN_WH_PIX or bh < MIN_WH_PIX:
            continue
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        dets.append(
            {
                "xc": float(xc / W),
                "yc": float(yc / H),
                "w": float(bw / W),
                "h": float(bh / H),
                "label": 0,
                "score": float(s),
            }
        )

    t_all1 = time.perf_counter()
    log.info(
        f"[profile] windows={len(windows)} | tiles={len(tiles)} | "
        f"windows_ms={windows_ms:.1f} | tiles_ms={tiles_ms:.1f} | "
        f"gpu_infer={gpu_total_ms:.1f} | decode={cpu_decode_ms:.1f} | nms={nms_ms:.1f} | "
        f"total={(t_all1 - t_all0) * 1000:.1f} ms"
    )
    log.info(f"[profile-extra] batches={num_batches} batch_size={MAX_TILES_PER_BATCH}")
    return dets


# =============================== Публичное API (как у тебя)
def predict(images: np.ndarray | list[np.ndarray]) -> list[list[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]
    return [infer_image_bbox(img) for img in images]
