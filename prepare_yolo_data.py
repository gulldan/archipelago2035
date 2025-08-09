#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Готовит YOLO-датасет (train/val) из папок Rescue-SAR, аккуратно:
- НИЧЕГО не пишет в source;
- Стратифицированный сплит по наличию объектов (pos/neg);
- Чинит аннотации (кламп [0,1], валидирует строки, класс -> 0);
- Пишет mapping.csv и dataset.yaml.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

# ---- НАСТРОЙКИ ----
DATASET_FOLDERS = [
    "01_train-s1__DataSet_Human_Rescue",
    "02_second_part_DataSet_Human_Rescue",
    "04_ladd",
    "05_pd",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXT = ".txt"

VAL_FRACTION = 0.20  # доля картинок в валидации
RANDOM_SEED = 42
SPLIT_BY_SCENE = (
    False  # если True — валид размечается по верхним папкам (минимизирует «слив» сцен)
)
ENSURE_MIN_VAL_POS = 50  # минимум позитивных картинок в val (если хватает)

# Фильтрация боксов на этапе подготовки (бережная)
CLAMP_TO_UNIT = True  # клампим xywhn в [0,1]
DROP_TINY_WH = 1e-6  # отбрасываем боксы с w/h <= этого порога (норм.)
FORCE_CLASS_ZERO = True  # все классы -> 0 (у нас 1 класс person)

# -------------------


@dataclass
class Item:
    img: Path
    lbl: Optional[Path]  # может отсутствовать
    has_pos: bool  # есть валидные боксы
    src_folder: str  # верхний уровень (для group split)


def find_image_files(root: Path) -> List[Path]:
    return [
        p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]


def guess_label_path(img_path: Path) -> Path:
    """
    Пытаемся найти txt-лейбл для картинки:
    - .../images/xxx.jpg -> .../labels/xxx.txt
    - или рядом (xxx.txt в той же папке)
    """
    # 1) если внутри .../images/... -> заменить на .../labels/...
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        parts[i] = "labels"
        cand = Path(*parts).with_suffix(LABEL_EXT)
        if cand.exists():
            return cand
    # 2) тот же каталог
    cand2 = img_path.with_suffix(LABEL_EXT)
    if cand2.exists():
        return cand2
    # 3) соседняя папка labels на том же уровне
    if "images" in parts:
        i = parts.index("images")
        alt = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(LABEL_EXT)
        if alt.exists():
            return alt
    return Path()  # пустой -> нет


def read_and_fix_labels(
    lbl_path: Path,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Считываем YOLO-разметку (class xc yc w h) и чиним:
    - пропускаем битые строки;
    - класс -> 0 (если FORCE_CLASS_ZERO);
    - клампим координаты в [0,1];
    - выкидываем нулевые/отрицательные боксы.
    Возвращаем список валидных записей.
    """
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return []

    valid = []
    with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c = float(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            if FORCE_CLASS_ZERO:
                c = 0.0

            if CLAMP_TO_UNIT:
                # Переводим в xyxy, клампим, обратно в xywh
                x1 = max(0.0, min(1.0, xc - w / 2))
                y1 = max(0.0, min(1.0, yc - h / 2))
                x2 = max(0.0, min(1.0, xc + w / 2))
                y2 = max(0.0, min(1.0, yc + h / 2))
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= DROP_TINY_WH or h <= DROP_TINY_WH:
                    continue
                xc = x1 + w / 2
                yc = y1 + h / 2
            else:
                if w <= DROP_TINY_WH or h <= DROP_TINY_WH:
                    continue

            valid.append((c, xc, yc, w, h))

    return valid


def collect_items(source_dir: str) -> List[Item]:
    src = Path(source_dir)
    items: List[Item] = []

    for folder in DATASET_FOLDERS:
        folder_path = src / folder
        if not folder_path.exists():
            print(f"⚠️  Папка не найдена: {folder_path}")
            continue
        top_name = folder_path.name

        print(f"📁 Сканируем {folder_path}")
        for img in tqdm(
            find_image_files(folder_path), desc=f"  ➜ images in {top_name}"
        ):
            lbl = guess_label_path(img)
            labels = read_and_fix_labels(lbl) if lbl and lbl.exists() else []
            items.append(
                Item(
                    img=img,
                    lbl=(lbl if lbl.exists() else None),
                    has_pos=(len(labels) > 0),
                    src_folder=top_name,
                )
            )
    return items


def stratified_split(
    items: List[Item],
    val_fraction: float,
    seed: int,
    split_by_scene: bool = False,
    ensure_min_val_pos: int = 0,
):
    """
    Делим на train/val:
    - стратификация по has_pos (позитив/негатив),
    - опция группировки по верхней папке (минимизирует утечку похожих сцен).
    """
    rng = random.Random(seed)

    if split_by_scene:
        # группируем по папкам; внутри каждой — стратифицированно
        train, val = [], []
        folders = {}
        for it in items:
            folders.setdefault(it.src_folder, []).append(it)
        for _, group in folders.items():
            pos = [it for it in group if it.has_pos]
            neg = [it for it in group if not it.has_pos]
            rng.shuffle(pos)
            rng.shuffle(neg)
            vp = int(round(len(pos) * val_fraction))
            vn = int(round(len(neg) * val_fraction))
            val.extend(pos[:vp] + neg[:vn])
            train.extend(pos[vp:] + neg[vn:])
    else:
        pos = [it for it in items if it.has_pos]
        neg = [it for it in items if not it.has_pos]
        rng.shuffle(pos)
        rng.shuffle(neg)
        vp = int(round(len(pos) * val_fraction))
        vn = int(round(len(neg) * val_fraction))

        # гарант минимум позитивов в валидации, если хватает
        if ensure_min_val_pos and len(pos) >= ensure_min_val_pos:
            vp = max(vp, ensure_min_val_pos)

        val = pos[:vp] + neg[:vn]
        train = pos[vp:] + neg[vn:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_dataset(
    items: List[Item],
    out_split_dir: Path,
    new_prefix: str,
    mapping_writer: csv.writer,
):
    """
    Копируем/пишем лейблы в out_split_dir/images|labels с новыми именами.
    Если исходного label нет — создаём пустой.
    Если label есть — переписываем ЧИНЕННЫЕ строки.
    """
    (out_split_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_split_dir / "labels").mkdir(parents=True, exist_ok=True)

    for i, it in enumerate(tqdm(items, desc=f"Копируем {out_split_dir.name}")):
        stem = f"{new_prefix}_{i:06d}"
        img_dst = out_split_dir / "images" / f"{stem}{it.img.suffix.lower()}"
        lbl_dst = out_split_dir / "labels" / f"{stem}{LABEL_EXT}"

        shutil.copy2(it.img, img_dst)

        if it.lbl and it.lbl.exists():
            fixed = read_and_fix_labels(it.lbl)
            if fixed:
                with open(lbl_dst, "w", encoding="utf-8") as fw:
                    for c, xc, yc, w, h in fixed:
                        fw.write(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            else:
                lbl_dst.touch()
        else:
            lbl_dst.touch()

        mapping_writer.writerow(
            [str(it.img), str(it.lbl) if it.lbl else "", str(img_dst), str(lbl_dst)]
        )


def prepare_yolo_dataset(
    source_dir: str = "./dataset",
    output_dir: str = "./dataset/yolo_dataset",
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
    split_by_scene: bool = SPLIT_BY_SCENE,
) -> bool:
    random.seed(seed)

    src = Path(source_dir)
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)

    items = collect_items(source_dir)
    if not items:
        print("❌ Данные не обнаружены — проверьте пути в DATASET_FOLDERS")
        return False

    # Статистика до сплита
    n_total = len(items)
    n_pos = sum(1 for it in items if it.has_pos)
    n_neg = n_total - n_pos
    print(
        f"\n✅ Найдено изображений: {n_total}  |  c объектами: {n_pos}  |  пустых: {n_neg}"
    )

    train, val = stratified_split(
        items,
        val_fraction,
        seed,
        split_by_scene=split_by_scene,
        ensure_min_val_pos=ENSURE_MIN_VAL_POS,
    )
    print(f"→ Train: {len(train)}  |  Val: {len(val)}")

    # Пишем mapping.csv
    (out).mkdir(parents=True, exist_ok=True)
    with open(out / "mapping.csv", "w", newline="", encoding="utf-8") as fmap:
        mw = csv.writer(fmap)
        mw.writerow(["src_image", "src_label", "dst_image", "dst_label"])
        write_dataset(train, out / "train", "train", mw)
        write_dataset(val, out / "val", "val", mw)

    # dataset.yaml
    yaml_text = f"""# Rescue-SAR YOLO dataset (auto-generated, v2)
path: {out.resolve()}
train: train/images
val: val/images

nc: 1
names: ['person']

# stats
train_images: {len(train)}
val_images: {len(val)}
total_images: {n_total}
pos_images: {n_pos}
neg_images: {n_neg}
"""
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"📋 dataset.yaml создан: {out / 'dataset.yaml'}")
    print(f"🧭 mapping.csv: {out / 'mapping.csv'}")

    return True


if __name__ == "__main__":
    ok = prepare_yolo_dataset()
    print("\n🎉 Данные готовы!" if ok else "\n❌ Ошибка подготовки данных")
