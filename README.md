## Архипелаг 2035 — SAR/UAV Object Detection

# SAR-UAV People Search — YOLOv11m (tiled)

Команда - Квадрицепс

3 место

[презентация](https://drive.google.com/file/d/1W9CSNmujWjeCk7zFkixa3bOproTdh3ou/view)

Репозиторий решения для хакатона по поиску пропавших людей на аэрофото/дрон-снимках.
Финальная версия использует **YOLOv11m + high-res + плиточный (tiled) инференс** c “coarse→tiles” оконным отбором и быстрым NMS.

---

## Что внутри

* **`train_yolo11_v4.py`** — обучение, плиточный инференс, грид-поиск гиперпараметров (один скрипт).
* **`solution.py`** — финальный, максимально быстрый инференс для сабмита/продакшена (API `predict()`).
* **`metric.py`** — обёртка над метрикой организаторов (`evaluate`, `df_to_bytes`, `open_df_as_bytes`).
* Примеры конфигов/данных:

  * `dataset.yaml` в формате Ultralytics YOLO (train/val в YOLO-разметке).
  * Директории `images/` и `labels/` (YOLO txt).

> Имена файлов можно поменять под ваш проект; в README используются вышеуказанные.

---

## TL;DR: результаты

**Метрика организаторов (Fβ, β=1):**

| Модель/режим                             | imgsz | Тайлы |       Скор |
| ---------------------------------------- | ----: | :---: | ---------: |
| YOLOv11m, дефолт Ultralytics             |   640 |   —   |     0.2064 |
| YOLOv11m, дефолт Ultralytics             |  1230 |   —   |     0.3130 |
| YOLOv11l, дефолт Ultralytics             |  1230 |   —   |     0.3479 |
| YOLOv11m, **1536**, **SAHI tiles**       |  1536 |   ✔   |     0.6231 |
| **Финал: YOLOv11m + наш tiled-инференс** |  1536 |   ✔   | **0.6479** |

Что пробовали, но уступило:

* **Faster R-CNN ResNet50-FPNv2** → 0.4807
* **TPH-YOLOv5++** → 0.5042

---

## Почему именно YOLOv11m

* **Лучший баланс точность/скорость/VRAM** при high-res и тайлинге.
* **Стабильное обучение** и быстрый цикл экспериментов (Ultralytics).
* **Главный прирост дал не размер модели, а разрешение + тайлы**: человек становится «крупнее» в пикселях, детектор видит больше деталей.

---

## Ключевая инженерия финального инференса

См. `solution.py` — это наше «боевое» решение.

* **Warm-up** на двух фикс-формах (тайл и “coarse” квадрат) — снимает холодный шип.
* **Coarse-окна:** быстрый прогон уменьшенной копии кадра (**1280×1280**) → обратная проекция боксов в исходное изображение → **расширение** и **квантование** под `TILE_SIZE` → **в разы меньше тайлов**.
* **Параллельная нарезка/декод**: два `ThreadPoolExecutor` (до 12 потоков) — CPU не простаивает.
* **Пакетный инференс тайлов** через `model.predict(source=[...])` (Ultralytics сам формирует батч) — стабильная загрузка GPU.
* **FP16 + `channels_last`** — экономия памяти и ускорение свёрток.
* **Лёгкое слияние**: **torchvision NMS (IoU=0.55)** на глобальных координатах + отсечка микробоксов (`MIN_WH_PIX=4`).
* **Подробный профилинг**: `windows_ms`, `gpu_infer`, `decode`, `nms`, `total` — легко оптимизировать в проде.

**Финальные гиперы инференса:**

```
TILE_SIZE=1536, OVERLAP=0.20,
COARSE_SHORT_SIDE=1280, COARSE_CONF=0.10, COARSE_EXPAND=1.8, MAX_TILES=64,
NMS_IOU=0.55, MIN_WH_PIX=4,
MAX_TILES_PER_BATCH=4, MAX_THREADS_CPU=12
```

---

## Обучение на RTX 4090 (24 GB)

* **AMP** (смешанная точность) + **TF32**, `channels_last`.
* Умеренные аугментации (Affine, цвет/контраст, лёгкий blur/noise) — бережно к мелким целям.
* При `imgsz=1536` — батч **8–12**; при нехватке VRAM — используйте накопление градиента.

---

## Установка

```bash
# UV рекомендуем
uv sync && source .venv/bin/activate

# базовые зависимости
ultralytics==8.* torch torchvision torchaudio \
            albumentations opencv-python pillow numpy pandas tqdm pyyaml
# (если нужен быстрый NMS/ops — torchvision уже включает)
```

> На CUDA убедитесь, что версия `torch` соответствует вашему драйверу/CUDA.

---

## Быстрый старт

### 1) Подготовка данных (YOLO-формат)

`dataset.yaml`:

```yaml
path: ./dataset/yolo_dataset
train: images/train
val: images/val
names:
  0: person
```

Разметка: `labels/<split>/<image>.txt` (YOLO: `class xc yc w h`, нормировано 0..1).

### 2) Обучение (плиточное)

```bash
python train_yolo11_v4.py train \
  --data ./dataset/yolo_dataset/dataset.yaml \
  --model m \
  --epochs 20 \
  --batch 16 \
  --imgsz 1536 \
  --tile-train 1 --tile 1024 --stride 512 \
  --pos-frac 0.5 --max-neg 6 --workers 8 \
  --pos-iou-thr 0.2
```

Скрипт автоматически подменяет лоадеры на **tile-train** и копирует лучший вес в `./best.pt`.

### 3) Грид-поиск гиперов инференса (опционально)

```bash
python train_yolo11_v4.py grid \
  --model ./best.pt \
  --val-images-dir ./dataset/yolo_dataset/images/val \
  --gt-csv ./val_groundtruth.csv \
  --beta 1.0
```

Результаты: `grid_search_log.csv`, `best_thresh.json`.

### 4) Инференс на папке изображений (SAHI-style)

```bash
python train_yolo11_v4.py infer \
  --model ./best.pt \
  --images-dir ./dataset/yolo_dataset/images/val \
  --out-csv ./predictions.csv \
  --slice-size 1024 --overlap 0.25 \
  --conf 0.10 --iou-nms 0.7 --wbf-iou 0.55 \
  --imgsz 1280
```

---

## Архитектура пайплайна

1. **YOLOv11m** (1 класс «person») с high-res тренировкой и tile-train даталоадерами.
2. **Инференс:** *coarse → windows → batch tiles → decode → NMS → submit*.
3. Отдельный **grid-search** гиперов инференса под метрику организаторов.
4. Единый публичный **`predict(images)`**.

---

## Дальнейшие улучшения (roadmap)

* Ещё умнее **coarse** (heatmap/segment-подсказки) → меньше пустых тайлов.
* Лёгкая **TTA** (flip/rotate) + NMS/WBF на инференсе (без retrain).
* **Pseudo-labeling** на неразмеченных полётах.
* **Видео-трекинг** (например, ByteTrack) — меньше fp/fn во времени.
* Экспорт **ONNX/TensorRT** для edge-GPU/серверов.

---

## Типичные вопросы

**Q:** Почему не взяли более «тяжёлую» модель?

**A:** На гигантских кадрах с людьми 10–40 px эффективнее «давать пиксели» через high-res + тайлы. YOLOv11m в этом сетапе даёт лучший trade-off.

**Q:** Почему не WBF в финале?

**A:** Наш оконный отбор и NMS на глобальных координатах уже достаточно чистый; **torchvision NMS** проще и быстрее, точности хватает.

---

### Источники/ссылки
- Описание задачи: [buildin.ai](https://buildin.ai/share/4af5d8c2-898e-45f4-ab31-8737a9ef2269?code=C017S4)
- Про VisDrone: [Ultralytics Docs](https://docs.ultralytics.com/ru/datasets/detect/visdrone)
---
Вопросы/идеи по улучшению: TensorRT EP (опционально), FP16/INT8 ONNX, ещё более агрессивный батчинг с IO binding.

# archipelago2035
техзрение.архипелаг2035.рф
