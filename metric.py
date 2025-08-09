import json
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import pandas as pd
from numba import jit

# Фиксируем сиды
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# [ Параметры ]
PUBLIC_GT_CSV_PATH: str = "public_gt_solution_24-10-24.csv"
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


def df_to_bytes(df: pd.DataFrame) -> bytes:
    df_byte: bytes = df.to_json().encode(encoding="utf-8")

    return df_byte


def bytes_to_dict(df_byte: bytes) -> dict:
    if isinstance(df_byte, bytes):
        df_byte = df_byte.decode("utf-8")
    df_byte = df_byte.replace("'", '"')
    df_dict: dict = json.loads(df_byte)

    return df_dict


def bytes_to_df(df_byte: bytes) -> pd.DataFrame:
    predicted_dict = bytes_to_dict(df_byte)
    predicted_df = pd.DataFrame(predicted_dict)

    return predicted_df


def open_df_as_bytes(csv_path: str) -> bytes:
    df = pd.read_csv(
        csv_path,
        sep=",",
        decimal=".",
        converters={"image_id": str, "time_spent": float},
    )
    df_bytes = df_to_bytes(df)

    return df_bytes


def set_types(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "image_id": str,
            "label": int,
            "xc": float,
            "yc": float,
            "w": float,
            "h": float,
            "w_img": int,
            "h_img": int,
        },
        errors="ignore",
    )


def get_time_spent(df: pd.DataFrame, m: int) -> np.ndarray:
    # Проверяем, что time_spent для каждого image_id одинаковые
    for image_id, group in df.groupby("image_id"):
        assert group["time_spent"].nunique() == 1, (
            f"Разные значения time_spent для image_id: {image_id}"
        )

    # Получаем первое значение time_spent для каждого уникального image_id
    time_spent = df.groupby("image_id")["time_spent"].first()
    time_spent = time_spent.reset_index()["time_spent"].values
    assert len(time_spent) == m, (
        f"Количество объектов time_spent должно быть {m} (у вас {len(time_spent)})"
    )

    return time_spent


def preprocess_predicted_df(
    predicted_file: bytes, gt_file: bytes, m: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    # Преобразуем байткод в датафрейм
    if gt_file is None:
        gt_file: bytes = open_df_as_bytes(PUBLIC_GT_CSV_PATH)
    predicted_df = bytes_to_df(predicted_file)
    gt_df = bytes_to_df(gt_file)
    # Валидируем данные
    assert set(predicted_df.columns) == set(COLUMNS), Exception(
        f"Ошибка названия столбцов в датафрейме: у вас {list(predicted_df.columns)}, должны быть {COLUMNS}"
    )

    assert "score" in predicted_df, (
        "Датафрейм должен содержать столбец score времени инференса на изображении"
    )
    assert not np.any(predicted_df["w"].values < 0.0) and not np.any(
        predicted_df["w"].values > 1
    ), "Ширина (w) должна быть в пределах [0, 1]."
    assert not np.any(predicted_df["h"].values < 0.0) and not np.any(
        predicted_df["h"].values > 1
    ), "Высота (h) должна быть в пределах [0, 1]."
    assert not np.any(predicted_df["w_img"].values < 1) and not np.any(
        predicted_df["w_img"].values > 15000
    ), "Ширина (w_img) должна быть в пределах [0, 15000]."
    assert not np.any(predicted_df["h_img"].values < 1) and not np.any(
        predicted_df["h_img"].values > 15000
    ), "Высота (h_img) должна быть в пределах [0, 15000]."
    assert not np.any(predicted_df["xc"].values < 0.0) and not np.any(
        predicted_df["xc"].values > 1.0
    ), "Центр объекта (xc) должен быть в пределах [0, 1]."
    assert not np.any(predicted_df["yc"].values < 0.0) and not np.any(
        predicted_df["yc"].values > 1.0
    ), "Центр объекта (yc) должен быть в пределах [0, 1]."
    assert not np.any(predicted_df["score"].values < 0.0) and not np.any(
        predicted_df["score"].values > 1.0
    ), "Столбец score должен быть в пределах [0, 1]"
    assert "time_spent" in predicted_df, (
        "Датафрейм должен содержать столбец time_spent времени инференса на изображении"
    )

    # Забираем время, в том числе для пустых предсказаний, и удаляем из df
    time_spent = get_time_spent(df=predicted_df, m=m)
    del predicted_df["time_spent"]
    predicted_df = predicted_df.dropna()

    # Приводим форматы столбцов к типам
    predicted_df = set_types(predicted_df)
    gt_df = set_types(gt_df)

    # Делаем image_id индексом и сортируем, чтобы сохранить порядок индексов
    gt_df = gt_df.set_index("image_id").sort_index()
    predicted_df = predicted_df.set_index("image_id").sort_index()

    # Получаем все уникальные индексы изображений
    unique_image_ids = tuple(set(predicted_df.index.to_list() + gt_df.index.to_list()))
    assert len(unique_image_ids) <= m, Exception(
        f"Количество уникальных ID изображений не должно превышать {m}!"
    )
    assert not predicted_df.empty and not gt_df.empty, (
        "Датафреймы не должны быть пустыми"
    )

    return gt_df, predicted_df, time_spent


def get_box_coordinates(row):
    """Преобразует центр и размеры в координаты углов бокса"""
    w_img = int(row["w_img"])
    h_img = int(row["h_img"])

    x1 = int((row["xc"] - row["w"] / 2) * w_img)
    y1 = int((row["yc"] - row["h"] / 2) * h_img)
    x2 = int((row["xc"] + row["w"] / 2) * w_img)
    y2 = int((row["yc"] + row["h"] / 2) * h_img)

    return (x1, y1, x2, y2)


@jit(nopython=True)
def compute_iou_from_coords(pred_box, gt_box):
    """Вычисляет IoU между двумя боксами по координатам"""
    # pred_box и gt_box в формате (x1, y1, x2, y2)
    x1_p, y1_p, x2_p, y2_p = pred_box
    x1_g, y1_g, x2_g, y2_g = gt_box
    # Находим координаты пересечения
    x_left = max(x1_p, x1_g)
    y_top = max(y1_p, y1_g)
    x_right = min(x2_p, x2_g)
    y_bottom = min(y2_p, y2_g)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Площади боксов
    box1_area = (x2_p - x1_p) * (y2_p - y1_p)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def process_image(pred_df, gt_df, thresholds):
    """Обработка одного изображения"""
    pred_boxes = [get_box_coordinates(row) for _, row in pred_df.iterrows()]
    gt_boxes = [get_box_coordinates(row) for _, row in gt_df.iterrows()]

    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    iou_matrix = np.zeros((num_pred, num_gt))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou_from_coords(pred_box, gt_box)

    results = {}
    for t in thresholds:
        matches = []
        iou_mat = iou_matrix.copy()
        iou_mat[iou_mat < t] = 0

        pred_indices = set()
        gt_indices = set()

        while True:
            max_iou = iou_mat.max()
            if max_iou == 0:
                break
            i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if i not in pred_indices and j not in gt_indices:
                pred_indices.add(i)
                gt_indices.add(j)
                matches.append((i, j))
            iou_mat[i, :] = 0
            iou_mat[:, j] = 0

        tp = len(matches)
        fp = num_pred - tp
        fn = num_gt - tp

        results[t] = {"tp": tp, "fp": fp, "fn": fn}

    return results


def compute_overall_metric(
    predicted_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    time_spent: np.ndarray,
    thresholds: np.ndarray,
    beta: float,
    m: int,
    parallelize: bool = True,
) -> np.float64:
    # Получаем все уникальные индексы изображений
    unique_image_ids = tuple(set(predicted_df.index.to_list() + gt_df.index.to_list()))

    total_tp = {t: 0 for t in thresholds}
    total_fp = {t: 0 for t in thresholds}
    total_fn = {t: 0 for t in thresholds}

    def process_image_id(image_id):
        pred_df_image_id = predicted_df[predicted_df.index == image_id]
        gt_df_image_id = gt_df[gt_df.index == image_id]

        # Случай, когда истинные значения есть, а предсказаний нет
        if not gt_df_image_id.empty and pred_df_image_id.empty:
            num_gt = np.float64(len(gt_df_image_id))
            return {t: {"tp": 0, "fp": 0, "fn": num_gt} for t in thresholds}

        # Случай, когда предсказания есть, а истинных значений нет
        elif not pred_df_image_id.empty and gt_df_image_id.empty:
            num_pred = np.float64(len(pred_df_image_id))
            return {t: {"tp": 0, "fp": num_pred, "fn": 0} for t in thresholds}

        # Оба случая не пустые
        elif not pred_df_image_id.empty and not gt_df_image_id.empty:
            # Вместо работы с масками передаем напрямую датафреймы
            result = process_image(pred_df_image_id, gt_df_image_id, thresholds)
            return result

        return None

    results = []
    if parallelize:
        # Распараллеленное многопоточное выполнение
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image_id, unique_image_ids))
    else:
        # Последовательный цикл выполнения
        for image_id in unique_image_ids:
            results.append(process_image_id(image_id))

    for result in results:
        if result:
            for t in thresholds:
                total_tp[t] += result[t]["tp"]
                total_fp[t] += result[t]["fp"]
                total_fn[t] += result[t]["fn"]

    metric, accuracy, fp_rate, avg_time = metric_counter(
        time_spent=time_spent,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        thresholds=thresholds,
        beta=beta,
        m=m,
    )

    return metric, accuracy, fp_rate, avg_time


def metric_counter(
    time_spent: np.ndarray,
    total_tp: dict,
    total_fp: dict,
    total_fn: dict,
    thresholds: np.ndarray,
    beta: float,
    m: int,
) -> np.float64:
    # Рачет Q (F-beta score)
    f_beta_scores = []
    beta_squared = beta**2
    for t in thresholds:
        tp = total_tp[t]
        fp = total_fp[t]
        fn = total_fn[t]
        numerator = (1 + beta_squared) * tp
        denominator = (1 + beta_squared) * tp + beta_squared * fn + fp
        if denominator == 0:
            f_beta_t = 0.0
        else:
            f_beta_t = numerator / denominator
        f_beta_scores.append(f_beta_t)

    # Метрика теперь равна только F-beta score (без учета скорости)
    metric = (1 / len(thresholds)) * np.sum(f_beta_scores)
    metric = np.round(metric, decimals=10)

    # Точность обнаружения
    accuracy = {
        float(t): round(float((total_tp[t] / (total_tp[t] + total_fp[t])) * 100), 3)
        if (total_tp[t] + total_fp[t]) > 0
        else 0
        for t in thresholds
    }
    # Число ложноположительных срабатываний
    fp_rate = {
        float(t): round(float((total_fp[t] / (total_fp[t] + total_tp[t])) * 100), 3)
        if (total_fp[t] + total_tp[t]) > 0
        else 0
        for t in thresholds
    }
    # Среднее время на обработку снимка
    avg_time = round(float(np.mean(time_spent)), 3)

    return metric, accuracy, fp_rate, avg_time


def evaluate(
    predicted_file: bytes,
    gt_file: bytes = None,
    thresholds: np.ndarray = np.round(np.arange(0.3, 1.0, 0.07), 2),
    beta: float = 1.0,
    m: int = 500,
    parallelize: bool = True,
) -> float:
    metric, accuracy, fp_rate, avg_time = 0.0, {}, {}, 0.0
    try:
        # Валидация данных, конвертация датафреймов, приведение типов
        gt_df, predicted_df, time_spent = preprocess_predicted_df(
            gt_file=gt_file, predicted_file=predicted_file, m=m
        )
        # Расчет метрики, выполняется параллельно (parallelize=True)
        metric, accuracy, fp_rate, avg_time = compute_overall_metric(
            predicted_df=predicted_df,
            gt_df=gt_df,
            thresholds=thresholds,
            beta=beta,
            m=m,
            parallelize=parallelize,
            time_spent=time_spent,
        )
    except Exception as e:
        raise Exception(f"Произошла ошибка выполнения скрипта: {str(e)}")

    return metric, accuracy, fp_rate, avg_time
    metric, accuracy, fp_rate, avg_time = 0.0, {}, {}, 0.0
    try:
        # Валидация данных, конвертация датафреймов, приведение типов
        gt_df, predicted_df, time_spent = preprocess_predicted_df(
            gt_file=gt_file,
            predicted_file=predicted_file,
            inference_min_time=inference_min_time,
            m=m,
        )
        # Расчет метрики, выполняется параллельно (parallelize=True)
        metric, accuracy, fp_rate, avg_time = compute_overall_metric(
            predicted_df=predicted_df,
            gt_df=gt_df,
            thresholds=thresholds,
            beta=beta,
            t_limit=t_limit,
            m=m,
            gamma=gamma,
            parallelize=parallelize,
            time_spent=time_spent,
        )
    except Exception as e:
        raise Exception(f"Произошла ошибка выполнения скрипта: {str(e)}")

    return metric, accuracy, fp_rate, avg_time
