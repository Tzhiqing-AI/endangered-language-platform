"""风险概率的聚合与校准工具。

本模块只做两件事，不参与模型训练：

1. 把五类概率聚合成可直接展示的累积概率、三档风险等级和 top-2 类别；
2. 校验并应用离线拟合的温度缩放校准器，同时提供 ECE / Brier 等评估函数。

设计口径与 ``ml/build_dataset.py`` 保持一致：
``unsafe < danger < endangered < seriously_endangered < extinct``，
其中 ``endangered`` 及以上三类即 ``HIGH_RISK``（高危及以上）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

CLASS_ORDER = ('unsafe', 'danger', 'endangered', 'seriously_endangered', 'extinct')

LEVEL_LABELS_ZH = {
    'unsafe': '不安全',
    'danger': '危险',
    'endangered': '濒危',
    'seriously_endangered': '垂危',
    'extinct': '灭绝',
}

# 与 ml/build_dataset.py 的 HIGH_RISK 完全一致：濒危、垂危、灭绝。
HIGH_RISK_CLASSES = ('endangered', 'seriously_endangered', 'extinct')
DANGER_PLUS_CLASSES = ('danger', 'endangered', 'seriously_endangered', 'extinct')
SEVERE_CLASSES = ('seriously_endangered', 'extinct')

RISK_TIER_LABELS = {
    'low': '低风险',
    'medium': '中风险',
    'high': '高风险',
}

RISK_TIER_COLORS = {
    'low': '#10b981',
    'medium': '#f59e0b',
    'high': '#dc2626',
}

# 分档阈值只看累积概率是否过半：
#   high   : P(>= 濒危) >= 0.5
#   medium : P(>= 危险) >= 0.5 且 P(>= 濒危) < 0.5
#   low    : 其余
TIER_THRESHOLD = 0.5

CALIBRATION_METHOD = 'temperature_scaling'
CALIBRATION_EPS = 1e-12
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def normalize_probabilities(values: Sequence[Any]) -> List[float]:
    """把任意长度的概率向量清洗并归一化到和为 1。

    非法值（None / NaN / 负数）按 0 处理；全零或无法解析时退化为均匀分布。
    """
    cleaned: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number) or number < 0:
            number = 0.0
        cleaned.append(number)
    if not cleaned:
        return []

    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)
    return [number / total for number in cleaned]


def temperature_scale(values: Sequence[Any], temperature: Any) -> List[float]:
    """对概率向量做温度缩放。

    ``p' = softmax(log(p) / T)``。T>1 会压低过度自信的概率；T=1 等价于原分布。
    T 非法时回退到归一化后的原始概率。
    """
    normalized = normalize_probabilities(values)
    if not normalized:
        return []

    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        return normalized
    if (not math.isfinite(temperature_value) or temperature_value <= 0
            or temperature_value > MAX_TEMPERATURE):
        return normalized

    logits = [math.log(max(value, CALIBRATION_EPS)) / temperature_value
              for value in normalized]
    maximum = max(logits)
    exps = [math.exp(value - maximum) for value in logits]
    total = sum(exps)
    if total <= 0:
        return normalized
    return [value / total for value in exps]


def validate_calibration_payload(payload: Any) -> Tuple[bool, str]:
    """校验校准产物结构；返回 (是否可用, 不可用原因)。"""
    if not isinstance(payload, dict):
        return False, '校准文件内容不是 JSON 对象'

    method = payload.get('method')
    if method != CALIBRATION_METHOD:
        return False, f'不支持的校准方法：{method!r}'

    temperature = payload.get('temperature')
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        return False, f'温度参数不可用：{temperature!r}'
    if (not math.isfinite(temperature_value) or temperature_value <= 0
            or temperature_value > MAX_TEMPERATURE):
        return False, f'温度参数超出有效范围：{temperature_value!r}'

    classes = payload.get('classes')
    if classes is not None and list(classes) != list(CLASS_ORDER):
        return False, '校准文件的类别顺序与风险服务不一致'

    return True, ''


def apply_calibration(values: Sequence[Any],
                      payload: Optional[Dict]) -> Tuple[List[float], bool, str]:
    """应用校准器；返回 (概率, 是否已校准, 未校准原因)。"""
    normalized = normalize_probabilities(values)
    if payload is None:
        return normalized, False, '校准文件缺失'

    valid, reason = validate_calibration_payload(payload)
    if not valid:
        return normalized, False, f'校准文件不可用：{reason}'

    return temperature_scale(normalized, payload.get('temperature')), True, ''


def _rounded_dict(probabilities: Sequence[float], digits: int = 4) -> Dict[str, float]:
    return {
        name: round(float(probabilities[index]), digits)
        for index, name in enumerate(CLASS_ORDER)
    }


def _pct(value: float, digits: int = 1) -> float:
    return round(float(value) * 100.0, digits)


def _sum_classes(probabilities: Sequence[float], classes: Sequence[str]) -> float:
    indices = [CLASS_ORDER.index(name) for name in classes]
    return float(sum(probabilities[index] for index in indices))


def risk_tier_from_probabilities(high_risk_prob: float,
                                 at_least_danger_prob: float) -> str:
    """按累积概率过半规则给出三档风险等级。"""
    if high_risk_prob >= TIER_THRESHOLD:
        return 'high'
    if at_least_danger_prob >= TIER_THRESHOLD:
        return 'medium'
    return 'low'


def build_probability_profile(values: Sequence[Any],
                               calibration_payload: Optional[Dict] = None,
                               calibration_error: str = '') -> Dict:
    """把一条原始五类概率转换成展示用字段。

    返回字段同时包含原始概率与校准后概率，便于界面如实标注。
    """
    raw = normalize_probabilities(values)
    calibrated, applied, apply_reason = apply_calibration(values, calibration_payload)

    # 校准文件整体缺失 / 损坏时，用调用方给出的原因覆盖笼统的“文件缺失”。
    reason = '' if applied else (calibration_error or apply_reason)

    predicted_index = int(max(range(len(calibrated)), key=lambda i: (calibrated[i], -i))
                          ) if calibrated else 0
    predicted_level = CLASS_ORDER[predicted_index] if calibrated else ''

    high_risk_prob = _sum_classes(calibrated, HIGH_RISK_CLASSES)
    at_least_danger_prob = _sum_classes(calibrated, DANGER_PLUS_CLASSES)
    severe_risk_prob = _sum_classes(calibrated, SEVERE_CLASSES)
    tier = risk_tier_from_probabilities(high_risk_prob, at_least_danger_prob)

    order = sorted(range(len(calibrated)), key=lambda i: (-calibrated[i], i))
    top2 = [
        {
            'level': CLASS_ORDER[index],
            'level_zh': LEVEL_LABELS_ZH[CLASS_ORDER[index]],
            'probability': round(float(calibrated[index]), 4),
            'probability_pct': _pct(calibrated[index]),
        }
        for index in order[:2]
    ]
    top2_margin = 0.0
    if len(order) >= 2:
        top2_margin = max(0.0, calibrated[order[0]] - calibrated[order[1]])

    probability_rows = [
        {
            'level': name,
            'level_zh': LEVEL_LABELS_ZH[name],
            'probability': round(float(calibrated[index]), 4),
            'pct': _pct(calibrated[index]),
        }
        for index, name in enumerate(CLASS_ORDER)
    ]

    calibration_meta = {
        'applied': applied,
        'method': CALIBRATION_METHOD if applied else None,
        'temperature': (round(float(calibration_payload.get('temperature')), 4)
                        if applied else None),
        'reason': reason,
    }

    return {
        'probabilities_raw': _rounded_dict(raw),
        'probabilities': _rounded_dict(calibrated),
        'probability_rows': probability_rows,
        'predicted_index': predicted_index,
        'predicted_level': predicted_level,
        'predicted_level_zh': LEVEL_LABELS_ZH.get(predicted_level, ''),
        'high_risk_prob': round(high_risk_prob, 4),
        'high_risk_prob_pct': _pct(high_risk_prob),
        'at_least_danger_prob': round(at_least_danger_prob, 4),
        'at_least_danger_prob_pct': _pct(at_least_danger_prob),
        'severe_risk_prob': round(severe_risk_prob, 4),
        'severe_risk_prob_pct': _pct(severe_risk_prob),
        'extinct_prob': round(float(calibrated[CLASS_ORDER.index('extinct')]), 4),
        'risk_tier': tier,
        'risk_tier_label': RISK_TIER_LABELS[tier],
        'risk_tier_color': RISK_TIER_COLORS[tier],
        'top2': top2,
        'top2_margin': round(top2_margin, 4),
        'top2_margin_pct': _pct(top2_margin),
        'confidence_raw': round(float(raw[predicted_index]), 4) if raw else 0.0,
        'confidence_calibrated': (round(float(calibrated[predicted_index]), 4)
                                  if calibrated else 0.0),
        'calibration': calibration_meta,
    }


# ---------------------------------------------------------------------------
# 评估函数：只读 OOF 概率与标签，供离线校准命令使用
# ---------------------------------------------------------------------------
def _as_matrix(probabilities: Sequence[Sequence[Any]]):
    import numpy as np

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    matrix = np.where(np.isfinite(matrix) & (matrix > 0), matrix, 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    matrix = matrix / safe_totals
    empty_rows = totals.reshape(-1) <= 0
    if matrix.shape[1]:
        matrix[empty_rows] = 1.0 / matrix.shape[1]
    return matrix


def temperature_scale_matrix(probabilities: Sequence[Sequence[Any]],
                             temperature: Any) -> "Any":
    """批量温度缩放；供校准命令和评估函数使用。"""
    import numpy as np

    matrix = _as_matrix(probabilities)
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        return matrix
    if (not math.isfinite(temperature_value) or temperature_value <= 0
            or temperature_value > MAX_TEMPERATURE):
        return matrix

    logits = np.log(np.clip(matrix, CALIBRATION_EPS, None)) / temperature_value
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def top_label_ece(probabilities: Sequence[Sequence[Any]],
                  labels: Sequence[int], n_bins: int = 10) -> float:
    """计算 top-label 期望校准误差（ECE），按置信度等宽分箱。"""
    import numpy as np

    matrix = _as_matrix(probabilities)
    labels_array = np.asarray(labels, dtype=int)
    if len(matrix) == 0:
        return 0.0

    confidence = matrix.max(axis=1)
    prediction = matrix.argmax(axis=1)
    correct = (prediction == labels_array).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper >= 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean())
                                            - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier(probabilities: Sequence[Sequence[Any]],
                     labels: Sequence[int]) -> float:
    """多类 Brier 分数：各类 (p - onehot)^2 的和再按样本平均。"""
    import numpy as np

    matrix = _as_matrix(probabilities)
    labels_array = np.asarray(labels, dtype=int)
    if len(matrix) == 0:
        return 0.0
    onehot = np.zeros_like(matrix)
    onehot[np.arange(len(labels_array)), labels_array] = 1.0
    return float(np.mean(np.sum((matrix - onehot) ** 2, axis=1)))


def confidence_summary(probabilities: Sequence[Sequence[Any]]) -> Dict[str, float]:
    """最大类概率的分布摘要。"""
    import numpy as np

    matrix = _as_matrix(probabilities)
    if len(matrix) == 0:
        return {'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'p99': 0.0,
                'share_ge_0_9': 0.0, 'share_ge_0_99': 0.0}
    confidence = matrix.max(axis=1)
    return {
        'mean': round(float(confidence.mean()), 4),
        'median': round(float(np.median(confidence)), 4),
        'p90': round(float(np.percentile(confidence, 90)), 4),
        'p99': round(float(np.percentile(confidence, 99)), 4),
        'share_ge_0_9': round(float((confidence >= 0.9).mean()), 4),
        'share_ge_0_99': round(float((confidence >= 0.99).mean()), 4),
    }


def macro_f1_score(probabilities: Sequence[Sequence[Any]],
                   labels: Sequence[int]) -> float:
    """校准后 argmax 的 macro F1。"""
    import numpy as np
    from sklearn.metrics import f1_score

    matrix = _as_matrix(probabilities)
    labels_array = np.asarray(labels, dtype=int)
    if len(matrix) == 0:
        return 0.0
    prediction = matrix.argmax(axis=1)
    return float(f1_score(
        labels_array, prediction,
        labels=list(range(matrix.shape[1])),
        average='macro', zero_division=0,
    ))


def _negative_log_likelihood(logits, labels: Sequence[int]) -> float:
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    labels_array = np.asarray(labels, dtype=int)
    return float(-log_prob[np.arange(len(labels_array)), labels_array].mean())


def fit_temperature(probabilities: Sequence[Sequence[Any]],
                    labels: Sequence[int],
                    bounds: Tuple[float, float] = (MIN_TEMPERATURE, MAX_TEMPERATURE)
                    ) -> float:
    """在 OOF 概率上最小化 NLL，拟合单一温度参数。"""
    import numpy as np

    matrix = _as_matrix(probabilities)
    if len(matrix) == 0:
        return 1.0
    logits = np.log(np.clip(matrix, CALIBRATION_EPS, None))

    def objective(temperature: float) -> float:
        return _negative_log_likelihood(logits / temperature, labels)

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(objective, bounds=bounds, method='bounded')
        if result.success and math.isfinite(result.x):
            return float(result.x)
    except Exception:  # noqa: BLE001
        pass

    # scipy 不可用时的兜底：在有效区间内做网格搜索。
    grid = np.linspace(bounds[0], bounds[1], 200)
    scores = [objective(float(value)) for value in grid]
    return float(grid[int(np.argmin(scores))])


def calibration_metrics(probabilities: Sequence[Sequence[Any]],
                        labels: Sequence[int], n_bins: int = 10) -> Dict:
    """一次性计算 ECE、Brier、最大置信度分布与 macro F1。"""
    return {
        'ece': round(top_label_ece(probabilities, labels, n_bins=n_bins), 6),
        'brier': round(multiclass_brier(probabilities, labels), 6),
        'max_confidence': confidence_summary(probabilities),
        'macro_f1': round(macro_f1_score(probabilities, labels), 6),
    }
