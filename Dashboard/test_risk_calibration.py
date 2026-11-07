"""风险概率聚合与校准的边界测试。

只覆盖本次改造相关逻辑，不依赖数据库；不运行全量测试树。
"""
from __future__ import annotations

import csv
import json

import pytest

from Dashboard import risk_calibration as rc
from Dashboard import risk_service


PREDICTION_COLUMNS = [
    'code_id', 'language_name', 'predicted_level', 'predicted_level_zh',
    'risk_score', 'confidence', 'risk_band', 'top_factors', 'prediction_source',
    'p_unsafe', 'p_danger', 'p_endangered', 'p_seriously_endangered', 'p_extinct',
]


@pytest.fixture(autouse=True)
def reset_risk_cache():
    risk_service.reload_cache()
    yield
    risk_service.reload_cache()


def _write_predictions(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(code_id, level='endangered', probabilities=None, risk_score=60.0,
         band='medium', source='oof'):
    values = probabilities or {}
    row = {
        'code_id': code_id,
        'language_name': f'语言{code_id}',
        'predicted_level': level,
        'predicted_level_zh': rc.LEVEL_LABELS_ZH.get(level, ''),
        'risk_score': risk_score,
        'confidence': max(values.values()) if values else 1.0,
        'risk_band': band,
        'top_factors': '[]',
        'prediction_source': source,
    }
    for name in rc.CLASS_ORDER:
        row[f'p_{name}'] = values.get(name, 0.0)
    return row


def _patch_service_paths(monkeypatch, tmp_path, prediction_file, calibration_file):
    monkeypatch.setattr(risk_service, 'PREDICTION_FILE', prediction_file)
    monkeypatch.setattr(risk_service, 'RESUBSTITUTION_FILE', tmp_path / 'missing_predictions.csv')
    monkeypatch.setattr(risk_service, 'CALIBRATION_FILE', calibration_file)


def test_single_class_exclusive_and_tier_boundaries():
    low = rc.build_probability_profile([1, 0, 0, 0, 0])
    assert low['high_risk_prob'] == 0.0
    assert low['risk_tier'] == 'low'
    assert low['top2'][0]['level'] == 'unsafe'

    medium = rc.build_probability_profile([0, 0.6, 0.4, 0, 0])
    assert medium['at_least_danger_prob'] == 1.0
    assert medium['high_risk_prob'] == 0.4
    assert medium['risk_tier'] == 'medium'

    high = rc.build_probability_profile([0, 0, 1, 0, 0])
    assert high['high_risk_prob'] == 1.0
    assert high['risk_tier'] == 'high'
    assert high['top2'][0]['level'] == 'endangered'

    # 边界：恰好 0.5 视为过半，high 优先。
    assert rc.build_probability_profile([0, 0.5, 0.5, 0, 0])['risk_tier'] == 'high'
    assert rc.build_probability_profile([0.5, 0.5, 0, 0, 0])['risk_tier'] == 'medium'


def test_probabilities_are_normalized_and_top2_margin_visible():
    profile = rc.build_probability_profile([2, 2, 2, 2, 2])
    assert pytest.approx(sum(profile['probabilities'].values()), abs=1e-6) == 1.0
    assert profile['high_risk_prob'] == pytest.approx(0.6)
    assert profile['risk_tier'] == 'high'
    # 五类等概率时 top-2 无法拉开差距，差值应为 0。
    assert profile['top2_margin'] == 0.0
    assert len(profile['top2']) == 2
    assert len(profile['probability_rows']) == 5


def test_temperature_scaling_preserves_argmax_and_normalizes():
    raw = [0.05, 0.15, 0.4, 0.3, 0.1]
    scaled = rc.temperature_scale(raw, 2.0)
    assert scaled.index(max(scaled)) == raw.index(max(raw))
    assert pytest.approx(sum(scaled), abs=1e-9) == 1.0
    assert rc.temperature_scale(raw, 1.0) == pytest.approx(raw, abs=1e-9)
    # 非法温度回退到原始分布。
    assert rc.temperature_scale(raw, 0) == pytest.approx(rc.normalize_probabilities(raw))


def test_apply_calibration_missing_invalid_and_valid():
    raw = [0, 0, 1, 0, 0]

    calibrated, applied, reason = rc.apply_calibration(raw, None)
    assert not applied
    assert '缺失' in reason

    calibrated, applied, reason = rc.apply_calibration(
        raw, {'method': 'temperature_scaling', 'temperature': 0})
    assert not applied
    assert '不可用' in reason

    calibrated, applied, reason = rc.apply_calibration(
        raw, {'method': 'temperature_scaling', 'temperature': 2, 'classes': ['x']})
    assert not applied
    assert '类别顺序' in reason

    calibrated, applied, reason = rc.apply_calibration(
        raw, {'method': 'temperature_scaling', 'temperature': 2,
              'classes': list(rc.CLASS_ORDER)})
    assert applied
    assert reason == ''
    assert pytest.approx(sum(calibrated), abs=1e-9) == 1.0
    assert calibrated.index(max(calibrated)) == 2


def test_calibration_metrics_edge_cases():
    # 五类各有样本，完美 one-hot 预测下 ECE/Brier/macro F1 均为最优。
    perfect = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
    labels = list(range(5))
    assert rc.top_label_ece(perfect, labels) == pytest.approx(0.0)
    assert rc.multiclass_brier(perfect, labels) == pytest.approx(0.0)
    assert rc.macro_f1_score(perfect, labels) == pytest.approx(1.0)

    wrong = [[1, 0, 0, 0, 0]]
    assert rc.top_label_ece(wrong, [4]) == pytest.approx(1.0)
    assert rc.multiclass_brier(wrong, [4]) == pytest.approx(2.0)
    summary = rc.confidence_summary(wrong)
    assert summary['share_ge_0_99'] == 1.0


def test_fit_temperature_returns_positive_and_preserves_argmax():
    probabilities = [
        [0.9, 0.1, 0, 0, 0],
        [0.8, 0.2, 0, 0, 0],
        [0, 0, 0.9, 0.1, 0],
        [0, 0, 0.85, 0.15, 0],
    ]
    labels = [0, 0, 2, 2]
    temperature = rc.fit_temperature(probabilities, labels)
    assert temperature > 0
    scaled = rc.temperature_scale_matrix(probabilities, temperature)
    assert list(scaled.argmax(axis=1)) == labels


def test_service_degrades_to_raw_when_calibration_missing(tmp_path, monkeypatch):
    prediction_file = tmp_path / 'predictions.csv'
    _write_predictions(prediction_file, [
        _row(1, probabilities={'endangered': 1.0}),
    ])
    calibration_file = tmp_path / 'missing_calibration.json'
    _patch_service_paths(monkeypatch, tmp_path, prediction_file, calibration_file)

    item = risk_service.get_risk_profile(1)
    assert item['calibration']['applied'] is False
    assert '缺失' in item['calibration']['reason']
    assert item['probabilities'] == item['probabilities_raw']
    assert item['risk_tier'] == 'high'
    assert item['high_risk_prob_pct'] == 100.0
    # 原有返回键必须保留，避免详情页/预警榜/旧调用方崩溃。
    for key in ('risk_score', 'confidence', 'risk_band', 'risk_band_label',
                'probabilities', 'predicted_level', 'prediction_source'):
        assert key in item


def test_service_applies_valid_calibration_and_corrupt_file_degrades(tmp_path, monkeypatch):
    prediction_file = tmp_path / 'predictions.csv'
    _write_predictions(prediction_file, [
        _row(1, probabilities={'endangered': 1.0}),
    ])
    calibration_file = tmp_path / 'risk_calibration.json'
    calibration_file.write_text(json.dumps({
        'version': 1,
        'method': 'temperature_scaling',
        'temperature': 2.0,
        'classes': list(rc.CLASS_ORDER),
    }), encoding='utf-8')
    _patch_service_paths(monkeypatch, tmp_path, prediction_file, calibration_file)

    item = risk_service.get_risk_profile(1)
    assert item['calibration']['applied'] is True
    assert item['calibration']['temperature'] == 2.0
    assert item['predicted_level'] == 'endangered'
    assert pytest.approx(sum(item['probabilities'].values()), abs=1e-6) == 1.0
    assert item['top2'][0]['level'] == 'endangered'

    # 损坏文件：不报错，降级为原始概率并明确标记。
    calibration_file.write_text('{not-json', encoding='utf-8')
    risk_service.reload_cache()
    item = risk_service.get_risk_profile(1)
    assert item['calibration']['applied'] is False
    assert '损坏' in item['calibration']['reason']
    assert item['probabilities'] == item['probabilities_raw']


def test_stale_calibration_is_not_silently_applied(tmp_path, monkeypatch):
    prediction_file = tmp_path / 'predictions.csv'
    _write_predictions(prediction_file, [
        _row(1, probabilities={'endangered': 1.0}),
    ])
    calibration_file = tmp_path / 'risk_calibration.json'
    calibration_file.write_text(json.dumps({
        'version': 1,
        'method': 'temperature_scaling',
        'temperature': 2.0,
        'classes': list(rc.CLASS_ORDER),
        'fitted_on': {'oof_sha256': 'deadbeef'},
    }), encoding='utf-8')
    _patch_service_paths(monkeypatch, tmp_path, prediction_file, calibration_file)

    item = risk_service.get_risk_profile(1)
    assert item['calibration']['applied'] is False
    assert '不匹配' in item['calibration']['reason']
    assert item['probabilities'] == item['probabilities_raw']


def test_alerts_sort_by_cumulative_probability_then_severity(tmp_path, monkeypatch):
    prediction_file = tmp_path / 'predictions.csv'
    _write_predictions(prediction_file, [
        _row(1, probabilities={'endangered': 1.0}, risk_score=60.0, band='medium'),
        _row(2, level='extinct', probabilities={'extinct': 1.0}, risk_score=100.0, band='critical'),
        _row(3, level='unsafe', probabilities={'unsafe': 1.0}, risk_score=20.0, band='low'),
    ])
    _patch_service_paths(monkeypatch, tmp_path, prediction_file,
                         tmp_path / 'missing_calibration.json')

    alerts = risk_service.get_alerts(limit=10, min_score=50)
    assert [item['code_id'] for item in alerts] == [2, 1]
    assert alerts[0]['high_risk_prob_pct'] == 100.0
    assert alerts[0]['risk_tier'] == 'high'

    stats = risk_service.get_risk_stats()
    assert stats['total'] == 3
    assert stats['tier_counts']['high'] == 2
    assert stats['tier_counts']['low'] == 1
    assert stats['calibration_applied'] is False
