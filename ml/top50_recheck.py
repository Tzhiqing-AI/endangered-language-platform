"""
预警榜 TOP-50 重算：新累积概率口径（温度缩放校准后的 P(≥高危)）。

旧口径：risk_score = proba @ [20, 40, 60, 80, 100]，193 条并列 100.0。
新口径：P(≥高危) = P(endangered) + P(seriously_endangered) + P(extinct)，
        概率先经温度缩放校准（T 来自 ml/artifacts/risk_calibration.json）。

排序与 tie-break（严格按此顺序，全部降序，最后 code_id 升序）：
  1. P(≥高危)            降序
  2. P(≥垂危) = P(seriously_endangered) + P(extinct)   降序
  3. P(extinct)          降序
  4. code_id             升序

两张榜：
  生产榜    全部可训练语言（3,462）
  可行动榜  speakers > 0 且模型 argmax 属于高危三类，并剔除 argmax=extinct
            （与旧榜同一候选池，只换排序键，便于隔离"口径变化"的影响）

只读输入：
  ml/endangerment_dataset.csv
  ml/artifacts/language_risk_oof.csv
  ml/artifacts/risk_calibration.json
  ml/reports/warning_top50.csv（旧榜，用于计算重叠）
输出（新增）：
  ml/reports/top50_recheck.json

复算：
  python ml/top50_recheck.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')

import django  # noqa: E402

django.setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from Dashboard.risk_calibration import CLASS_ORDER, temperature_scale  # noqa: E402

ML_DIR = BASE_DIR
DATASET_PATH = ML_DIR / 'endangerment_dataset.csv'
OOF_PATH = ML_DIR / 'artifacts' / 'language_risk_oof.csv'
CALIB_PATH = ML_DIR / 'artifacts' / 'risk_calibration.json'
OLD_TOP50_PATH = ML_DIR / 'reports' / 'warning_top50.csv'
OUT_JSON = ML_DIR / 'reports' / 'top50_recheck.json'

TOP_N = 50
HIGH = ('endangered', 'seriously_endangered', 'extinct')
HIGH_LEVELS = ('endangered', 'seriously_endangered', 'extinct')


def main() -> None:
    started = time.time()

    calib = json.loads(CALIB_PATH.read_text(encoding='utf-8'))
    temperature = float(calib['temperature'])
    assert calib['method'] == 'temperature_scaling'
    assert list(calib['classes']) == list(CLASS_ORDER)

    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    df = df[df['trainable'].astype(bool)].reset_index(drop=True)
    oof = pd.read_csv(OOF_PATH, encoding='utf-8-sig')

    data = df.merge(
        oof[['code_id', 'predicted_level', 'risk_score', 'confidence', 'top_factors']
            + [f'p_{c}' for c in CLASS_ORDER]],
        on='code_id', how='left', validate='one_to_one')
    assert data['predicted_level'].notna().all(), 'OOF 未覆盖全部可训练语言'

    raw = data[[f'p_{c}' for c in CLASS_ORDER]].to_numpy(dtype=float)
    calibrated = np.array([temperature_scale(row, temperature) for row in raw], dtype=float)

    high_idx = [CLASS_ORDER.index(c) for c in HIGH]
    severe_idx = [CLASS_ORDER.index(c) for c in ('seriously_endangered', 'extinct')]
    ext_idx = CLASS_ORDER.index('extinct')

    data['p_high'] = calibrated[:, high_idx].sum(axis=1)
    data['p_severe'] = calibrated[:, severe_idx].sum(axis=1)
    data['p_extinct_cal'] = calibrated[:, ext_idx]
    data['old_risk_score'] = data['risk_score']

    # ---- 并列情况 ----
    p_high_rounded = data['p_high'].round(10)
    max_high = float(data['p_high'].max())
    top_tie_n = int((p_high_rounded == round(max_high, 10)).sum())
    tie_1_0_exact = int((data['p_high'] == 1.0).sum())
    tie_1_0_round4 = int((data['p_high'].round(4) == 1.0).sum())
    # 50 名边界上的并列规模
    ranked_all = data.sort_values(
        ['p_high', 'p_severe', 'p_extinct_cal', 'code_id'],
        ascending=[False, False, False, True]).reset_index(drop=True)
    boundary_value = round(float(ranked_all.loc[TOP_N - 1, 'p_high']), 10)
    boundary_tie_n = int((ranked_all['p_high'].round(10) == boundary_value).sum())
    boundary_next = float(ranked_all.loc[TOP_N, 'p_high'])

    # 显示精度（4 位小数）下重排：检验"前 50 名是否由 p_high 的真实差异决定"
    ranked_r4 = data.assign(p_high_r4=data['p_high'].round(4)).sort_values(
        ['p_high_r4', 'p_severe', 'p_extinct_cal', 'code_id'],
        ascending=[False, False, False, True]).reset_index(drop=True)
    top_r4 = set(ranked_r4.head(TOP_N)['code_id'].astype(int))

    def pick(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(
            ['p_high', 'p_severe', 'p_extinct_cal', 'code_id'],
            ascending=[False, False, False, True]).head(TOP_N).reset_index(drop=True)

    production = pick(data)

    pool = data[(data['speakers'] > 0)
                & data['predicted_level'].isin(HIGH_LEVELS)
                & (data['predicted_level'] != 'extinct')]
    actionable = pick(pool)

    # 对照：沿用旧池（speakers>0 且高危三类，含 argmax=extinct），只换排序键
    pool_same = data[(data['speakers'] > 0)
                     & data['predicted_level'].isin(HIGH_LEVELS)]
    actionable_same_pool = pick(pool_same)

    def stats(frame: pd.DataFrame) -> dict:
        order = {c: i for i, c in enumerate(CLASS_ORDER)}
        pred_rank = frame['predicted_level'].map(order)
        legacy_rank = frame['legacy_endangerment_level'].map(order)
        leads = []
        for value in frame['top_factors'].fillna('[]'):
            try:
                items = json.loads(value)
            except (TypeError, ValueError):
                items = []
            leads.append(items[0].get('feature', '') if items else '')
        return {
            'n': int(len(frame)),
            'predicted_extinct': int((frame['predicted_level'] == 'extinct').sum()),
            'speakers_zero': int((frame['speakers'] == 0).sum()),
            'same_as_label': int((frame['predicted_level']
                                  == frame['legacy_endangerment_level']).sum()),
            'more_severe_than_label': int((pred_rank > legacy_rank).sum()),
            'less_severe_than_label': int((pred_rank < legacy_rank).sum()),
            'predicted_level_counts': frame['predicted_level'].value_counts().to_dict(),
            'p_high_min_in_list': round(float(frame['p_high'].min()), 6),
            'top_factor_lead_counts': pd.Series(leads).value_counts().to_dict(),
            'speakers_field_lead': int(pd.Series(leads).isin(
                ['speakers', 'log_speakers', 'max_speaker_number']).sum()),
            'country_top': frame['country'].value_counts().head(5).to_dict(),
        }

    old = pd.read_csv(OLD_TOP50_PATH, encoding='utf-8-sig')
    old_raw = set(old.loc[old['list_type'] == 'raw_risk_score', 'code_id'].astype(int))
    old_act = set(old.loc[old['list_type'] == 'actionable_speakers_gt_0',
                         'code_id'].astype(int))
    new_raw = set(production['code_id'].astype(int))
    new_act = set(actionable['code_id'].astype(int))
    new_act_same_pool = set(actionable_same_pool['code_id'].astype(int))

    rank_of = {int(cid): i + 1 for i, cid in enumerate(ranked_all['code_id'])}
    old_prod_ranks = sorted(rank_of[cid] for cid in old_raw)

    # 旧榜在同一批数据上的复现指标（确认旧口径 193 并列）
    old_saturated = int((data['old_risk_score'] >= 100.0).sum())

    def rows(frame: pd.DataFrame) -> list[dict]:
        out = []
        for rank, row in frame.iterrows():
            out.append({
                'rank': int(rank) + 1,
                'code_id': int(row['code_id']),
                'language_name': row['language_name'],
                'country': row['country'],
                'family_top': row['family_top'],
                'p_high_risk': round(float(row['p_high']), 6),
                'p_severe': round(float(row['p_severe']), 6),
                'p_extinct': round(float(row['p_extinct_cal']), 6),
                'predicted_level': row['predicted_level'],
                'legacy_endangerment_level': row['legacy_endangerment_level'],
                'speakers': None if pd.isna(row['speakers']) else float(row['speakers']),
                'old_risk_score': round(float(row['old_risk_score']), 2),
            })
        return out

    overlap = {
        'production': {
            'n_new': len(new_raw), 'n_old': len(old_raw),
            'overlap': len(new_raw & old_raw),
            'jaccard': round(len(new_raw & old_raw) / len(new_raw | old_raw), 6),
        },
        'actionable': {
            'n_new': len(new_act), 'n_old': len(old_act),
            'overlap': len(new_act & old_act),
            'jaccard': round(len(new_act & old_act) / len(new_act | old_act), 6),
        },
        'actionable_same_pool_only_rerank': {
            'note': '沿用旧池（speakers>0 且高危三类，含 argmax=extinct），只换排序键',
            'n_new': len(new_act_same_pool), 'n_old': len(old_act),
            'overlap': len(new_act_same_pool & old_act),
            'jaccard': round(len(new_act_same_pool & old_act)
                             / len(new_act_same_pool | old_act), 6),
        },
    }

    from scipy.stats import spearmanr
    spearman = float(spearmanr(data['old_risk_score'], data['p_high']).statistic)

    out = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'ml/top50_recheck.py',
        'purpose': '新累积概率口径下重算 TOP-50，并复核旧结论',
        'calibration': {
            'method': calib['method'],
            'temperature': temperature,
            'source': 'ml/artifacts/risk_calibration.json',
            'applied_to': 'OOF 五类概率（逐行 normalize → softmax(log p / T)）',
        },
        'ranking_rule': {
            'primary': 'P(>=高危) = P(endangered)+P(seriously_endangered)+P(extinct)，降序',
            'tie_break': ['P(>=垂危)=P(seriously_endangered)+P(extinct) 降序',
                          'P(extinct) 降序',
                          'code_id 升序'],
        },
        'pool_definition': {
            'production': '全部可训练语言 3,462 条',
            'actionable': 'speakers>0 且 argmax ∈ {endangered, seriously_endangered, extinct}，并剔除 argmax=extinct',
            'actionable_n': int(len(pool)),
        },
        'ties': {
            'new_p_high_max': round(max_high, 6),
            'new_top_tie_rows': top_tie_n,
            'p_high_eq_1_0_exact': tie_1_0_exact,
            'p_high_eq_1_0_round4': tie_1_0_round4,
            'rank50_p_high': boundary_value,
            'rank50_tie_rows': boundary_tie_n,
            'rank51_p_high': round(boundary_next, 6),
            'old_score_eq_100': old_saturated,
        },
        'stability': {
            'p_high_range_in_top50': round(float(production['p_high'].max()
                                                 - production['p_high'].min()), 6),
            'rank50_51_gap': round(boundary_value - boundary_next, 6),
            'p_high_ge_0_997': int((data['p_high'] >= 0.997).sum()),
            'p_high_ge_0_996': int((data['p_high'] >= 0.996).sum()),
            'p_high_ge_0_99': int((data['p_high'] >= 0.99).sum()),
            'top50_overlap_if_rounded_4dp': len(new_raw & top_r4),
            'old_top50_best_new_rank': int(min(old_prod_ranks)),
            'old_top50_median_new_rank': int(np.median(old_prod_ranks)),
            'old_top50_worst_new_rank': int(max(old_prod_ranks)),
        },
        'summary': {
            'production': stats(production),
            'actionable': stats(actionable),
            'old_actionable_reference': {
                'n': len(old_act),
                'same_as_label': int(old.loc[old['list_type'] == 'actionable_speakers_gt_0',
                                             'predicted_level'].eq(
                    old.loc[old['list_type'] == 'actionable_speakers_gt_0',
                            'legacy_endangerment_level']).sum()),
            },
        },
        'overlap_with_old': overlap,
        'spearman_old_score_vs_new_p_high': round(spearman, 6),
        'production_top50': rows(production),
        'actionable_top50': rows(actionable),
        'runtime_seconds': round(time.time() - started, 1),
    }

    tmp = OUT_JSON.with_name(OUT_JSON.name + '.tmp')
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, OUT_JSON)

    print(json.dumps({
        'T': temperature,
        'ties': out['ties'],
        'summary': out['summary'],
        'overlap': overlap,
        'spearman': out['spearman_old_score_vs_new_p_high'],
        'runtime': out['runtime_seconds'],
    }, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
