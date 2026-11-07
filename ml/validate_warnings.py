"""
预警有效性验证：异源对照 + 内部一致性 + TOP-50 核查表。

不覆盖任何既有训练产物，只新增：
  ml/reports/warning_validation.json
  ml/reports/warning_top50.csv

复算：
  python ml/validate_warnings.py

口径说明（与文档一致）：
- 标签：legacy_endangerment_level（来自 LanguageVitality 的 preferred 评估，实测一致率 99.83%）。
- 异源对照：UNESCO Atlas（language_vitality.source_id=1521），排除该来源恰为 preferred 的语言。
  它不是完全独立的第三方验证，只作“另一套编目”的一致率参考。
- 内部一致性：同一份折索引下换随机种子；另换一份折划分；分档精确率；漏判/误报代价。
- 风险分沿用生产口径：proba @ [20, 40, 60, 80, 100]。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')

import django

django.setup()

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier

from Dashboard.etl.steps import RAW_LEVEL_MAP
from Dashboard.models import LanguageVitality
from ml.train_models import (
    LEVEL_ORDER, RISK_WEIGHTS_5, SklearnRiskModel, prepare_frame,
)

ML_DIR = BASE_DIR / 'ml'
DATASET_PATH = ML_DIR / 'endangerment_dataset.csv'
OOF_PATH = ML_DIR / 'artifacts' / 'language_risk_oof.csv'
MODEL_PATH = ML_DIR / 'artifacts' / 'risk_model.joblib'
META_PATH = ML_DIR / 'artifacts' / 'risk_service_meta.json'
SPLITS_PATH = ML_DIR / 'reports' / 'splits_group_family_5fold.json'
OUT_JSON = ML_DIR / 'reports' / 'warning_validation.json'
OUT_TOP50 = ML_DIR / 'reports' / 'warning_top50.csv'

SEEDS = [42, 7, 13, 2024, 99]
ALT_SPLIT_SEED = 7
TOP_N = 50
HIGH_LEVELS = ['seriously_endangered', 'extinct']
LOW_LEVELS = ['unsafe', 'danger']

ORDER_INDEX = {k: i for i, k in enumerate(LEVEL_ORDER)}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                              default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                   encoding='utf-8')
    os.replace(tmp, path)


def map_raw_level(value: str) -> str:
    if not value:
        return ''
    return RAW_LEVEL_MAP.get(value.split('(')[0].strip().lower(), '')


# ---------------------------------------------------------------------------
# 一、异源对照
# ---------------------------------------------------------------------------
def db_source_levels(source_id: int) -> tuple[dict[int, str], int]:
    """返回 {code_id: 5 级映射}，以及该来源恰为 preferred 的语言数（这些语言需排除）。"""
    rows = list(LanguageVitality.objects.filter(source_id=source_id)
                .values('code_id', 'endangerment_level', 'id').order_by('id'))
    levels: dict[int, str] = {}
    for row in rows:
        lv = map_raw_level(row['endangerment_level'])
        if lv in ORDER_INDEX and row['code_id'] not in levels:
            levels[row['code_id']] = lv
    preferred = {r['code_id'] for r in LanguageVitality.objects.filter(
        preferred=True, source_id=source_id).values('code_id')}
    return levels, len(preferred)


def compare_pairs(reference: pd.Series, predicted: pd.Series, name: str) -> dict:
    """reference 视为对照标准，predicted 视为模型/标签输出。"""
    mask = reference.isin(ORDER_INDEX) & predicted.isin(ORDER_INDEX)
    ref = reference[mask].to_numpy()
    pred = predicted[mask].to_numpy()
    if len(ref) == 0:
        return {'name': name, 'n': 0}
    exact = float(np.mean(ref == pred))
    diff = np.array([abs(ORDER_INDEX[a] - ORDER_INDEX[b]) for a, b in zip(ref, pred)])
    adjacent = float(np.mean(diff <= 1))
    cm = confusion_matrix(ref, pred, labels=LEVEL_ORDER)
    per_class = {}
    for i, cls in enumerate(LEVEL_ORDER):
        support = int(cm[i].sum())
        hit = int(cm[i, i])
        per_class[cls] = {
            'support': support,
            'recall': round(hit / support, 6) if support else None,
            'f1': round(float(f1_score(ref, pred, labels=[cls], average='macro', zero_division=0)), 6),
        }
    return {
        'name': name,
        'n': int(len(ref)),
        'exact_agreement': round(exact, 6),
        'adjacent_agreement_within_1': round(adjacent, 6),
        'quadratic_weighted_kappa': round(float(cohen_kappa_score(
            ref, pred, labels=LEVEL_ORDER, weights='quadratic')), 6),
        'linear_weighted_kappa': round(float(cohen_kappa_score(
            ref, pred, labels=LEVEL_ORDER, weights='linear')), 6),
        'macro_f1': round(float(f1_score(ref, pred, labels=LEVEL_ORDER,
                                         average='macro', zero_division=0)), 6),
        'per_class': per_class,
        'confusion_matrix_rows_reference_cols_pred': cm.tolist(),
        'mean_abs_level_gap': round(float(diff.mean()), 6),
    }


def external_validation(data: pd.DataFrame) -> dict:
    sources = {
        'unesco_atlas': 1521,
        'ethnologue_18th': 98650,
        'ethnologue_16th': 1511,
        'glottolog': 90420,
        'glottolog_2_3': 98057,
    }
    availability = {}
    results = {}
    for name, sid in sources.items():
        levels, pref_n = db_source_levels(sid)
        mapped = sum(1 for c in data['code_id'] if c in levels)
        availability[name] = {
            'source_id': sid,
            'mapped_nonempty': mapped,
            'preferred_overlap': pref_n,
        }
        if mapped == 0:
            continue
        ext = data['code_id'].map(levels)
        pref_ids = set(LanguageVitality.objects.filter(
            preferred=True, source_id=sid).values_list('code_id', flat=True))
        keep = ~data['code_id'].isin(pref_ids)
        results[f'model_oof_vs_{name}'] = compare_pairs(
            ext[keep], data.loc[keep, 'predicted_level'], f'model_vs_{name}')
        results[f'label_vs_{name}'] = compare_pairs(
            ext[keep], data.loc[keep, 'legacy_endangerment_level'], f'label_vs_{name}')

        if name == 'unesco_atlas':
            mask = (keep & ext.isin(ORDER_INDEX) & data['predicted_level'].isin(ORDER_INDEX)
                    & data['legacy_endangerment_level'].isin(ORDER_INDEX))
            ref = ext[mask].to_numpy()
            model_ok = (data.loc[mask, 'predicted_level'].to_numpy() == ref)
            label_ok = (data.loc[mask, 'legacy_endangerment_level'].to_numpy() == ref)
            b = int((model_ok & ~label_ok).sum())
            c = int((~model_ok & label_ok).sum())
            results['mcnemar_model_vs_label_against_atlas'] = {
                'n': int(mask.sum()),
                'model_correct_label_wrong': b,
                'model_wrong_label_correct': c,
                'exact_binomial_p': round(float(binomtest(b, b + c, 0.5).pvalue), 6) if b + c else None,
                'note': '检验“模型与 Atlas 的一致率”是否显著不同于“标签与 Atlas 的一致率”',
            }

    # Glottolog 的空值情况单独记录，避免被误读成“没有数据”
    glotto_empty = int(LanguageVitality.objects.filter(
        source_id__in=[90420, 98057], endangerment_level='').count())
    availability['glottolog']['empty_rows'] = glotto_empty
    return {
        'availability': availability,
        'comparisons': results,
        'primary': 'model_oof_vs_unesco_atlas',
        'caveat': ('UNESCO Atlas 与标签同属 UNESCO 分级框架，且 ELCat 编制时可能参考过；'
                   '它是异源编目，不是完全独立的第三方验证。'),
    }


# ---------------------------------------------------------------------------
# 二、内部一致性
# ---------------------------------------------------------------------------
def fit_oof(X: pd.DataFrame, y: np.ndarray, splits, seed: int) -> np.ndarray:
    proba = np.zeros((len(y), len(LEVEL_ORDER)))
    for tr, te in splits:
        model = SklearnRiskModel(LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, class_weight='balanced',
            n_jobs=-1, random_state=seed, verbose=-1,
        ))
        model.fit(X.iloc[tr].reset_index(drop=True), y[tr])
        raw = model.predict_proba(X.iloc[te].reset_index(drop=True))
        classes = list(model.estimator.classes_)
        proba[te] = raw[:, [classes.index(c) for c in LEVEL_ORDER]]
    return proba


def rank_key(proba: np.ndarray, code_ids: np.ndarray) -> pd.DataFrame:
    score = proba @ RISK_WEIGHTS_5
    expected = proba @ np.arange(len(LEVEL_ORDER))
    return pd.DataFrame({
        'code_id': code_ids,
        'risk_score': score,
        'expected_level': expected,
        'p_extinct': proba[:, LEVEL_ORDER.index('extinct')],
        'p_seriously': proba[:, LEVEL_ORDER.index('seriously_endangered')],
    })


def top_ids(frame: pd.DataFrame, by: str, n: int = TOP_N) -> list[int]:
    if by == 'risk_score':
        sub = frame.sort_values(
            ['risk_score', 'p_extinct', 'p_seriously', 'code_id'],
            ascending=[False, False, False, True])
    else:
        sub = frame.sort_values(['expected_level', 'p_extinct', 'code_id'],
                                ascending=[False, False, True])
    return sub['code_id'].head(n).tolist()


def overlap(a: list[int], b: list[int]) -> dict:
    sa, sb = set(a), set(b)
    return {
        'overlap': len(sa & sb),
        'jaccard': round(len(sa & sb) / len(sa | sb), 6),
        'share_of_top50': round(len(sa & sb) / len(a), 6),
    }


def internal_validation(data: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                        splits, groups: pd.Series) -> dict:
    code_ids = data['code_id'].to_numpy()
    seed_frames = {}
    for seed in SEEDS:
        proba = fit_oof(X, y, splits, seed)
        seed_frames[seed] = rank_key(proba, code_ids)
        log(f'  seed {seed} done')

    ref = seed_frames[SEEDS[0]]
    ref_top = top_ids(ref, 'risk_score')
    ref_top_expected = top_ids(ref, 'expected_level')

    stability = {'reference_seed': SEEDS[0], 'seeds': {}}
    ref_top100 = top_ids(ref, 'risk_score', 100)
    for seed in SEEDS:
        f = seed_frames[seed]
        stability['seeds'][str(seed)] = {
            'top50_overlap_with_reference': overlap(ref_top, top_ids(f, 'risk_score')),
            'top100_overlap_with_reference': overlap(
                ref_top100, top_ids(f, 'risk_score', 100)),
            'top50_expected_level_overlap': overlap(
                ref_top_expected, top_ids(f, 'expected_level')),
            'spearman_risk_score': round(float(spearmanr(
                ref['risk_score'], f['risk_score']).statistic), 6),
        }

    # 换一份折划分（同方案、不同 shuffle 种子）
    alt_splits = list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=ALT_SPLIT_SEED).split(X, y, groups=groups))
    alt_proba = fit_oof(X, y, alt_splits, SEEDS[0])
    alt_frame = rank_key(alt_proba, code_ids)
    stability['alt_split'] = {
        'scheme': 'group_family_5fold',
        'shuffle_seed': ALT_SPLIT_SEED,
        'top50_overlap_with_reference': overlap(ref_top, top_ids(alt_frame, 'risk_score')),
        'top50_expected_level_overlap': overlap(
            ref_top_expected, top_ids(alt_frame, 'expected_level')),
        'spearman_risk_score': round(float(spearmanr(
            ref['risk_score'], alt_frame['risk_score']).statistic), 6),
    }

    # 分档精确率（用 OOF 产物自带的 risk_band）
    band_rows = {}
    actual_high = data['target_5'].isin(HIGH_LEVELS)
    actual_low = data['target_5'].isin(LOW_LEVELS)
    base_rate = float(actual_high.mean())
    for band in ['critical', 'high', 'medium', 'low']:
        mask = data['risk_band'] == band
        n = int(mask.sum())
        band_rows[band] = {
            'n': n,
            'share_of_all': round(n / len(data), 6),
            'actual_high_risk': int(actual_high[mask].sum()),
            'precision_high_risk': round(float(actual_high[mask].mean()), 6) if n else None,
            'actual_low_risk': int(actual_low[mask].sum()),
            'mean_risk_score': round(float(data.loc[mask, 'risk_score'].mean()), 4) if n else None,
        }

    thresholds = {}
    for name, cond in {
        'critical_only(score>=80)': data['risk_score'] >= 80,
        'high_and_critical(score>=60)': data['risk_score'] >= 60,
    }.items():
        tp = int((cond & actual_high).sum())
        fp_low = int((cond & actual_low).sum())
        over_alert_endangered = int((cond & (data['target_5'] == 'endangered')).sum())
        fp_all = int((cond & ~actual_high).sum())
        fn = int((~cond & actual_high).sum())
        tn = int((~cond & ~actual_high).sum())
        precision_strict = tp / (tp + fp_all) if tp + fp_all else None
        precision_low_only = tp / (tp + fp_low) if tp + fp_low else None
        recall = tp / (tp + fn) if tp + fn else None
        thresholds[name] = {
            'alerted': int(cond.sum()),
            'tp': tp, 'fp_low_only': fp_low, 'fp_all_not_high': fp_all,
            'over_alert_endangered': over_alert_endangered,
            'fn': fn, 'tn': tn,
            'precision_strict': round(precision_strict, 6) if precision_strict else None,
            'precision_low_only': round(precision_low_only, 6) if precision_low_only else None,
            'recall': round(recall, 6) if recall is not None else None,
            'lift_vs_base_rate_strict': round(precision_strict / base_rate, 4) if precision_strict else None,
            'cost_per_1000': {
                str(r): round((r * fn + fp_all) / len(data) * 1000, 2) for r in (2, 5, 10, 20)
            },
            'cost_note': ('C_miss = r, C_false_alarm = 1；误报含“实际为 endangered”的过度预警；'
                          'r 为政策参数，非货币单位'),
        }

    return {
        'seeds': SEEDS,
        'stability': stability,
        'bands': band_rows,
        'base_rate_high_risk': round(base_rate, 6),
        'thresholds': thresholds,
        'ranking_note': ('风险分在 100 处饱和：2026-11 全量 OOF 中 193 条得分恰为 100.0，'
                         '因此 TOP-50 需要确定性 tie-break（p_extinct → p_seriously → code_id）。'),
    }


# ---------------------------------------------------------------------------
# 三、TOP-50 核查表（SHAP top3）
# ---------------------------------------------------------------------------
def shap_top3(model: dict, frame: pd.DataFrame) -> list[list[dict]]:
    est = model['estimator']
    pre = model['preprocessor']
    features = list(model['feature_columns'])
    Xt = pre.transform(prepare_frame(frame, features))
    names = list(pre.get_feature_names_out())
    raw_names = []
    for name in names:
        base = name.split('__', 1)[1] if '__' in name else name
        matched = base
        for f in sorted(features, key=len, reverse=True):
            if base == f or base.startswith(f + '_'):
                matched = f
                break
        raw_names.append(matched)

    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    elif values.ndim == 2:
        arr = values[..., None]
    else:
        arr = values

    pred_idx = [LEVEL_ORDER.index(v) for v in frame['predicted_level']]
    out = []
    for i, k in enumerate(pred_idx):
        vals = arr[i, :, k]
        agg: dict[str, float] = {}
        for j, raw in enumerate(raw_names):
            agg[raw] = agg.get(raw, 0.0) + float(vals[j])
        top = sorted(agg.items(), key=lambda x: -abs(x[1]))[:3]
        out.append([{'feature': f, 'shap': round(v, 4)} for f, v in top])
    return out


def preliminary_judgement(row: pd.Series) -> str:
    """规则生成的初步判断，非专家结论。

    只用现有字段（人口、传承、趋势、与旧分级的差值），不做语义判断。
    """
    speakers = row['speakers']
    transmission = str(row['transmission'])
    predicted = row['predicted_level']
    legacy = row['legacy_endangerment_level']
    pred_i = ORDER_INDEX.get(predicted)
    legacy_i = ORDER_INDEX.get(legacy)

    if pd.notna(speakers) and speakers == 0:
        return '初步合理：使用者为 0（已无在世使用者）'
    if pred_i is not None and legacy_i is not None and pred_i > legacy_i:
        return '存疑：模型比现有分级更悲观（差 1 档以上）'
    if pd.notna(speakers) and speakers >= 1000:
        return '存疑：人口规模与“高危”不一致，需核实口径'
    if pd.notna(speakers) and speakers <= 10:
        return '初步合理：使用者 ≤10，与高危一致'
    if transmission in {'5', '6', '7', '8'}:
        return '初步合理：传承中断/仅老年使用者'
    return '需人工复核：人口与传承证据不足以单独定级'


def build_top50(data: pd.DataFrame) -> pd.DataFrame:
    """两张表：生产口径原始 TOP-50，以及“仍有使用者”的可行动 TOP-50。

    原始榜按 risk_score 排序时会被 193 条 100.0 分记录占满，且全部是 speakers=0 的
    已灭绝语言；这本身是重要发现，因此保留并与可行动榜并列输出。
    """
    model = joblib.load(MODEL_PATH)

    def rank(frame: pd.DataFrame) -> pd.DataFrame:
        ranked = frame.sort_values(
            ['risk_score', 'p_extinct', 'p_seriously_endangered', 'code_id'],
            ascending=[False, False, False, True]).head(TOP_N).reset_index(drop=True)
        top3 = shap_top3(model, ranked)
        ranked['shap_top3'] = [
            '；'.join(f"{item['feature']}({item['shap']:+.3f})" for item in row)
            for row in top3
        ]
        ranked['shap_top3_json'] = [json.dumps(row, ensure_ascii=False) for row in top3]
        ranked['judgement'] = ranked.apply(preliminary_judgement, axis=1)
        ranked['judgement_note'] = '规则生成，非专家结论'
        return ranked

    raw = rank(data)
    raw['list_type'] = 'raw_risk_score'

    pool = data[(data['speakers'] > 0) & data['predicted_level'].isin(HIGH_LEVELS)]
    actionable = rank(pool)
    actionable['list_type'] = 'actionable_speakers_gt_0'

    combined = pd.concat([raw, actionable], ignore_index=True)
    cols = [
        'list_type', 'code_id', 'language_name', 'country', 'family_top',
        'predicted_level', 'predicted_level_zh', 'risk_score', 'risk_band',
        'confidence', 'legacy_endangerment_level', 'speakers', 'transmission',
        'speaker_trend', 'shap_top3', 'judgement', 'judgement_note',
    ]
    return combined[cols]


def summarize_top50(frame: pd.DataFrame) -> dict:
    out = {}
    for list_type, sub in frame.groupby('list_type'):
        out[list_type] = {
            'n': int(len(sub)),
            'band_counts': sub['risk_band'].value_counts().to_dict(),
            'predicted_level_counts': sub['predicted_level'].value_counts().to_dict(),
            'judgement_counts': sub['judgement'].value_counts().to_dict(),
            'same_as_label': int((sub['predicted_level'] == sub['legacy_endangerment_level']).sum()),
            'speakers_zero': int((sub['speakers'] == 0).sum()),
            'countries': sub['country'].value_counts().head(8).to_dict(),
        }
    return out


def main() -> None:
    data = pd.read_csv(DATASET_PATH)
    data = data[data['trainable'].astype(bool)].reset_index(drop=True)
    oof = pd.read_csv(OOF_PATH)
    data = data.merge(
        oof[['code_id', 'predicted_level', 'predicted_level_zh', 'risk_score',
             'risk_band', 'confidence'] + [f'p_{c}' for c in LEVEL_ORDER]],
        on='code_id', how='left', validate='one_to_one')
    assert data['predicted_level'].notna().all(), 'OOF 预测未覆盖全部可训练语言'
    log(f'trainable={len(data)} OOF 覆盖={data["predicted_level"].notna().sum()}')

    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    features = list(meta['feature_columns'])
    X = prepare_frame(data, features)
    y = data['target_5'].to_numpy()
    groups = data['family_top']
    splits_payload = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
    splits = [(np.array(f['train_idx']), np.array(f['test_idx']))
              for f in sorted(splits_payload['folds'], key=lambda x: x['fold'])]

    log('异源对照...')
    external = external_validation(data)

    log('内部一致性（多种子 + 换折）...')
    internal = internal_validation(data, X, y, splits, groups)

    log('TOP-50 + SHAP...')
    top50 = build_top50(data)

    label_map = dict(zip(data['code_id'], data['legacy_endangerment_level']))
    pref_levels: dict[int, str] = {}
    for r in LanguageVitality.objects.filter(preferred=True).values(
            'code_id', 'endangerment_level'):
        if r['code_id'] in pref_levels:
            continue
        lv = map_raw_level(r['endangerment_level'])
        if lv in ORDER_INDEX:
            pref_levels[r['code_id']] = lv
    pref_pairs = [(lv, label_map[c]) for c, lv in pref_levels.items()
                  if label_map.get(c) in ORDER_INDEX]
    pref_agree = sum(1 for a, b in pref_pairs if a == b)

    payload = {
        'generated_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'dataset': str(DATASET_PATH),
        'oof_artifact': str(OOF_PATH),
        'splits_artifact': str(SPLITS_PATH),
        'n_trainable': int(len(data)),
        'label_source': {
            'field': 'legacy_endangerment_level',
            'preferred_rows': len(pref_pairs),
            'preferred_agree': pref_agree,
            'preferred_agreement': round(pref_agree / len(pref_pairs), 6),
        },
        'external': external,
        'internal': internal,
        'top50_summary': summarize_top50(top50),
    }
    dump_json(OUT_JSON, payload)

    tmp = OUT_TOP50.with_name(OUT_TOP50.name + '.tmp')
    top50.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, OUT_TOP50)
    log(f'写出 {OUT_JSON} 与 {OUT_TOP50}')


if __name__ == '__main__':
    main()
