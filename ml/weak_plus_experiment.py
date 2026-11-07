"""
weak_plus（正式口径，16 列）：弱信息 9 列 + 7 个低信息量派生特征。

正式特征集（16 列，见 ml/feature_groups.json → weak_plus_features）：
  弱信息 9 列
    latitude, longitude, has_location, country, region,
    family_top, family_depth, classification_depth, neighbor_density_100km
  派生 7 列（均不接外部数据源）
    A 文献时间分布（language_source.year，有效窗口 1800–2017）
      1. bib_recent5_count      有效年份 ≥ 2013（库内最大年份 2017 的近 5 年）的文献条数
      2. bib_year_span          有效年份跨度 max−min（无有效年份为 NaN，仅 1 条为 0）
      3. bib_year_min           最早有效年份（无则 NaN）
      4. bib_year_variance      有效年份的方差（不足 2 条记 0；无则 NaN）
    B 地理与语系邻域（只用坐标/语系/区域计数，不使用任何使用者人数或档案评估字段）
      5. same_family_other_count   同 family_top 的其他可训练语言数（family=unknown 记 NaN）
      6. same_region_other_count   同 region 的其他可训练语言数（region=unknown 记 NaN）
      7. neighbor_count_50km       50km 内其他可训练语言数（无坐标为 NaN）

**明确排除**：dist_nearest_extinct_km 由标签（target_5 == extinct）派生，
禁止用于任何评估指标，也不进入正式特征集；本脚本只把它作为"反例"重跑一遍留证。

口径：
  - splits_group_family_5fold.json 的同一份折索引与同一 holdout（2,258 / 1,204）
  - 同一 trainable 3,462 行、同一 SklearnRiskModel 预处理（训练折内中位数填充 / unknown + OneHot）
  - 模型：LightGBM、RandomForest、XGBoost；同时跑 9 列基线用于逐折配对比较

产物（final 命名，不覆盖旧文件）：
  ml/reports/weak_plus_final_experiment.json
  ml/reports/weak_plus_final_results.csv
  ml/reports/weak_plus_final_features.csv
旧的 17 列产物（weak_plus_experiment.json / weak_plus_results.csv / weak_plus_features.csv）
作为反例证据保留，不删除、不覆盖。

复算：
  python ml/weak_plus_experiment.py
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
from django.db import connection  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics.pairwise import haversine_distances  # noqa: E402

from train_models import (  # noqa: E402
    CATEGORICAL, DATASET_PATH, LEVEL_ORDER, META_PATH, REPORT_DIR, SEED,
    SPLITS_PATH, build_weak_matrix, compute_metrics, dump_json,
    load_feature_groups, log, model_factories, prepare_frame,
    run_cv, summarize_folds,
)

MODEL_NAMES = ['LightGBM', 'RandomForest', 'XGBoost']
OUT_JSON = REPORT_DIR / 'weak_plus_final_experiment.json'
OUT_CSV = REPORT_DIR / 'weak_plus_final_results.csv'
OUT_FEATURES = REPORT_DIR / 'weak_plus_final_features.csv'

YEAR_MIN, YEAR_MAX = 1800, 2017
RECENT_WINDOW_START = YEAR_MAX - 4  # 2013
EARTH_RADIUS_KM = 6371.0088
EXTINCT_IDX = LEVEL_ORDER.index('extinct')

# 由标签派生的列：禁止用于任何评估指标，默认不进入正式特征集。
LABEL_DERIVED_FEATURES = ['dist_nearest_extinct_km']

NEW_FEATURES = {
    'bib_recent5_count': {
        'group': 'A 文献时间分布',
        'source': 'bibliography_link.code_id → language_source.year',
        'formula': f'count(year >= {RECENT_WINDOW_START})，年份有效区间 [1800, 2017]',
        'status': 'official',
    },
    'bib_year_span': {
        'group': 'A 文献时间分布',
        'source': '同上',
        'formula': 'max(year) − min(year)；无有效年份 NaN，仅 1 条记 0',
        'status': 'official',
    },
    'bib_year_min': {
        'group': 'A 文献时间分布',
        'source': '同上',
        'formula': 'min(year)；无有效年份 NaN',
        'status': 'official',
    },
    'bib_year_variance': {
        'group': 'A 文献时间分布',
        'source': '同上',
        'formula': 'var(year)（总体方差）；不足 2 条记 0，无有效年份 NaN',
        'status': 'official',
    },
    'same_family_other_count': {
        'group': 'B 地理与语系邻域',
        'source': 'endangerment_dataset.family_top',
        'formula': '同 family_top 的可训练语言数 − 1；family=unknown 记 NaN',
        'status': 'official',
    },
    'same_region_other_count': {
        'group': 'B 地理与语系邻域',
        'source': 'endangerment_dataset.region',
        'formula': '同 region 的可训练语言数 − 1；region=unknown 记 NaN',
        'status': 'official',
    },
    'neighbor_count_50km': {
        'group': 'B 地理与语系邻域',
        'source': 'endangerment_dataset.latitude/longitude',
        'formula': 'haversine 距离 ≤ 50km 的其他可训练语言数；无坐标 NaN',
        'status': 'official',
    },
    'dist_nearest_extinct_km': {
        'group': 'C 距离最近灭绝语言（已剔除）',
        'source': '坐标 + 训练折内的 target_5==extinct',
        'formula': 'haversine 到最近其他 extinct 语言的距离（km），按折使用训练部分标签，排除自身',
        'status': 'forbidden_label_derived',
    },
}

RED_LINE = ('邻域/语系聚合只使用坐标、family_top、region 计数，未使用 speakers、'
            'max_speaker_number、speaker_trend、transmission、domain_of_use 及其派生量；'
            '正式特征集不含任何由标签派生的列')

OFFICIAL_NEW_FEATURES = {k for k, v in NEW_FEATURES.items() if v['status'] == 'official'}


def load_weak_plus_spec(groups_payload: dict) -> dict:
    """读取 weak_plus 正式定义，并在加载时校验列数/重复/禁用列。"""
    spec = groups_payload.get('weak_plus_features')
    if not spec:
        raise ValueError('feature_groups.json 缺少 weak_plus_features 定义')
    columns = list(spec.get('columns', []))
    expected_n = int(spec.get('n_features', -1))
    if expected_n != len(columns):
        raise ValueError(
            f'weak_plus 列数不一致：n_features={expected_n}，columns 实际 {len(columns)} 列；'
            '加了列就必须同步更新 feature_groups.json')
    if len(set(columns)) != len(columns):
        raise ValueError('weak_plus 列名存在重复')
    forbidden = set(spec.get('forbidden_label_derived', [])) | set(LABEL_DERIVED_FEATURES)
    hit = forbidden & set(columns)
    if hit:
        raise ValueError(f'weak_plus 正式特征集混入标签派生特征：{sorted(hit)}')
    return spec


def bib_year_features(code_ids: set) -> pd.DataFrame:
    """按 code_id 聚合文献年份。"""
    with connection.cursor() as cur:
        cur.execute("""
            SELECT bl.code_id, ls.year
            FROM bibliography_link bl
            JOIN language_source ls ON bl.source_id = ls.id
            WHERE ls.year BETWEEN %s AND %s
        """, [YEAR_MIN, YEAR_MAX])
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=['code_id', 'year'])
    frame = frame[frame['code_id'].isin(code_ids)]

    grouped = frame.groupby('code_id')['year']
    out = pd.DataFrame({
        'bib_year_min': grouped.min(),
        'bib_year_max': grouped.max(),
        'bib_year_count': grouped.size(),
        'bib_year_variance': grouped.var(ddof=0),
        'bib_recent5_count': frame.assign(
            is_recent=(frame['year'] >= RECENT_WINDOW_START).astype(int)
        ).groupby('code_id')['is_recent'].sum(),
    })
    out['bib_year_span'] = out['bib_year_max'] - out['bib_year_min']
    out.loc[out['bib_year_count'] < 2, 'bib_year_variance'] = 0.0
    out['bib_year_variance'] = out['bib_year_variance'].fillna(0.0)
    return out[['bib_recent5_count', 'bib_year_span', 'bib_year_min', 'bib_year_variance']]


def distance_matrix(frame: pd.DataFrame) -> np.ndarray:
    coords = frame[['latitude', 'longitude']].to_numpy(dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    dist = np.full((len(frame), len(frame)), np.inf, dtype=float)
    if valid.sum() > 0:
        radians = np.radians(coords[valid])
        block = haversine_distances(radians) * EARTH_RADIUS_KM
        idx = np.where(valid)[0]
        dist[np.ix_(idx, idx)] = block
    np.fill_diagonal(dist, np.inf)
    return dist


def geo_family_features(frame: pd.DataFrame, dist: np.ndarray) -> pd.DataFrame:
    family = frame['family_top'].fillna('unknown').astype(str).str.strip()
    region = frame['region'].fillna('unknown').astype(str).str.strip()

    family_count = family.map(family.value_counts()).astype(float)
    region_count = region.map(region.value_counts()).astype(float)
    family_count[family == 'unknown'] = np.nan
    region_count[region == 'unknown'] = np.nan

    has_coord = np.isfinite(frame[['latitude', 'longitude']].to_numpy(dtype=float)).all(axis=1)
    neighbor_50 = np.where(has_coord, (dist <= 50.0).sum(axis=1), np.nan)

    return pd.DataFrame({
        'same_family_other_count': family_count.to_numpy() - 1.0,
        'same_region_other_count': region_count.to_numpy() - 1.0,
        'neighbor_count_50km': neighbor_50,
    }, index=frame.index)


def extinct_distance(dist: np.ndarray, y: np.ndarray, reference_mask: np.ndarray,
                     has_coord: np.ndarray) -> np.ndarray:
    """到最近其他 extinct 语言的 haversine 距离；reference 只用传入的训练索引。

    警告：该特征由标签（target_5 == extinct）派生，禁止用于任何评估指标，
    默认不进入 weak_plus 正式特征集（16 列）。本脚本仅用于第 5 节的
    17 列反例重跑，以便留下"加它 CV 降、去它 CV 升"的证据。
    """
    ref = np.where(reference_mask & (y == EXTINCT_IDX) & has_coord)[0]
    out = np.full(len(y), np.nan, dtype=float)
    if len(ref) == 0:
        return out
    block = dist[:, ref].copy()
    pos = {idx: j for j, idx in enumerate(ref)}
    for idx in ref:
        block[idx, pos[idx]] = np.inf
    best = block.min(axis=1)
    out[:] = np.where(np.isfinite(best), best, np.nan)
    return out


def paired_delta(base_folds: dict, new_folds: dict) -> dict:
    diffs = np.array([new_folds[f] - base_folds[f] for f in sorted(base_folds)])
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return {
        'per_fold_new': [float(new_folds[f]) for f in sorted(base_folds)],
        'per_fold_base': [float(base_folds[f]) for f in sorted(base_folds)],
        'per_fold_diff': [float(d) for d in diffs],
        'mean_diff': mean,
        'se_diff': se,
        't_approx': float(mean / se) if se > 0 else None,
        'exceeds_2se': bool(abs(mean) > 2 * se) if se > 0 else None,
    }


def main() -> None:
    started = time.time()

    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    features = [f for f in meta['features'] if f in df.columns]
    trainable = df[df['trainable'] == 1].reset_index(drop=True)
    for col in CATEGORICAL:
        if col not in features:
            features.append(col)

    X_full = prepare_frame(trainable, features)
    y5 = trainable['target_5'].map({c: i for i, c in enumerate(LEVEL_ORDER)}).astype(int).values
    groups_payload = load_feature_groups(features)
    weak_plus_spec = load_weak_plus_spec(groups_payload)
    official_columns = list(weak_plus_spec['columns'])
    X_weak, weak_cols, dropped = build_weak_matrix(X_full, groups_payload)

    payload = json.loads(SPLITS_PATH.read_text(encoding='utf-8'))
    if payload.get('n_rows') != len(trainable):
        raise ValueError('折索引行数与数据集不一致')
    folds = [(np.array(f['train_idx'], dtype=int), np.array(f['test_idx'], dtype=int))
             for f in payload['folds']]
    holdout_tr = np.array(payload['holdout']['train_idx'], dtype=int)
    holdout_te = np.array(payload['holdout']['test_idx'], dtype=int)

    # ---- 派生特征（只含正式 16 列需要的 7 个）----
    code_ids = set(trainable['code_id'].astype(int))
    bib = bib_year_features(code_ids)
    dist = distance_matrix(trainable)
    geo = geo_family_features(trainable, dist)
    has_coord = np.isfinite(trainable[['latitude', 'longitude']].to_numpy(dtype=float)).all(axis=1)

    derived = pd.DataFrame({'code_id': trainable['code_id'].astype(int)})
    derived = derived.merge(bib, left_on='code_id', right_index=True, how='left')
    derived['bib_recent5_count'] = derived['bib_recent5_count'].fillna(0.0)
    derived.index = trainable.index

    X_plus = pd.concat([X_weak, geo, derived.drop(columns=['code_id'])], axis=1)

    # 加载后校验：列集合必须与 feature_groups.json 完全一致，且不得含标签派生列
    missing = [c for c in official_columns if c not in X_plus.columns]
    extra = [c for c in X_plus.columns if c not in set(official_columns)]
    if missing or extra or X_plus.shape[1] != len(official_columns):
        raise ValueError(
            f'weak_plus 矩阵与分组定义不一致：missing={missing} extra={extra} '
            f'实际 {X_plus.shape[1]} 列 / 定义 {len(official_columns)} 列')
    hit = [c for c in X_plus.columns if c in set(LABEL_DERIVED_FEATURES)]
    if hit:
        raise ValueError(f'正式特征集混入标签派生列，禁止继续：{hit}')
    X_plus = X_plus[official_columns]
    log(f'weak_plus 正式列数 {X_plus.shape[1]}（weak {X_weak.shape[1]} + 新增 7）')

    # 特征快照（只写正式 16 列；17 列反例不进快照）
    snapshot = pd.DataFrame({
        'code_id': trainable['code_id'].astype(int),
        'language_name': trainable['language_name'],
        'target_5': trainable['target_5'],
    })
    snapshot = pd.concat([snapshot, X_plus.reset_index(drop=True)], axis=1)
    snapshot.to_csv(OUT_FEATURES, index=False, encoding='utf-8-sig')

    core = model_factories(len(LEVEL_ORDER), include_torch=False)
    models = {name: core[name] for name in MODEL_NAMES}

    # 17 列反例专用：每折重建 dist_nearest_extinct_km（标签派生，仅用于反例）
    X_label = X_plus.copy()
    X_label['dist_nearest_extinct_km'] = np.nan

    def fit_predict_label(train_idx, test_idx, reference_mask, factory):
        feat = extinct_distance(dist, y5, reference_mask, has_coord)
        X_tr = X_label.iloc[train_idx].copy()
        X_te = X_label.iloc[test_idx].copy()
        X_tr['dist_nearest_extinct_km'] = feat[train_idx]
        X_te['dist_nearest_extinct_km'] = feat[test_idx]
        model = factory()
        model.fit(X_tr.reset_index(drop=True), y5[train_idx])
        proba = model.predict_proba(X_te.reset_index(drop=True))
        return model, X_te.reset_index(drop=True), proba

    results = {'weak_9': {}, 'weak_plus': {}, 'weak_plus_17_counterexample': {}}
    base_folds_map = {}
    for name, factory in models.items():
        # 基线：与既有实验完全相同的 run_cv
        rows, _, _ = run_cv(X_weak, y5, folds, factory, LEVEL_ORDER,
                            'target_5', 'group_family_5fold', name, verbose=False)
        base_summary = summarize_folds(rows, LEVEL_ORDER)
        base_folds = {r['fold']: r['macro_f1'] for r in rows}
        base_folds_map[name] = base_folds

        base_model = factory()
        base_model.fit(X_weak.iloc[holdout_tr].reset_index(drop=True), y5[holdout_tr])
        base_proba = base_model.predict_proba(X_weak.iloc[holdout_te].reset_index(drop=True))
        base_holdout = compute_metrics(
            y5[holdout_te], np.argmax(base_proba, axis=1), base_proba, LEVEL_ORDER)

        # 正式 16 列：同折同预处理，直接复用 run_cv（无折相关列）
        rows_official, _, _ = run_cv(X_plus, y5, folds, factory, LEVEL_ORDER,
                                     'target_5', 'group_family_5fold', name, verbose=False)
        official_summary = summarize_folds(rows_official, LEVEL_ORDER)
        official_folds = {r['fold']: r['macro_f1'] for r in rows_official}

        official_model = factory()
        official_model.fit(X_plus.iloc[holdout_tr].reset_index(drop=True), y5[holdout_tr])
        official_proba = official_model.predict_proba(
            X_plus.iloc[holdout_te].reset_index(drop=True))
        official_holdout = compute_metrics(
            y5[holdout_te], np.argmax(official_proba, axis=1), official_proba, LEVEL_ORDER)

        # 17 列反例（含标签派生特征）：逐折重建距离
        rows_label, label_folds = [], {}
        for fold, (tr, te) in enumerate(folds, start=1):
            _, X_te, proba = fit_predict_label(tr, te, np.isin(np.arange(len(y5)), tr), factory)
            metrics = compute_metrics(y5[te], np.argmax(proba, axis=1), proba, LEVEL_ORDER)
            label_folds[fold] = metrics['macro_f1']
            rows_label.append({**metrics, 'fold': fold})
        label_summary = summarize_folds(
            [{'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
              'weighted_f1': r['weighted_f1'], 'balanced_accuracy': r['balanced_accuracy'],
              **{f'recall_{c}': r['per_class'][c]['recall'] for c in LEVEL_ORDER},
              **{f'f1_{c}': r['per_class'][c]['f1'] for c in LEVEL_ORDER}}
             for r in rows_label], LEVEL_ORDER)

        _, X_ho, proba = fit_predict_label(
            holdout_tr, holdout_te, np.isin(np.arange(len(y5)), holdout_tr), factory)
        label_holdout = compute_metrics(y5[holdout_te], np.argmax(proba, axis=1), proba, LEVEL_ORDER)

        results['weak_9'][name] = {
            'cv': base_summary,
            'per_fold_macro_f1': base_folds,
            'holdout': base_holdout,
        }
        results['weak_plus'][name] = {
            'cv': official_summary,
            'per_fold_macro_f1': official_folds,
            'holdout': official_holdout,
            'paired_delta': paired_delta(base_folds, official_folds),
        }
        results['weak_plus_17_counterexample'][name] = {
            'cv': label_summary,
            'per_fold_macro_f1': label_folds,
            'holdout': label_holdout,
            'paired_delta': paired_delta(base_folds, label_folds),
        }
        log(f'{name}: weak9 {base_summary["macro_f1"]["mean"]:.4f} → '
            f'weak_plus16 {official_summary["macro_f1"]["mean"]:.4f} '
            f'(Δ {results["weak_plus"][name]["paired_delta"]["mean_diff"]:+.4f}, '
            f'SE {results["weak_plus"][name]["paired_delta"]["se_diff"]:.4f}) '
            f'holdout {official_holdout["macro_f1"]:.4f} | 17列反例 '
            f'{label_summary["macro_f1"]["mean"]:.4f}')

    # 置换重要性：正式 16 列 LightGBM，留出集，macro F1 评分
    importance = {}
    if 'LightGBM' in models:
        model = models['LightGBM']()
        model.fit(X_plus.iloc[holdout_tr].reset_index(drop=True), y5[holdout_tr])
        X_ho = X_plus.iloc[holdout_te].reset_index(drop=True)
        perm = permutation_importance(
            model, X_ho, y5[holdout_te], scoring='f1_macro',
            n_repeats=10, random_state=SEED, n_jobs=-1)
        importance = {
            col: {'mean': float(perm.importances_mean[i]),
                  'std': float(perm.importances_std[i]),
                  'is_new': col in OFFICIAL_NEW_FEATURES}
            for i, col in enumerate(X_ho.columns)
        }

    full_cv, full_ho = 0.883598100812527, 0.9008388433901905
    comparison = {}
    for name in MODEL_NAMES:
        official = results['weak_plus'][name]
        comparison[name] = {
            'weak_9_cv': results['weak_9'][name]['cv']['macro_f1'],
            'weak_plus_cv': official['cv']['macro_f1'],
            'weak_9_holdout': results['weak_9'][name]['holdout']['macro_f1'],
            'weak_plus_holdout': official['holdout']['macro_f1'],
            'paired_delta_vs_weak9': official['paired_delta'],
            'delta_vs_0_2878': official['cv']['macro_f1']['mean'] - 0.2878451033742687,
            'gap_to_full_cv': full_cv - official['cv']['macro_f1']['mean'],
            'gap_to_full_holdout': full_ho - official['holdout']['macro_f1'],
        }

    out = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'ml/weak_plus_experiment.py',
        'purpose': 'weak_plus 正式口径（16 列 = 弱信息 9 列 + 7 个低信息量派生特征）对照实验',
        'status': 'final',
        'data': {
            'dataset': 'ml/endangerment_dataset.csv',
            'n_trainable': int(len(trainable)),
            'class_counts': {c: int((y5 == i).sum()) for i, c in enumerate(LEVEL_ORDER)},
        },
        'split': {
            'splits_file': 'ml/reports/splits_group_family_5fold.json',
            'folds': 5,
            'holdout_train': int(len(holdout_tr)),
            'holdout_test': int(len(holdout_te)),
        },
        'bibliography_year_check': {
            'field': 'language_source.year (IntegerField)',
            'links_total': 25611,
            'links_with_year': 22809,
            'valid_window': [YEAR_MIN, YEAR_MAX],
            'links_in_window': 22759,
            'excluded_reason': '排除 1/7/19/27/31/150/190/196/198/199/200/1000 等明显错值，以及 1594–1781 的少量古籍；库内最大年份 2017',
            'trainable_with_ge1_year': 3321,
            'trainable_coverage': round(3321 / len(trainable), 4),
            'recent_window': [RECENT_WINDOW_START, YEAR_MAX],
        },
        'new_features': NEW_FEATURES,
        'forbidden_label_derived': {
            'features': LABEL_DERIVED_FEATURES,
            'rule': '由标签（target_5 == extinct）派生，禁止用于任何评估指标，默认不进入 weak_plus 正式特征集；仅保留为 17 列反例',
        },
        'red_line': RED_LINE,
        'feature_columns': official_columns,
        'n_features': int(X_plus.shape[1]),
        'models': MODEL_NAMES,
        'results': results,
        'comparison': comparison,
        'permutation_importance_holdout_lightgbm': importance,
        'reference_numbers': {
            'weak_9_cv_macro_f1': 0.2878451033742687,
            'weak_9_holdout_macro_f1': 0.2909329439039365,
            'full_cv_macro_f1': full_cv,
            'full_holdout_macro_f1': full_ho,
            'source': 'ml/reports/metrics.json（只读）',
        },
        'runtime_seconds': round(time.time() - started, 1),
    }
    dump_json(OUT_JSON, out)

    rows = []
    n_map = {'weak_9': 9, 'weak_plus': int(X_plus.shape[1]),
             'weak_plus_17_counterexample': int(X_plus.shape[1]) + 1}
    for group in ('weak_9', 'weak_plus', 'weak_plus_17_counterexample'):
        for name in MODEL_NAMES:
            r = results[group][name]
            rows.append({
                'feature_set': group,
                'n_features': n_map[group],
                'model': name,
                'cv_macro_f1_mean': r['cv']['macro_f1']['mean'],
                'cv_macro_f1_std': r['cv']['macro_f1']['std'],
                'holdout_macro_f1': r['holdout']['macro_f1'],
                'holdout_accuracy': r['holdout']['accuracy'],
                'cv_extinct_recall': r['cv']['per_class_recall']['extinct'],
                'holdout_extinct_recall': r['holdout']['per_class']['extinct']['recall'],
                'mean_diff_vs_weak9': (0.0 if group == 'weak_9'
                                       else r['paired_delta']['mean_diff']),
                'se_diff': (0.0 if group == 'weak_9'
                            else r['paired_delta']['se_diff']),
                'exceeds_2se': (None if group == 'weak_9'
                                else r['paired_delta']['exceeds_2se']),
            })
    frame = pd.DataFrame(rows)
    tmp = OUT_CSV.with_name(OUT_CSV.name + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(OUT_CSV)

    log(f'完成，用时 {time.time() - started:.0f}s → '
        f'{OUT_JSON.name}, {OUT_CSV.name}, {OUT_FEATURES.name}')


if __name__ == '__main__':
    main()
