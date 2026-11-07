"""
构建濒危语言风险预测的语言级数据集。

输出：
  ml/endangerment_dataset.csv
  ml/dataset_meta.json

原则：
- 一行一种语言
- 标签使用 legacy_endangerment_level
- 不把 level_of_endangerment、LanguageVitality.endangerment_level 作为特征
- 统计类特征从语言明细表聚合
- 特征口径必须与 Dashboard/management/commands/risk.py 的单条推理一致
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')

import django

django.setup()

import numpy as np
import pandas as pd
from django.db import connection
from django.db.models import Count, Max, Q

from Dashboard.models import (
    BibliographyLink,
    CountryRegion,
    Language,
    LanguageActivity,
    LanguageCode,
    LanguageContext,
    LanguageCountry,
    LanguageDocumentation,
    LanguageLocationDetail,
    LanguageMedia,
    LanguageOrthography,
    LanguageResource,
    LanguageSample,
    LanguageVitality,
    Region,
    SpeakerData,
)

OUTPUT_DIR = BASE_DIR / 'ml'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = OUTPUT_DIR / 'endangerment_dataset.csv'
META_PATH = OUTPUT_DIR / 'dataset_meta.json'

MAIN_LEVELS = ['unsafe', 'danger', 'endangered', 'seriously_endangered', 'extinct']
HIGH_RISK = {'endangered', 'seriously_endangered', 'extinct'}
LOW_RISK = {'unsafe', 'danger'}


def qs_to_df(qs):
    return pd.DataFrame(list(qs))


def merge_counts(base, frame, name):
    if frame.empty:
        base[name] = 0
        return base
    if 'code_id' in frame.columns:
        frame = frame.drop_duplicates('code_id', keep='last')
    return base.merge(frame, on='code_id', how='left')


def build_frame():
    """按当前数据库构建语言级特征表，返回 (DataFrame, 特征列名)。"""
    # -----------------------------------------------------------------------
    # 1. 语言主表
    # -----------------------------------------------------------------------
    lang = qs_to_df(Language.objects.values(
        'language_id', 'language_name', 'number_of_speakers',
        'speaker_trends', 'intergenerational_transmission',
        'language_family', 'country', 'legacy_endangerment_level',
        'additional_notes', 'media',
    )).rename(columns={'language_id': 'code_id'})

    lang['speakers'] = pd.to_numeric(lang['number_of_speakers'], errors='coerce').fillna(0).clip(lower=0)
    lang['log_speakers'] = np.log1p(lang['speakers'])
    lang['has_speakers'] = (lang['speakers'] > 0).astype(int)
    lang['notes_length'] = lang['additional_notes'].fillna('').astype(str).str.len()
    lang['has_media'] = lang['media'].fillna('').astype(str).str.len().gt(0).astype(int)

    lang['family_full'] = lang['language_family'].fillna('unknown').astype(str).str.strip()
    lang['family_top'] = lang['family_full'].apply(
        lambda x: x.split(';')[0].strip() if x and x != 'unknown' else 'unknown'
    )
    lang['family_depth'] = lang['family_full'].apply(
        lambda x: len([p for p in x.split(';') if p.strip()]) if x != 'unknown' else 0
    )
    lang['country'] = lang['country'].fillna('unknown').astype(str).str.strip().replace('', 'unknown')
    lang['speaker_trend'] = lang['speaker_trends'].fillna('unknown').astype(str)
    lang['transmission'] = lang['intergenerational_transmission'].fillna('unknown').astype(str)

    # 同一语言名只保留字段最完整的一条，避免重复记录跨训练/测试集泄漏
    lang['_completeness'] = (
        lang['legacy_endangerment_level'].notna().astype(int)
        + (lang['number_of_speakers'].notna() & (lang['number_of_speakers'] > 0)).astype(int)
        + lang['country'].notna().astype(int)
        + lang['language_family'].notna().astype(int)
        + lang['additional_notes'].fillna('').astype(str).str.len().gt(0).astype(int)
    )
    lang = (
        lang.sort_values(['_completeness', 'code_id'], ascending=[False, False])
            .drop_duplicates('language_name', keep='first')
            .drop(columns=['_completeness'])
            .reset_index(drop=True)
    )

    lang['target_5'] = lang['legacy_endangerment_level'].where(
        lang['legacy_endangerment_level'].isin(MAIN_LEVELS), 'excluded')
    lang['target_binary'] = np.select(
        [
            lang['legacy_endangerment_level'].isin(HIGH_RISK),
            lang['legacy_endangerment_level'].isin(LOW_RISK),
        ],
        ['high_risk', 'low_risk'],
        default='excluded',
    )
    lang['trainable'] = (lang['target_5'] != 'excluded').astype(int)

    # -----------------------------------------------------------------------
    # 2. 样本数量与类型
    # -----------------------------------------------------------------------
    sample = qs_to_df(LanguageSample.objects.values('code_id', 'sample_type').annotate(
        n=Count('id')
    ))
    if not sample.empty:
        sample_pivot = sample.pivot_table(
            index='code_id', columns='sample_type', values='n', fill_value=0
        )
        sample_pivot.columns = [f'sample_{c}' for c in sample_pivot.columns]
        # 只在纯类型列上聚合，避免把汇总列自己算成一种类型或计入样本总数
        sample_type_cols = list(sample_pivot.columns)
        sample_pivot['sample_count'] = sample_pivot[sample_type_cols].sum(axis=1)
        sample_pivot['sample_type_count'] = (sample_pivot[sample_type_cols] > 0).sum(axis=1)
        sample_pivot = sample_pivot.reset_index()
    else:
        sample_pivot = pd.DataFrame({'code_id': []})

    # -----------------------------------------------------------------------
    # 3. 文献关联
    # -----------------------------------------------------------------------
    bib = qs_to_df(BibliographyLink.objects.values('code_id').annotate(
        bibliography_count=Count('id'),
        source_count=Count('source_id', distinct=True),
        cited_count=Count('id', filter=Q(cited=True)),
    ))

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT bl.code_id, COUNT(DISTINCT ls.source_type_id) AS source_type_count
            FROM bibliography_link bl
            JOIN language_source ls ON bl.source_id = ls.id
            GROUP BY bl.code_id
        """)
        source_type_rows = cursor.fetchall()
    source_type = pd.DataFrame(source_type_rows, columns=['code_id', 'source_type_count'])

    # -----------------------------------------------------------------------
    # 4. 活动、正字法、文档化、资源、媒体、语境
    # -----------------------------------------------------------------------
    activity = qs_to_df(LanguageActivity.objects.values('code_id').annotate(
        activity_count=Count('id'),
        activity_type_count=Count('obj_type', distinct=True),
    ))

    orthography = qs_to_df(LanguageOrthography.objects.values('code_id').annotate(
        orthography_count=Count('id'),
        script_count=Count('scripts', distinct=True, filter=~Q(scripts='')),
    ))

    documentation = qs_to_df(LanguageDocumentation.objects.values('code_id').annotate(
        documentation_count=Count('id'),
        max_documentation_index=Max('documentation_index'),
    ))

    resource = qs_to_df(LanguageResource.objects.values('code_id').annotate(
        resource_count=Count('id'),
    ))

    media = qs_to_df(LanguageMedia.objects.values('code_id').annotate(
        media_count=Count('id'),
    ))

    context = qs_to_df(LanguageContext.objects.values('code_id').annotate(
        context_count=Count('id'),
        government_support_flag=Count(
            'id', filter=Q(government_support__isnull=False) & ~Q(government_support='')),
        institutional_support_flag=Count(
            'id', filter=Q(institutional_support__isnull=False) & ~Q(institutional_support='')),
        speaker_attitude_flag=Count(
            'id', filter=Q(speaker_attitude__isnull=False) & ~Q(speaker_attitude='')),
    ))

    # -----------------------------------------------------------------------
    # 5. 位置明细
    # -----------------------------------------------------------------------
    location_count = qs_to_df(LanguageLocationDetail.objects.values('code_id').annotate(
        location_count=Count('id'),
    ))

    loc_rows = qs_to_df(LanguageLocationDetail.objects.exclude(coordinates='').values(
        'code_id', 'coordinates', 'preferred', 'id'
    ))
    if not loc_rows.empty:
        # 与 risk.py 一致：preferred 升序、id 升序，取最后一条（确定性）
        loc_rows = loc_rows.sort_values(['code_id', 'preferred', 'id']).drop_duplicates(
            'code_id', keep='last'
        )

        def parse_coord(value):
            if not value:
                return (np.nan, np.nan)
            first = str(value).split(';')[0].strip()
            parts = [p.strip() for p in first.split(',')]
            if len(parts) < 2:
                return (np.nan, np.nan)
            try:
                lat = float(parts[0])
                lng = float(parts[1])
                return (lat, lng)
            except (TypeError, ValueError):
                return (np.nan, np.nan)

        loc_rows[['latitude', 'longitude']] = loc_rows['coordinates'].apply(
            lambda x: pd.Series(parse_coord(x))
        )
        loc_rows['has_location'] = loc_rows[['latitude', 'longitude']].notna().all(axis=1).astype(int)
        loc_rows = loc_rows[['code_id', 'latitude', 'longitude', 'has_location']]
    else:
        loc_rows = pd.DataFrame({'code_id': []})

    # -----------------------------------------------------------------------
    # 6. 语言-国家、区域
    # -----------------------------------------------------------------------
    country_map = qs_to_df(LanguageCountry.objects.values('language_id', 'country_id'))
    if not country_map.empty:
        country_map = country_map.rename(columns={'language_id': 'code_id'})
        country_count = country_map.groupby('code_id').size().reset_index(name='country_count')
        # 与 risk.py 一致：按 country_id 升序取第一个国家（确定性）
        first_country = country_map.sort_values(['code_id', 'country_id']).drop_duplicates(
            'code_id', keep='first'
        )[['code_id', 'country_id']]

        # 与 risk.py 一致：同一国家按 country_region.id 升序取第一条（确定性）
        country_region = qs_to_df(CountryRegion.objects.values('country_id', 'region_id', 'id'))
        country_region = country_region.sort_values(
            ['country_id', 'id']
        ).drop_duplicates('country_id', keep='first')[['country_id', 'region_id']]
        region_names = qs_to_df(Region.objects.values('id', 'name')).rename(
            columns={'id': 'region_id', 'name': 'region'}
        )
        first_country = first_country.merge(country_region, on='country_id', how='left')
        first_country = first_country.merge(region_names, on='region_id', how='left')
        first_country = first_country.drop_duplicates('code_id', keep='first')
        first_country = first_country[['code_id', 'region']]
    else:
        country_count = pd.DataFrame({'code_id': []})
        first_country = pd.DataFrame({'code_id': []})

    # -----------------------------------------------------------------------
    # 7. 语言代码：ISO、别名、分类深度
    # -----------------------------------------------------------------------
    code = qs_to_df(LanguageCode.objects.values(
        'id', 'iso_code', 'alternate_names', 'classification'
    )).rename(columns={'id': 'code_id'})
    code['has_iso'] = code['iso_code'].fillna('').astype(str).str.len().gt(0).astype(int)
    code['alternate_name_count'] = code['alternate_names'].fillna('').astype(str).apply(
        lambda x: len([p for p in x.split(';') if p.strip()])
    )
    code['classification_depth'] = code['classification'].fillna('').astype(str).apply(
        lambda x: len([p for p in x.split(';') if p.strip()])
    )
    code = code[['code_id', 'has_iso', 'alternate_name_count', 'classification_depth']]

    # -----------------------------------------------------------------------
    # 8. 活力与使用者明细
    # -----------------------------------------------------------------------
    vitality_count = qs_to_df(LanguageVitality.objects.values('code_id').annotate(
        vitality_count=Count('id'),
    ))

    vitality_rows = qs_to_df(LanguageVitality.objects.values(
        'code_id', 'domains_of_use_id', 'preferred', 'id'
    ))
    if not vitality_rows.empty:
        # 与 risk.py 一致：preferred 升序、id 升序，取最后一条（确定性）
        vitality_rows = vitality_rows.sort_values(
            ['code_id', 'preferred', 'id']
        ).drop_duplicates('code_id', keep='last')
        vitality_rows['domain_of_use'] = vitality_rows['domains_of_use_id'].apply(
            lambda x: f'domain_{int(x)}' if pd.notna(x) else 'unknown'
        )
        vitality_rows = vitality_rows[['code_id', 'domain_of_use']]
    else:
        vitality_rows = pd.DataFrame({'code_id': []})

    speaker = qs_to_df(SpeakerData.objects.values('code_id').annotate(
        speaker_records=Count('id'),
        max_speaker_number=Max('speaker_number'),
        max_ethnic_population=Max('ethnic_population'),
        max_semi_speakers=Max('semi_speakers'),
        max_children=Max('children'),
        max_young_adults=Max('young_adults'),
        max_older_adults=Max('older_adults'),
        max_elders=Max('elders'),
    ))

    # -----------------------------------------------------------------------
    # 9. 合并
    # -----------------------------------------------------------------------
    df = lang.copy()
    for frame in [
        sample_pivot, bib, source_type, activity, orthography, documentation,
        resource, media, context, location_count, loc_rows, country_count,
        first_country, code, vitality_count, vitality_rows, speaker,
    ]:
        df = merge_counts(df, frame, 'code_id')

    # 数值列缺失填 0
    numeric_cols = [
        'sample_count', 'sample_type_count', 'bibliography_count', 'source_count',
        'cited_count', 'source_type_count', 'activity_count', 'activity_type_count',
        'orthography_count', 'script_count', 'documentation_count',
        'max_documentation_index', 'resource_count', 'media_count', 'context_count',
        'government_support_flag', 'institutional_support_flag', 'speaker_attitude_flag',
        'has_iso', 'alternate_name_count', 'classification_depth',
        'location_count', 'country_count', 'vitality_count', 'speaker_records',
        'max_speaker_number', 'max_ethnic_population', 'max_semi_speakers',
        'max_children', 'max_young_adults', 'max_older_adults', 'max_elders',
        'has_location',
    ]
    numeric_cols += [c for c in df.columns if c.startswith('sample_')]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 分类型缺失填 unknown
    for col in ['family_top', 'country', 'region', 'speaker_trend', 'transmission', 'domain_of_use']:
        if col not in df.columns:
            df[col] = 'unknown'
        df[col] = df[col].fillna('unknown').astype(str).replace('', 'unknown')

    # 高基数类别合并为 other，避免独热维度爆炸
    for col, keep in [('family_top', 30), ('country', 40)]:
        top = df[col].value_counts().head(keep).index
        df[col] = df[col].where(df[col].isin(top), 'other')

    # 对使用者数量做对数特征
    df['log_ethnic_population'] = np.log1p(df['max_ethnic_population'].clip(lower=0))
    df['log_speaker_number_max'] = np.log1p(df['max_speaker_number'].clip(lower=0))
    df['log_children'] = np.log1p(df['max_children'].clip(lower=0))
    df['log_young_adults'] = np.log1p(df['max_young_adults'].clip(lower=0))
    df['log_older_adults'] = np.log1p(df['max_older_adults'].clip(lower=0))
    df['log_elders'] = np.log1p(df['max_elders'].clip(lower=0))

    # 特征列
    feature_columns = [
        'speakers', 'log_speakers', 'has_speakers', 'notes_length', 'has_media',
        'family_depth', 'alternate_name_count', 'classification_depth', 'has_iso',
        'family_top', 'country', 'region', 'speaker_trend', 'transmission', 'domain_of_use',
        'sample_count', 'sample_type_count', 'bibliography_count', 'source_count',
        'cited_count', 'source_type_count', 'activity_count', 'activity_type_count',
        'orthography_count', 'script_count', 'documentation_count',
        'max_documentation_index', 'resource_count', 'media_count', 'context_count',
        'government_support_flag', 'institutional_support_flag', 'speaker_attitude_flag',
        'location_count', 'country_count', 'vitality_count', 'speaker_records',
        'max_speaker_number', 'max_ethnic_population', 'max_semi_speakers',
        'max_children', 'max_young_adults', 'max_older_adults', 'max_elders',
        'log_ethnic_population', 'log_speaker_number_max', 'log_children',
        'log_young_adults', 'log_older_adults', 'log_elders',
        'latitude', 'longitude', 'has_location',
    ]
    feature_columns += [c for c in df.columns if c.startswith('sample_') and c not in feature_columns]

    # 保证顺序稳定
    feature_columns = list(dict.fromkeys(feature_columns))
    return df, feature_columns


def write_outputs(df, feature_columns):
    """原子写出数据集 CSV 与元信息。"""
    output_columns = [
        'code_id', 'language_name', 'legacy_endangerment_level', 'target_5',
        'target_binary', 'trainable',
    ] + feature_columns
    output_columns = [c for c in output_columns if c in df.columns]
    tmp_dataset = DATASET_PATH.with_name(DATASET_PATH.name + '.tmp')
    df[output_columns].to_csv(tmp_dataset, index=False, encoding='utf-8-sig')
    os.replace(tmp_dataset, DATASET_PATH)  # 原子替换，避免读到半成品数据集

    meta = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        # 记录相对项目根目录的路径，避免把开发机绝对路径写进产物
        'dataset_path': os.path.relpath(DATASET_PATH, BASE_DIR).replace(os.sep, '/'),
        'rows': int(len(df)),
        'trainable_rows': int(df['trainable'].sum()),
        'features': feature_columns,
        'target_5_counts': df['target_5'].value_counts().to_dict(),
        'target_binary_counts': df['target_binary'].value_counts().to_dict(),
        'legacy_target_counts': df['legacy_endangerment_level'].value_counts(dropna=False).to_dict(),
        'family_top_categories': df['family_top'].value_counts().to_dict(),
        'country_categories': df['country'].value_counts().head(20).to_dict(),
        'notes': [
            'level_of_endangerment 未作为特征，避免直接泄漏 ELCat 等级',
            'LanguageVitality.endangerment_level 未作为特征',
            'label 使用 legacy_endangerment_level',
            '同一 language_name 只保留字段最完整的一条，避免重复记录泄漏',
            'holdout 与交叉验证均按语系/国家分组，不随机切分同一语言',
            '风险预测只覆盖 trainable=1 的语言，awakening/safe/unknown 不展示预测',
            'trainable=1 表示标签属于 unsafe/danger/endangered/seriously_endangered/extinct',
            'sample_type_count 只统计数据集的类型列，不包含 sample_count 汇总列',
            '区域取值按 country_id 升序、country_region.id 升序取第一条，与单条推理一致',
        ],
    }
    tmp_meta = META_PATH.with_name(META_PATH.name + '.tmp')
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp_meta, META_PATH)


def main():
    df, feature_columns = build_frame()
    write_outputs(df, feature_columns)
    print(f'dataset rows: {len(df)}')
    print(f'trainable rows: {int(df["trainable"].sum())}')
    print('target_5:')
    print(df['target_5'].value_counts().to_string())
    print('target_binary:')
    print(df['target_binary'].value_counts().to_string())
    print(f'features: {len(feature_columns)}')
    print(f'saved: {DATASET_PATH}')
    print(f'meta: {META_PATH}')


if __name__ == '__main__':
    main()
