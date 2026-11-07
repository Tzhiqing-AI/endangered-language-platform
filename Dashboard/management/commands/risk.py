"""风险预测单条推理入口：python manage.py risk predict --code-id <language_id>。

命令按 code_id 从数据库组装一条特征记录（口径与 ml/build_dataset.py 一致），
调用 ``Dashboard.risk_service.predict_features`` 做推理，只读、不写库。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Max, Q

from Dashboard import risk_service
from Dashboard.common import display_language_name
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
    LanguageSource,
    LanguageVitality,
    Region,
    SpeakerData,
)

# 训练时 dataset 已把这些字段缺失值填 0；lat/lng 保持 NaN 交给中位数填充器
NUMERIC_FEATURES_WITHOUT_DEFAULT = ('latitude', 'longitude')


def _parse_coord(value: str) -> Tuple[float, float]:
    """与 ml/build_dataset.py 的 parse_coord 保持同一口径（多坐标取第一个）。"""
    if not value:
        return float('nan'), float('nan')
    first = str(value).split(';')[0].strip()
    parts = [part.strip() for part in first.split(',')]
    if len(parts) < 2:
        return float('nan'), float('nan')
    try:
        return float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        return float('nan'), float('nan')


def _category(value) -> str:
    return 'unknown' if value in (None, '') else str(value)


def _build_feature_row(
    code_id: int,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """返回 (特征行, 语言基础信息)；语言不存在时返回 (None, None)。"""
    lang = Language.objects.filter(language_id=code_id).values(
        'language_id', 'language_name', 'number_of_speakers', 'speaker_trends',
        'intergenerational_transmission', 'language_family', 'country',
        'additional_notes', 'media',
    ).first()
    if lang is None:
        return None, None

    row: Dict[str, object] = {}

    speakers = max(int(lang['number_of_speakers'] or 0), 0)
    row['speakers'] = float(speakers)
    row['log_speakers'] = math.log1p(speakers)
    row['has_speakers'] = 1.0 if speakers > 0 else 0.0
    row['notes_length'] = float(len(str(lang['additional_notes'] or '')))
    row['has_media'] = 1.0 if len(str(lang['media'] or '')) > 0 else 0.0

    raw_family = str(lang['language_family'] or 'unknown').strip()
    if raw_family in ('', 'unknown'):
        row['family_top'] = 'unknown'
        row['family_depth'] = 0.0
    else:
        top = raw_family.split(';')[0].strip()
        row['family_top'] = top or 'unknown'
        row['family_depth'] = float(len([p for p in raw_family.split(';') if p.strip()]))

    row['country'] = str(lang['country'] or 'unknown').strip() or 'unknown'
    row['speaker_trend'] = _category(lang['speaker_trends'])
    row['transmission'] = _category(lang['intergenerational_transmission'])

    # 语言样本
    sample_count = 0
    sample_types = set()
    sample_rows = LanguageSample.objects.filter(code_id=code_id).values(
        'sample_type').annotate(n=Count('id'))
    for item in sample_rows:
        count = int(item['n'] or 0)
        sample_count += count
        sample_type = (item['sample_type'] or '').strip()
        if count and sample_type:
            sample_types.add(sample_type)
            row[f'sample_{sample_type}'] = float(count)
    row['sample_count'] = float(sample_count)
    row['sample_type_count'] = float(len(sample_types))

    # 文献关联
    bib = BibliographyLink.objects.filter(code_id=code_id).aggregate(
        bibliography_count=Count('id'),
        source_count=Count('source_id', distinct=True),
        cited_count=Count('id', filter=Q(cited=True)),
    )
    for key, value in bib.items():
        row[key] = float(value or 0)
    row['source_type_count'] = float(
        LanguageSource.objects.filter(
            id__in=BibliographyLink.objects.filter(code_id=code_id).values('source_id'),
        ).exclude(source_type_id__isnull=True).values('source_type_id').distinct().count()
    )

    # 活动 / 正字法 / 文档化 / 资源 / 媒体 / 语境
    row.update({
        key: float(value or 0) for key, value in LanguageActivity.objects.filter(
            code_id=code_id).aggregate(
            activity_count=Count('id'),
            activity_type_count=Count('obj_type', distinct=True),
        ).items()
    })
    row.update({
        key: float(value or 0) for key, value in LanguageOrthography.objects.filter(
            code_id=code_id).aggregate(
            orthography_count=Count('id'),
            script_count=Count('scripts', distinct=True, filter=~Q(scripts='')),
        ).items()
    })
    row.update({
        key: float(value or 0) for key, value in LanguageDocumentation.objects.filter(
            code_id=code_id).aggregate(
            documentation_count=Count('id'),
            max_documentation_index=Max('documentation_index'),
        ).items()
    })
    row['resource_count'] = float(
        LanguageResource.objects.filter(code_id=code_id).count())
    row['media_count'] = float(
        LanguageMedia.objects.filter(code_id=code_id).count())
    row.update({
        key: float(value or 0) for key, value in LanguageContext.objects.filter(
            code_id=code_id).aggregate(
            context_count=Count('id'),
            government_support_flag=Count(
                'id', filter=Q(government_support__isnull=False) & ~Q(government_support='')),
            institutional_support_flag=Count(
                'id', filter=Q(institutional_support__isnull=False) & ~Q(institutional_support='')),
            speaker_attitude_flag=Count(
                'id', filter=Q(speaker_attitude__isnull=False) & ~Q(speaker_attitude='')),
        ).items()
    })

    # 位置
    row['location_count'] = float(
        LanguageLocationDetail.objects.filter(code_id=code_id).count())
    location = (
        LanguageLocationDetail.objects.filter(code_id=code_id)
        .exclude(coordinates='')
        .order_by('preferred', 'id')
        .last()
    )
    latitude, longitude = (
        _parse_coord(location.coordinates) if location else (float('nan'), float('nan'))
    )
    if not math.isnan(latitude) and not math.isnan(longitude):
        row['latitude'] = latitude
        row['longitude'] = longitude
        row['has_location'] = 1.0
    else:
        row['has_location'] = 0.0

    # 国家 / 区域
    row['country_count'] = float(
        LanguageCountry.objects.filter(language_id=code_id).count())
    first_country = (
        LanguageCountry.objects.filter(language_id=code_id)
        .order_by('country_id').values('country_id').first()
    )
    region = 'unknown'
    if first_country:
        country_region = (
            CountryRegion.objects.filter(country_id=first_country['country_id'])
            .order_by('id').first()
        )
        if country_region:
            region_row = Region.objects.filter(id=country_region.region_id).values('name').first()
            if region_row and region_row['name']:
                region = region_row['name']
    row['region'] = region

    # 语言代码
    code = LanguageCode.objects.filter(id=code_id).values(
        'iso_code', 'alternate_names', 'classification').first()
    if code:
        row['has_iso'] = 1.0 if len(str(code['iso_code'] or '')) > 0 else 0.0
        row['alternate_name_count'] = float(len([
            part for part in str(code['alternate_names'] or '').split(';') if part.strip()
        ]))
        row['classification_depth'] = float(len([
            part for part in str(code['classification'] or '').split(';') if part.strip()
        ]))
    else:
        row['has_iso'] = 0.0
        row['alternate_name_count'] = 0.0
        row['classification_depth'] = 0.0

    # 活力 / 使用领域
    row['vitality_count'] = float(
        LanguageVitality.objects.filter(code_id=code_id).count())
    vitality = (
        LanguageVitality.objects.filter(code_id=code_id)
        .order_by('preferred', 'id').last()
    )
    if vitality is None or vitality.domains_of_use_id is None:
        row['domain_of_use'] = 'unknown'
    else:
        row['domain_of_use'] = f'domain_{int(vitality.domains_of_use_id)}'

    # 使用者明细
    row.update({
        key: float(value or 0) for key, value in SpeakerData.objects.filter(
            code_id=code_id).aggregate(
            speaker_records=Count('id'),
            max_speaker_number=Max('speaker_number'),
            max_ethnic_population=Max('ethnic_population'),
            max_semi_speakers=Max('semi_speakers'),
            max_children=Max('children'),
            max_young_adults=Max('young_adults'),
            max_older_adults=Max('older_adults'),
            max_elders=Max('elders'),
        ).items()
    })
    log_pairs = {
        'max_ethnic_population': 'log_ethnic_population',
        'max_speaker_number': 'log_speaker_number_max',
        'max_children': 'log_children',
        'max_young_adults': 'log_young_adults',
        'max_older_adults': 'log_older_adults',
        'max_elders': 'log_elders',
    }
    for source, target in log_pairs.items():
        row[target] = math.log1p(max(float(row[source]), 0.0))

    # 缺失特征按训练口径补齐：数值 0、类别 unknown；坐标保留 NaN 由填充器处理
    categorical = set(categorical_columns)
    for column in feature_columns:
        if column in row:
            continue
        if column in NUMERIC_FEATURES_WITHOUT_DEFAULT:
            continue
        row[column] = 'unknown' if column in categorical else 0.0

    return row, lang


def _category_vocab(bundle: Dict) -> Dict[str, set]:
    """从已训练的一热编码器取各类别白名单，用于复现 top-N 归并为 other。"""
    onehot = bundle['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    return {
        name: set(values)
        for name, values in zip(bundle['categorical_columns'], onehot.categories_)
    }


def _risk_band(score: float) -> str:
    """与 ml/train_models.py 的 pandas.cut 分箱保持一致。"""
    if score <= 40:
        return 'low'
    if score <= 60:
        return 'medium'
    if score <= 80:
        return 'high'
    return 'critical'


class Command(BaseCommand):
    help = '风险预测推理命令（只读）：python manage.py risk predict --code-id <id>'

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest='action', required=True)
        predict = sub.add_parser('predict', help='对单个语言 code_id 做风险推理')
        predict.add_argument('--code-id', dest='code_id', type=int, required=True,
                             help='语言主键（Language.language_id / 风险数据 code_id）')

    def handle(self, *args, **options):
        if options['action'] != 'predict':
            raise CommandError(f'未知 action: {options["action"]}')

        code_id = options['code_id']
        if not Language.objects.filter(language_id=code_id).exists():
            raise CommandError(
                f'未找到 code_id={code_id} 的语言记录，请确认 ID 是否正确。')

        try:
            bundle = risk_service.get_model_bundle()
        except ImportError as exc:
            raise CommandError(
                '缺少推理依赖，无法加载模型：%s\n'
                '模型需要 lightgbm / scikit-learn / joblib / pandas / numpy，'
                '请安装 requirements-ml.txt 后重试。' % exc
            )
        if bundle is None:
            raise CommandError(
                f'未找到模型文件：{risk_service.MODEL_FILE}\n'
                '请先执行 python manage.py etl ml 生成训练产物。')

        feature_columns = list(bundle['feature_columns'])
        categorical_columns = list(bundle['categorical_columns'])
        row, lang = _build_feature_row(code_id, feature_columns, categorical_columns)

        vocab = _category_vocab(bundle)
        for column in ('family_top', 'country'):
            if row.get(column) not in vocab.get(column, set()):
                row[column] = 'other'

        try:
            result = risk_service.predict_features(row)
        except ImportError as exc:
            raise CommandError(f'推理依赖缺失：{exc}')
        if result is None:
            raise CommandError('模型未能完成推理（模型文件可能已失效）。')

        meta = risk_service.get_service_meta()
        labels = meta.get('level_labels_zh', {})
        name = display_language_name({
            'language_id': code_id,
            'language_name': lang.get('language_name'),
        })

        level = result['predicted_level']
        score = result['risk_score']
        band = _risk_band(score)

        self.stdout.write(f'code_id: {code_id}')
        self.stdout.write(f'language_name: {name}')
        self.stdout.write(f'model: {bundle.get("model_name", "")} '
                          f'(trained_at={bundle.get("trained_at", "")})')
        self.stdout.write(f'predicted_level: {level} ({labels.get(level, "")})')
        self.stdout.write(f'risk_score: {score:.1f}')
        self.stdout.write(f'risk_band: {band} '
                          f'({risk_service.RISK_BAND_LABELS.get(band, band)})')
        self.stdout.write(f'confidence: {result["confidence"]:.3f}')
        self.stdout.write('probabilities:')
        for class_name in bundle['class_names']:
            probability = result['probabilities'].get(class_name, 0.0)
            self.stdout.write(
                f'  {class_name:<20} {probability:.4f} ({labels.get(class_name, "")})')
        self.stdout.write('（只读推理，未写入数据库）')
