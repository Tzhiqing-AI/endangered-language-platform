"""ETL 配置：数据源、归属表、关键字段与人工名称清单。

该模块只依赖 settings 与标准库，供 management command 与各阶段复用。
"""
from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings

BASE_DIR = Path(settings.BASE_DIR)
DATA_DIR = BASE_DIR / 'data'
NAMES_FILE = DATA_DIR / 'language_names.json'
EXCLUDED_FILE = DATA_DIR / 'excluded_languages.json'
BACKUP_DIR = BASE_DIR / 'backups' / 'etl'

CSV_DEPRECATION_REASON = (
    'csv_output 目录已删除、CSV 导入路径已废弃：该历史导出与 elcat.sql 逐表比对确认存在系统性数据丢失——'
    '多 INSERT 表只保留了最后一条 INSERT（如 auth_user 1437/17813、'
    'django_session 5765/6653416），文本表因未转义换行/引号被合并或截断'
    '（language_context 65/14704、language_codes 1027/3669），且出现 '
    '57,936 列的畸形记录。请改用 elcat_source 数据库重建。'
)

# 源库表 -> 目标模型表；仅用于 status/verify 对照
SOURCE_TARGET_PAIRS = [
    ('language_language', 'language'),
    ('language_codes', 'language_code'),
    ('language_sources', 'language_source'),
    ('language_bibliography', 'bibliography_link'),
    ('language_vitality', 'language_vitality'),
    ('language_speakers', 'speaker_data'),
    ('language_locations', 'language_location_detail'),
    ('language_context', 'language_context'),
    ('language_orthography', 'language_orthography'),
    ('language_resources', 'language_resource'),
    ('language_languagemedia', 'language_media'),
    ('language_changesuggestion', 'change_suggestion'),
    ('language_notification', 'notification'),
    ('language_notification_users', 'notification_user'),
    ('language_documentation', 'language_documentation'),
    ('language_activity', 'language_activity'),
    ('language_sample', 'language_sample'),
    ('language_samplecategory', 'sample_category_link'),
    ('language_samplecontributor', 'sample_contributor_link'),
    ('language_sampletranslation', 'sample_translation'),
    ('categories_category', 'sample_category'),
    ('language_country', 'country'),
    ('language_region', 'region'),
    ('language_country_region', 'country_region'),
    ('language_language_country', 'language_country_map'),
    ('language_transmission', 'transmission'),
    ('language_speakernumbertrends', 'speaker_number_trend'),
    ('language_domainsofuse', 'domain_of_use'),
    ('language_sourcetypes', 'source_type'),
]

# 允许的已知差异（源库自身脏数据或测试记录），verify 不视为失败
KNOWN_ALLOWED_DIFF = {
    'language': 21,          # 21 条源记录缺少 code_val_cache/simplified_level，历史导入未落库
    'language_vitality': 1,  # code_id 10875 测试记录重建时跳过
    'speaker_data': 1,
    'language_location_detail': 1,
    'language_context': 1,
    'language_orthography': 1,
    'sample_category_link': 2,  # 源库自身悬空引用 sample 8327
    'language_activity': 479,   # 源库 code_id 为 NULL 的动态记录，历史导入跳过
}

# status 命令展示的关键字段（模型表名 -> 字段）
STATUS_FIELDS = {
    'language': [
        'language_name', 'legacy_endangerment_level', 'level_of_endangerment',
        'number_of_speakers', 'country', 'language_family',
        'intergenerational_transmission', 'speaker_trends', 'name_source',
    ],
    'language_code': ['iso_code', 'language_name', 'alternate_names', 'classification'],
    'language_source': ['title', 'author', 'year', 'isbn'],
    'bibliography_link': ['source_id', 'code_id'],
    'language_vitality': ['endangerment_level', 'transmission_id', 'preferred'],
    'speaker_data': ['speaker_number', 'speaker_number_text', 'preferred'],
    'language_location_detail': ['places', 'coordinates', 'preferred'],
    'language_context': ['government_support', 'institutional_support', 'speaker_attitude'],
    'language_orthography': ['scripts'],
    'language_sample': ['title', 'url', 'sample_type'],
    'language_activity': ['date', 'code_id', 'obj_type'],
}

# 重建时只允许 ETL 清理/裁剪的镜像表（不含运行期用户数据）
MIRROR_TABLES = {
    'region', 'transmission', 'speaker_number_trend', 'domain_of_use', 'source_type',
    'score_level', 'language_code', 'country', 'country_region', 'language_country_map',
    'language_source', 'bibliography_link', 'language_vitality', 'speaker_data',
    'language_location_detail', 'language_context', 'language_orthography',
    'language_resource', 'language_media', 'language_documentation',
    'sample_category', 'language_sample', 'sample_category_link',
    'sample_contributor_link', 'sample_translation',
}

# 运行期用户数据：只 upsert，不裁剪
RUNTIME_TABLES = {
    'change_suggestion', 'notification', 'notification_user', 'language_activity',
}


def load_names_config() -> dict:
    """读取人工核实名称/测试记录清单（JSON 以避免新增 PyYAML 依赖）。"""
    if not NAMES_FILE.exists():
        return {'version': 1, 'names': {}, 'unresolved': {}, 'test_records': {}}
    with NAMES_FILE.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def name_overrides() -> dict:
    cfg = load_names_config()
    return {int(k): (v['name'], v.get('source', '')) for k, v in cfg.get('names', {}).items()}


def unresolved_ids() -> dict:
    cfg = load_names_config()
    return {int(k): v for k, v in cfg.get('unresolved', {}).items()}


def test_records() -> dict:
    cfg = load_names_config()
    return {int(k): v for k, v in cfg.get('test_records', {}).items()}


def load_excluded_config() -> dict:
    """读取抑制名单：源库存在但不得写入目标库的 language_id。"""
    if not EXCLUDED_FILE.exists():
        return {'version': 1, 'excluded': {}}
    with EXCLUDED_FILE.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def excluded_ids() -> dict:
    cfg = load_excluded_config()
    return {int(k): v for k, v in cfg.get('excluded', {}).items()}


def hard_excluded_ids() -> dict:
    """从未导入目标库、源行应整体扣除的抑制条目（不含合并别名）。"""
    return {k: v for k, v in excluded_ids().items() if not v.get('merged_into')}


def merged_dropped_rows() -> dict:
    """合并时因唯一键冲突被舍弃的行数：目标表 -> 行数。"""
    totals = {}
    for entry in excluded_ids().values():
        for table, count in (entry.get('dropped_rows') or {}).items():
            totals[table] = totals.get(table, 0) + int(count)
    return totals


# 抑制名单在源库中的引用位置：目标表 -> (源表, 关联列)
# 用于 verify 扣除「本就不该导入」的源行，使行数对照仍然有意义。
SUPPRESSED_SOURCE_REFS = {
    'language': ('language_language', 'code_id'),
    'language_vitality': ('language_vitality', 'code_id'),
    'speaker_data': ('language_speakers', 'code_id'),
    'language_location_detail': ('language_locations', 'code_id'),
    'language_context': ('language_context', 'code_id'),
    'language_orthography': ('language_orthography', 'code_id'),
    'language_resource': ('language_resources', 'code_id'),
    'language_media': ('language_languagemedia', 'code_id'),
    'language_documentation': ('language_documentation', 'code_id'),
    'bibliography_link': ('language_bibliography', 'code_id'),
    'language_activity': ('language_activity', 'code_id'),
    'language_sample': ('language_sample', 'code_id'),
}

# 样本附属表：按被抑制语言的 sample_id 扣除
SUPPRESSED_SAMPLE_LINK_SOURCES = {
    'sample_category_link': 'language_samplecategory',
    'sample_contributor_link': 'language_samplecontributor',
    'sample_translation': 'language_sampletranslation',
}
