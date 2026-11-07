"""
Dashboard 应用通用模块

包含常量配置、工具函数、AI客户端等共享资源
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

from django.db.models import Count

from openai import OpenAI

from Dashboard.constants import (
    CHART_DATA_CACHE_KEY, ELCAT_SIMPLE_LEVEL_KEYS, ELCAT_TO_LEGACY,
    ENDANGERMENT_LEVELS, KNOWN_LEGACY_LEVEL_KEYS, LEGACY_CHART_CACHE_KEYS,
    MAP_SNAPSHOT_CACHE_KEY, UNESCO_LEVEL_KEYS,
)

# 配置日志
logger = logging.getLogger(__name__)

# =============================================================================
# 常量配置
# =============================================================================

# API 配置（从 settings 读取）
from django.conf import settings
OPENAI_API_KEY = getattr(settings, 'OPENAI_API_KEY', '')
OPENAI_BASE_URL = getattr(settings, 'OPENAI_BASE_URL', 'https://api.moonshot.cn/v1')
OPENAI_MODEL = getattr(settings, 'OPENAI_MODEL', 'kimi-k2.7-code')
OPENAI_TEMPERATURE = getattr(settings, 'OPENAI_TEMPERATURE', 1.0)

# 分页配置
PAGINATION_PAGE_SIZE = 20
PAGINATION_ORPHANS = 3

# CSV 导入配置
CSV_BATCH_SIZE = 100
CSV_MAX_FILE_SIZE_MB = 10
CSV_ALLOWED_EXTENSIONS = ['.csv', '.txt']
CSV_ALLOWED_MIME_TYPES = ['text/csv', 'text/plain', 'application/csv', 'application/vnd.ms-excel']

# 濒危程度映射、主流语种排除列表等共享常量见 Dashboard.constants

# 搜索配置
MAX_SEARCH_LENGTH = 100

# Chat AI 配置
CHAT_HISTORY_LIMIT = 50  # 聊天历史记录保留条数
CHAT_RECENT_LIMIT = 10   # 最近对话轮次

# =============================================================================
# OpenAI 客户端配置
# =============================================================================

def get_openai_client() -> OpenAI:
    """
    获取 OpenAI 客户端实例（带 30 秒超时）

    Returns:
        OpenAI: 配置好的 OpenAI 客户端
    """
    import httpx
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=httpx.Timeout(30.0, connect=10.0),
    )


# 全局客户端（模块级别单例）
client = get_openai_client()


# =============================================================================
# 工具函数
# =============================================================================

def get_user_context(request) -> Tuple[Optional[Any], str, int]:
    """
    从 Session 获取用户信息、名称和管理员标识

    Args:
        request: HTTP 请求对象

    Returns:
        Tuple[Optional[Any], str, int]: (用户对象, 用户名, 是否管理员)，未登录返回 (None, "", 0)
    """
    from language.utils import get_authenticated_user
    user = get_authenticated_user(request)
    if user:
        name = user.name if user.name else user.username
        admin = getattr(user, 'admin', 0)
        return user, name, admin
    return None, "", 0


def get_chart_data() -> Dict[str, int]:
    """
    统一获取濒危语言统计数据，用于可视化图表（缓存 5 分钟）

    Returns:
        Dict[str, int]: 各濒危程度级别的语言数量统计
    """
    from django.core.cache import cache

    cached = cache.get(CHART_DATA_CACHE_KEY)
    # 兼容旧缓存结构：字段不全时视为未命中，避免页面读到过期口径
    if isinstance(cached, dict) and 'legacy_awakening' in cached and 'unknown' in cached:
        return cached

    from Dashboard.models import Language

    chart_data = {key: 0 for key in ENDANGERMENT_LEVELS}
    chart_data.update({'legacy_awakening': 0, 'legacy_safe': 0})

    # 主分类：UNESCO 五级（由 language_vitality 原始分级映射）
    legacy_counts = Language.objects.values(
        'legacy_endangerment_level'
    ).annotate(
        count=Count('language_id')
    )
    for item in legacy_counts:
        level = item['legacy_endangerment_level']
        count = item['count']
        if level == 'awakening':
            chart_data['legacy_awakening'] += count
        elif level == 'safe':
            chart_data['legacy_safe'] += count
        elif level in UNESCO_LEVEL_KEYS:
            chart_data[level] += count
        else:
            chart_data['unknown'] += count

    # 兼容 ELCat 简化分级（洞察页使用）
    raw_counts = Language.objects.values(
        'level_of_endangerment'
    ).annotate(
        count=Count('language_id')
    )
    for item in raw_counts:
        level = item['level_of_endangerment']
        if level in ELCAT_SIMPLE_LEVEL_KEYS or level == 'awakening':
            chart_data[level] = item['count']

    cache.set(CHART_DATA_CACHE_KEY, chart_data, 300)  # 缓存 5 分钟
    return chart_data


def invalidate_language_stats_cache() -> None:
    """清除语言统计缓存。

    在语言数据写入后调用（视图显式调用 + Language 模型的 post_save/post_delete 信号），
    保证 ETL/管理页/后台写入后缓存不会滞留 5 分钟。
    地图快照同样依赖语言与位置数据，一并失效。
    """
    from django.core.cache import cache

    for key in LEGACY_CHART_CACHE_KEYS:
        cache.delete(key)
    cache.delete(MAP_SNAPSHOT_CACHE_KEY)


UNNAMED_LANGUAGE_LABEL = '未命名语言'
INTERNAL_NAME_SOURCE_PREFIX = 'manual:unresolved'


def _pick(obj, key):
    """同时兼容模型实例与字典的取值方式。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def display_language_name(obj) -> str:
    """语言名称展示的唯一出口：空名回退为带 ID 的占位，避免空白标题。

    占位携带 language_id（或风险数据里的 code_id），保证多条空名记录可区分。
    """
    name = str(_pick(obj, 'language_name') or '').strip()
    if name:
        return name
    language_id = _pick(obj, 'language_id')
    if language_id in (None, ''):
        language_id = _pick(obj, 'code_id')
    if language_id in (None, ''):
        return UNNAMED_LANGUAGE_LABEL
    return f'{UNNAMED_LANGUAGE_LABEL} #{language_id}'


def display_name_source(value) -> str:
    """名称来源的对外表述：内部标记不直接暴露给普通用户。"""
    source = str(value or '').strip()
    if not source:
        return ''
    if source.startswith(INTERNAL_NAME_SOURCE_PREFIX):
        return '待核实'
    return source


def split_country_parts(value) -> List[str]:
    """拆分逗号分隔的多国字符串（空格与空段忽略）。"""
    if not value:
        return []
    return [part.strip() for part in str(value).split(',') if part.strip()]


def split_family_parts(value) -> List[str]:
    """取语系分类路径的第一层，避免把完整路径当成语系。"""
    if not value:
        return []
    top = str(value).split(';')[0].strip()
    return [top] if top else []


def count_distinct_families() -> int:
    """按第一层语系统计数量（数据库去重后再拆分，避免回传全部行）。"""
    from Dashboard.models import Language

    rows = Language.objects.exclude(language_family__isnull=True).exclude(
        language_family=''
    ).values_list('language_family', flat=True).distinct()
    names = set()
    for family in rows:
        names.update(split_family_parts(family))
    return len(names)


def count_distinct_countries() -> int:
    """按逗号拆分统计国家/地区数量（数据库去重后再拆分）。"""
    from Dashboard.models import Language

    rows = Language.objects.exclude(country__isnull=True).exclude(
        country=''
    ).values_list('country', flat=True).distinct()
    names = set()
    for country in rows:
        names.update(split_country_parts(country))
    return len(names)


def build_country_counter(queryset=None):
    """按语言行统计每个国家的语言数（保留重复行的计数口径）。

    Returns:
        collections.Counter: 国家名 → 语言记录数
    """
    from collections import Counter
    from Dashboard.models import Language

    qs = (queryset if queryset is not None else Language.objects).filter(
        country__isnull=False
    ).exclude(country='')
    counter = Counter()
    for country_text in qs.values_list('country', flat=True):
        for part in split_country_parts(country_text):
            counter[part] += 1
    return counter


def pagination_window(current: int, total: int, head: int = 5, tail: int = 2) -> List[Any]:
    """生成“前几页 + 省略号 + 最后两页”的页码序列。"""
    if total <= head + tail + 3:
        return list(range(1, total + 1))

    pages = list(range(1, head + 1))
    start = max(head + 1, current - 1)
    end = min(total - tail, current + 1)

    if start > head + 1:
        pages.append('...')
    if start <= end:
        pages.extend(range(start, end + 1))
    if end < total - tail:
        pages.append('...')

    pages.extend(range(total - tail + 1, total + 1))
    return pages


def get_paginated_languages(queryset, page_number: int = 1, per_page: int = PAGINATION_PAGE_SIZE) -> Tuple[Any, Any]:
    """
    获取分页后的语言数据

    Args:
        queryset: 语言数据查询集
        page_number: 当前页码，默认为 1
        per_page: 每页显示数量

    Returns:
        Tuple[Any, Any]: (分页对象, 当前页数据)
    """
    from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

    paginator = Paginator(
        queryset,
        per_page,
        orphans=PAGINATION_ORPHANS,
        allow_empty_first_page=True
    )

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    return paginator, page_obj


def validate_csv_file(file_obj) -> Tuple[bool, str]:
    """
    验证上传的 CSV 文件

    Args:
        file_obj: 上传的文件对象

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    if not file_obj:
        return False, "未选择文件"

    file_size_mb = file_obj.size / (1024 * 1024)
    if file_size_mb > CSV_MAX_FILE_SIZE_MB:
        return False, f"文件大小超过限制（最大 {CSV_MAX_FILE_SIZE_MB}MB）"

    filename = file_obj.name.lower()
    if not any(filename.endswith(ext) for ext in CSV_ALLOWED_EXTENSIONS):
        return False, f"不支持的文件格式，仅支持: {', '.join(CSV_ALLOWED_EXTENSIONS)}"

    file_mime_type = getattr(file_obj, 'content_type', '')
    if file_mime_type:
        allowed_mime_patterns = ['csv', 'text/', 'excel', 'spreadsheet']
        is_allowed = any(pattern in file_mime_type.lower() for pattern in allowed_mime_patterns)
        if not is_allowed:
            return False, f"文件类型不合法（MIME: {file_mime_type}）"

    try:
        file_obj.seek(0)
        first_bytes = file_obj.read(3)
        file_obj.seek(0)
        if first_bytes:
            if first_bytes.startswith(b'\x89PNG') or first_bytes.startswith(b'\xff\xd8\xff'):
                return False, "文件内容不是有效的 CSV 格式"
    except Exception:
        pass

    return True, ""


def parse_csv_row(row: List[str], line_number: int) -> Tuple[bool, Optional[Dict], str]:
    """
    解析单行 CSV 数据

    Args:
        row: CSV 行数据列表
        line_number: 行号

    Returns:
        Tuple[bool, Optional[Dict], str]: (是否成功, 解析后的数据字典, 错误信息)
    """
    if len(row) < 8:
        return False, None, f"第 {line_number} 行: 列数不足（需要 8 列，实际 {len(row)} 列）"

    cleaned_row = [cell.strip() for cell in row]

    if not cleaned_row[0]:
        return False, None, f"第 {line_number} 行: 语言名称不能为空"

    number_of_speakers = 0
    if cleaned_row[2]:
        try:
            number_of_speakers = int(cleaned_row[2])
            if number_of_speakers < 0:
                return False, None, f"第 {line_number} 行: 使用者数量不能为负数"
        except ValueError:
            return False, None, f"第 {line_number} 行: 使用者数量格式错误（应为整数）"

    level = cleaned_row[1] or 'unsafe'
    legacy_level = level if level in KNOWN_LEGACY_LEVEL_KEYS else ELCAT_TO_LEGACY.get(level, '')

    data = {
        'language_name': cleaned_row[0],
        'level_of_endangerment': level,
        'legacy_endangerment_level': legacy_level,
        'number_of_speakers': number_of_speakers,
        'speaker_trends': cleaned_row[3] or 'Stable',
        'intergenerational_transmission': cleaned_row[4] or 'some',
        'language_family': cleaned_row[5] or '',
        'country': cleaned_row[6] or '',
        'additional_notes': cleaned_row[7] or ''
    }

    return True, data, ""


def check_duplicate_language(language_name: str, exclude_id: Optional[int] = None) -> bool:
    """
    检查语言名称是否已存在

    Args:
        language_name: 语言名称
        exclude_id: 排除的 ID（用于更新时排除自身）

    Returns:
        bool: True 表示存在重复
    """
    from Dashboard.models import Language
    if not language_name:
        return False
    queryset = Language.objects.filter(language_name__iexact=language_name)
    if exclude_id:
        queryset = queryset.exclude(language_id=exclude_id)
    return queryset.exists()
