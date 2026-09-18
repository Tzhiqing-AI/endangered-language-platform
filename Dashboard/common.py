"""
Dashboard 应用通用模块

包含常量配置、工具函数、AI客户端等共享资源
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

from django.db.models import Count

from openai import OpenAI

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

# 濒危程度映射 - UNESCO 5级分类
ENDANGERMENT_LEVELS = {
    'unsafe': '不安全',
    'danger': '危险',
    'endangered': '濒危',
    'seriously_endangered': '垂危',
    'extinct': '灭绝'
}

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

    cached = cache.get('chart_data')
    if cached is not None:
        return cached

    from Dashboard.models import Language
    language_counts = Language.objects.values(
        'level_of_endangerment'
    ).annotate(
        count=Count('level_of_endangerment')
    )

    # 初始化默认值 - UNESCO 5级分类
    chart_data = {
        'unsafe': 0,
        'danger': 0,
        'endangered': 0,
        'seriously_endangered': 0,
        'extinct': 0
    }

    # 填充实际数据
    for item in language_counts:
        level = item['level_of_endangerment']
        if level in chart_data:
            chart_data[level] = item['count']

    cache.set('chart_data', chart_data, 300)  # 缓存 5 分钟
    return chart_data


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

    data = {
        'language_name': cleaned_row[0],
        'level_of_endangerment': cleaned_row[1] or 'unsafe',
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
    queryset = Language.objects.filter(language_name__iexact=language_name)
    if exclude_id:
        queryset = queryset.exclude(language_id=exclude_id)
    return queryset.exists()
