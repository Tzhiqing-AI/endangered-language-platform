"""
Dashboard 应用视图模块

此文件仅用于向后兼容，所有视图已拆分到以下模块：
- language_views: 语言列表、搜索功能
- ai_views: AI 翻译、可视化、聊天功能
- nlp_views: NLP 自然语言处理功能

新视图请添加到对应的专用模块中。
"""

# 语言管理视图
from Dashboard.language_views import (
    language,
    language_list,
    language_search,
    language_detail,
    user_management,
    user_management_post,
    admin_dashboard,
)

# AI 功能视图
from Dashboard.ai_views import (
    language_translation,
    language_visualization,
    language_map,
    language_chat,
    get_chat_history,
    translation_memory_list,
    translation_rate,
    glossary_manage,
)

# NLP 功能视图
from Dashboard.nlp_views import (
    nlp_page,
    nlp_process,
)

__all__ = [
    'language',
    'language_list',
    'language_search',
    'language_detail',
    'language_translation',
    'language_visualization',
    'language_map',
    'language_chat',
    'nlp_page',
    'nlp_process',
    'user_management',
    'user_management_post',
    'get_chat_history',
    'translation_memory_list',
    'translation_rate',
    'glossary_manage',
    'admin_dashboard',
]
