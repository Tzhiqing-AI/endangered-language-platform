"""
Dashboard 跨模块共享常量

集中定义等级映射、主流语种排除列表与分级键集合，避免在多个视图模块中重复。

注意：本模块不导入任何模型，可被 models.py 与 common.py 安全引用。
"""

# =============================================================================
# 等级标签映射（唯一来源）
# =============================================================================

# UNESCO 五级分类（本系统原有分类）
LEGACY_LEVEL_LABELS = {
    'unsafe': '不安全',
    'danger': '危险',
    'endangered': '濒危',
    'seriously_endangered': '垂危',
    'extinct': '灭绝',
    'awakening': '复苏',
    'safe': '安全',
}

# ELCat 原始分级
ELCAT_LEVEL_LABELS = {
    'high': '高危',
    'mid': '中危',
    'low': '低危',
    'dormant': '休眠',
    'awakening': '复苏',
}

# 合并视图：UNESCO 五级 + ELCat 简化分级（页面下拉框与详情标签使用）
ENDANGERMENT_LEVELS = {
    **LEGACY_LEVEL_LABELS,
    **ELCAT_LEVEL_LABELS,
    'unknown': '未分类',
}

# ELCat 简化分级 → 本系统 UNESCO 五级（导入/兜底用）
ELCAT_TO_LEGACY = {
    'high': 'seriously_endangered',
    'mid': 'endangered',
    'low': 'danger',
    'dormant': 'extinct',
    'awakening': 'awakening',
}

# =============================================================================
# 分级键集合（统计口径）
# =============================================================================

# UNESCO 五级中计入“濒危”序列的键（不含复苏/安全）
UNESCO_LEVEL_KEYS = (
    'unsafe', 'danger', 'endangered', 'seriously_endangered', 'extinct',
)

# 所有已知的 legacy 分级键（用于计算“未分类”）
KNOWN_LEGACY_LEVEL_KEYS = UNESCO_LEVEL_KEYS + ('awakening', 'safe')

# ELCat 源库简化分级键
ELCAT_SIMPLE_LEVEL_KEYS = ('high', 'mid', 'low', 'dormant')

# =============================================================================
# 主流大语种排除列表
# =============================================================================
# 首页、洞察页、管理页统一使用，避免同一份名单在多个视图中漂移
EXCLUDED_MAJOR_LANGUAGES = (
    'English', '英语', 'Mandarin', '普通话', 'Hindi', '印地语',
    'Spanish', '西班牙语', 'French', '法语', 'Arabic', '阿拉伯语',
    'Bengali', '孟加拉语', 'Russian', '俄语', 'Portuguese', '葡萄牙语',
    'Indonesian', '印度尼西亚语',
)

# =============================================================================
# 缓存键
# =============================================================================
CHART_DATA_CACHE_KEY = 'chart_data_v2'
# 历史版本键：写入后一并清理，避免旧页面读到过期数据
# 选择 chart_data_v2 作为主键，可直接复用现有 ETL 脚本里已有的 cache.delete('chart_data_v2')
LEGACY_CHART_CACHE_KEYS = ('chart_data_v2', 'chart_data_v3', 'chart_data')

# 地图全量快照：页面渲染与按视口/分级裁剪都从这里取数，避免每次请求重跑三条全表扫描
MAP_SNAPSHOT_CACHE_KEY = 'map_snapshot_v1'
MAP_SNAPSHOT_TTL = 300  # 秒，与图表统计缓存保持一致
