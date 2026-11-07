"""Dashboard 模型信号

语言数据发生写入/删除后立即清除统计缓存，避免首页与洞察页最多 5 分钟的旧数据。
地图快照同样由语言 + 位置明细聚合而来，三张表变动都要失效，否则裁剪结果会滞后。
"""

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from Dashboard.models import Language, LanguageLocation, LanguageLocationDetail


@receiver(post_save, sender=Language)
@receiver(post_delete, sender=Language)
@receiver(post_save, sender=LanguageLocation)
@receiver(post_delete, sender=LanguageLocation)
@receiver(post_save, sender=LanguageLocationDetail)
@receiver(post_delete, sender=LanguageLocationDetail)
def _invalidate_chart_cache(sender, **kwargs):
    # 延迟导入，避免应用加载阶段初始化 AI 客户端
    from Dashboard.common import invalidate_language_stats_cache

    invalidate_language_stats_cache()
