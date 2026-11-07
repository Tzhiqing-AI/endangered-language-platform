from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Dashboard'

    def ready(self):
        # 注册语言数据写入后的缓存失效信号
        from Dashboard import signals  # noqa: F401
