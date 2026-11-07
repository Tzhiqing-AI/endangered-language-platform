"""
Django 测试配置

用于 pytest 和 CI/CD 环境
"""
from .settings import *

# 测试数据库配置（使用 SQLite 内存数据库）
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# 简化密码哈希（加速测试）
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]

# 测试环境不依赖 Redis，避免 CI 因缺少外部服务而失败
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    }
}
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

ALLOWED_HOSTS = ['testserver', 'localhost', '127.0.0.1']

# 日志级别降低
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
}

# 关闭调试模式
DEBUG = False

# 测试用的 OpenAI API Key
OPENAI_API_KEY = 'test-api-key-for-ci'

# 测试环境默认关闭联网搜索，避免消耗真实额度；需要测试时用 override_settings 打开并 mock
SEARCH_ENABLED = False
SEARCH_API_KEY = ''
SEARCH_PROVIDER = 'tavily'
