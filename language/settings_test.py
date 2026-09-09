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
