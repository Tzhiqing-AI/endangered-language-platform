"""已收敛到 ``python manage.py etl enrich --only names``。

人工名称清单位于 ``data/language_names.json``；本文件只是兼容入口。
"""
import os
import sys

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')
django.setup()

from django.core.management import call_command


def main():
    call_command('etl', 'enrich', '--only', 'names')
    return 0


if __name__ == '__main__':
    sys.exit(main())
