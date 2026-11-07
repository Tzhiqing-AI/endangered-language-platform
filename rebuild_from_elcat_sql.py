"""已收敛到 ``python manage.py etl rebuild --source elcat --confirm``。

本文件只是兼容入口，不再包含独立实现；重建逻辑见
``Dashboard/etl/steps.py`` 的 ``rebuild_elcat``。
"""
import os
import sys

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')
django.setup()

from django.core.management import call_command
from django.core.management.base import CommandError


def main():
    if '--confirm' not in sys.argv:
        print('拒绝执行：重建会修改 language_v2 的多张表。')
        print('请设置 ALLOW_DESTRUCTIVE_DB_REBUILD=1 并追加 --confirm；'
              '仅预览请执行 python manage.py etl rebuild --dry-run。')
        return 2
    try:
        call_command('etl', 'rebuild', '--source', 'elcat', '--confirm')
    except CommandError as exc:
        print(f'执行失败：{exc}')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
