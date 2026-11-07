"""已收敛到 ``python manage.py etl enrich --only backfill``。

回填逻辑见 ``Dashboard/etl/steps.py`` 的 ``backfill_language_fields``；
legacy 等级只填空值，不再覆盖人工核实结果。
"""
import os
import sys

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'language.settings')
django.setup()

from django.core.management import call_command


def main():
    call_command('etl', 'enrich', '--only', 'backfill')
    return 0


if __name__ == '__main__':
    sys.exit(main())
