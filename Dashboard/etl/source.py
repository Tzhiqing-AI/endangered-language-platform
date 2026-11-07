"""elcat_source 只读访问与 ETL 字段清洗函数。"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, time

from django.utils import timezone


def s(value, max_length=None):
    if value is None:
        return ''
    text = str(value).strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]
    return text


INT_MAX = 2_147_483_647


def i(value, default=None):
    if value is None:
        return default
    text = str(value).strip().replace(',', '')
    match = re.search(r'-?\d+', text)
    if not match:
        return default
    number = int(match.group())
    lower = text.lower()
    if 'million' in lower:
        number *= 1_000_000
    elif 'billion' in lower:
        number *= 1_000_000_000
    # 目标库整型列为 int（32 位）：长句描述里的数字会被放大到天文数字，
    # 直接写入会触发 1264 Out of range，超范围一律视为无法解析。
    if abs(number) > INT_MAX:
        return default
    return number


def f(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bo(value):
    return str(value or '').strip().lower() in ('1', 'true', 't', 'yes')


def aware_dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if timezone.is_naive(value):
            return timezone.make_aware(value, timezone.get_current_timezone())
        return value
    if isinstance(value, date):
        return timezone.make_aware(
            datetime.combine(value, time.min), timezone.get_current_timezone())
    return None


def as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def parse_coordinates(value):
    text = s(value)
    if not text:
        return None
    first = text.split(';')[0].strip()
    parts = [p.strip() for p in first.split(',')]
    if len(parts) < 2:
        return None
    lat, lng = parts[0], parts[1]
    try:
        float(lat)
        float(lng)
    except ValueError:
        return None
    return lat, lng


class ElcatSource:
    """通过 pymysql 直连 elcat_source，避免污染 Django DATABASES。"""

    def __init__(self):
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover - 环境缺失时给出明确提示
            raise RuntimeError(
                '缺少 pymysql。ETL 需要安装 ETL 依赖（requirements-etl.txt）。') from exc

        self._pymysql = pymysql
        self.database = os.getenv('ELCAT_DB_NAME', 'elcat_source')
        self._conn = pymysql.connect(
            host=os.getenv('ELCAT_DB_HOST', 'localhost'),
            user=os.getenv('ELCAT_DB_USER', 'root'),
            password=os.getenv('ELCAT_DB_PASSWORD', os.getenv('DB_PASSWORD', '')),
            database=self.database,
            port=int(os.getenv('ELCAT_DB_PORT', 3306)),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
        )

    def rows(self, table):
        with self._conn.cursor() as cursor:
            cursor.execute(f'SELECT * FROM `{table}`')
            return cursor.fetchall()

    def count(self, table):
        with self._conn.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) AS n FROM `{table}`')
            return int(cursor.fetchone()['n'])

    def count_where(self, table, column, values):
        values = [v for v in values if v is not None]
        if not values:
            return 0
        placeholders = ','.join(['%s'] * len(values))
        with self._conn.cursor() as cursor:
            cursor.execute(
                f'SELECT COUNT(*) AS n FROM `{table}` WHERE `{column}` IN ({placeholders})',
                values)
            return int(cursor.fetchone()['n'])

    def ids_where(self, table, column, values, select='id'):
        values = [v for v in values if v is not None]
        if not values:
            return []
        placeholders = ','.join(['%s'] * len(values))
        with self._conn.cursor() as cursor:
            cursor.execute(
                f'SELECT `{select}` FROM `{table}` WHERE `{column}` IN ({placeholders})',
                values)
            return [row[select] for row in cursor.fetchall()]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
