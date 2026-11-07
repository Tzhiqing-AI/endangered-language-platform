"""幂等写入：按主键 upsert，并只裁剪 ETL 台账内、源库已不存在的行。"""
from __future__ import annotations

from typing import Dict, Iterable

from django.db import transaction

from Dashboard.models import ETLManagedRow


def _same(left, right):
    if left is None and right is None:
        return True
    if left in ('', None) and right in ('', None):
        return True
    return left == right


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def upsert_rows(model, incoming: Dict[int, dict], fields: Iterable[str], *,
                scope='elcat', run_id=None, prune=False, dry_run=False,
                batch_size=2000) -> dict:
    """按主键写入 ``incoming``。

    - 新行 bulk_create，已有行只更新发生变化的字段；
    - ``prune=True`` 时，仅删除 ETL 台账内且源库已不存在的行；
    - ``dry_run=True`` 时只统计，不写库、不写台账。
    """
    table = model._meta.db_table
    pk_name = model._meta.pk.name
    fields = list(fields)
    stats = {'table': table, 'created': 0, 'updated': 0, 'unchanged': 0,
             'deleted': 0, 'source_rows': len(incoming)}

    if not incoming:
        if prune:
            owned = set(ETLManagedRow.objects.filter(
                scope=scope, table_name=table).values_list('row_id', flat=True))
            stats['deleted'] = len(owned) if dry_run else _delete_ids(model, owned)
            if not dry_run:
                ETLManagedRow.objects.filter(scope=scope, table_name=table).delete()
        return stats

    keys = list(incoming)
    existing = {
        row[pk_name]: row
        for row in model.objects.filter(pk__in=keys).values(pk_name, *fields)
    }

    creates, updates = [], []
    for key in keys:
        payload = incoming[key]
        current = existing.get(key)
        if current is None:
            creates.append(model(**payload))
            continue
        changed = {name: payload[name] for name in fields
                   if not _same(current.get(name), payload[name])}
        if changed:
            # bulk_update 按 update_fields 读取整行属性：只赋值变更字段时，
            # 未赋值的非空整型列会退化成 None，MySQL 会报 1048 cannot be null。
            updates.append(model(**{pk_name: key, **payload}))
        else:
            stats['unchanged'] += 1

    stats['created'] = len(creates)
    stats['updated'] = len(updates)

    if dry_run:
        if prune:
            owned = set(ETLManagedRow.objects.filter(
                scope=scope, table_name=table).values_list('row_id', flat=True))
            if not owned:
                owned = set(model.objects.values_list(pk_name, flat=True))
            stats['deleted'] = len(owned - set(keys))
        return stats

    with transaction.atomic():
        if prune:
            owned = set(ETLManagedRow.objects.filter(
                scope=scope, table_name=table).values_list('row_id', flat=True))
            if not owned:
                # 首次运行：把目标表现存行视为 ETL 自有，之后只按台账裁剪
                owned = set(model.objects.values_list(pk_name, flat=True))
            stale = owned - set(keys)
            stats['deleted'] = _delete_ids(model, stale)
            if stale:
                # 被 prune 的行必须同步移出台账，否则残留归属会让下一次
                # prune 误删手工重建的同主键行；目标行早已不存在的
                # 僵尸台账也在这一步一并清掉。
                ETLManagedRow.objects.filter(
                    scope=scope, table_name=table, row_id__in=list(stale)
                ).delete()

        for chunk in _chunks(creates, batch_size):
            model.objects.bulk_create(chunk, batch_size=batch_size)
        update_fields = fields
        for chunk in _chunks(updates, batch_size):
            model.objects.bulk_update(chunk, update_fields, batch_size=batch_size)

        ledger_rows = [
            ETLManagedRow(scope=scope, table_name=table, row_id=key, run_id=run_id)
            for key in keys
        ]
        for chunk in _chunks(ledger_rows, batch_size):
            ETLManagedRow.objects.bulk_create(chunk, ignore_conflicts=True, batch_size=batch_size)

    return stats


def _delete_ids(model, ids: Iterable[int]) -> int:
    ids = list(ids)
    total = 0
    for chunk in _chunks(ids, 2000):
        total += model.objects.filter(pk__in=chunk).delete()[0]
    return total


def rebuild_derived(model, objects, *, dry_run=False, batch_size=2000):
    """全量派生表（如地图兼容层）重建；该表无用户数据。"""
    stats = {'table': model._meta.db_table, 'deleted': model.objects.count(),
             'created': len(objects)}
    if dry_run:
        return stats
    with transaction.atomic():
        model.objects.all().delete()
        for chunk in _chunks(objects, batch_size):
            model.objects.bulk_create(chunk, batch_size=batch_size)
    return stats
