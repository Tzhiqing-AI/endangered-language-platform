"""ETL 幂等写入测试。

重点回归历史上的 MySQL 1048 事故：更新对象只赋值变更字段时，
``bulk_update`` 会把未赋值的 NOT NULL 整型列写成 NULL。
这里用一张显式含 "NOT NULL 且无默认值" 整型列的临时表复现，
SQLite 同样会拒绝 NULL，避免测试假绿。
"""
from __future__ import annotations

import pytest
from django.db import IntegrityError, connection, models

from Dashboard.etl.upsert import rebuild_derived, upsert_rows
from Dashboard.models import ETLManagedRow

pytestmark = pytest.mark.django_db(transaction=True)


class UpsertTestRow(models.Model):
    """测试用临时表：counter 为 NOT NULL 且无默认值。"""

    row_id = models.IntegerField(primary_key=True)
    label = models.CharField(max_length=64)
    counter = models.IntegerField()  # NOT NULL，无 default

    class Meta:
        app_label = 'Dashboard'
        db_table = 'test_etl_upsert_row'
        managed = False


@pytest.fixture
def row_model():
    """在测试库内建表/删表，确保 SQLite 真正执行 NOT NULL 约束。

    SQLite 的 schema editor 不能在 atomic 事务里关闭外键检查，
    因此本文件整体使用 ``django_db(transaction=True)``。
    """
    with connection.schema_editor() as editor:
        editor.create_model(UpsertTestRow)
    try:
        yield UpsertTestRow
    finally:
        with connection.schema_editor() as editor:
            editor.delete_model(UpsertTestRow)


FIELDS = ('label', 'counter')


def _payload(row_id, label='a', counter=1):
    # 显式主键必须随载荷传入，否则 bulk_create 走 AutoField 分支
    return {'row_id': row_id, 'label': label, 'counter': counter}


def test_not_null_constraint_is_really_enforced(row_model):
    """先证明临时表不是“假约束”：只赋值变更字段的写法必须报错。"""
    row_model.objects.create(row_id=1, label='old', counter=7)

    partial = row_model(row_id=1, label='new')  # counter 未赋值 -> None
    with pytest.raises(IntegrityError):
        row_model.objects.bulk_update([partial], ['label', 'counter'])


def test_second_write_is_idempotent(row_model):
    incoming = {1: _payload(1, 'a', 1), 2: _payload(2, 'b', 2)}

    first = upsert_rows(row_model, incoming, FIELDS, scope='test')
    assert first['created'] == 2
    assert first['updated'] == 0
    assert row_model.objects.count() == 2

    second = upsert_rows(row_model, incoming, FIELDS, scope='test')
    assert second['created'] == 0
    assert second['updated'] == 0
    assert second['unchanged'] == 2
    assert row_model.objects.count() == 2


def test_complete_payload_preserves_unchanged_not_null_column(row_model):
    """历史事故回归：只改 label 时，counter 不能被写成 NULL。"""
    upsert_rows(row_model, {1: _payload(1, 'old', 7)}, FIELDS, scope='test')
    stats = upsert_rows(row_model, {1: _payload(1, 'new', 7)}, FIELDS, scope='test')

    assert stats['updated'] == 1
    row = row_model.objects.get(row_id=1)
    assert row.label == 'new'
    assert row.counter == 7  # 若回归为“只赋变更字段”，这里会先抛 IntegrityError


def test_ledger_tracks_scope_table_and_run(row_model):
    upsert_rows(row_model, {1: _payload(1), 2: _payload(2)}, FIELDS,
                scope='elcat', run_id=42)

    rows = list(ETLManagedRow.objects.filter(
        scope='elcat', table_name='test_etl_upsert_row').order_by('row_id'))
    assert [r.row_id for r in rows] == [1, 2]
    assert all(r.run_id == 42 for r in rows)


def test_prune_deletes_only_rows_owned_by_etl(row_model):
    """手工插入、未进台账的行不能被 prune 误删。"""
    upsert_rows(row_model, {1: _payload(1, 'a', 1), 2: _payload(2, 'b', 2)},
                FIELDS, scope='test')
    row_model.objects.create(row_id=3, label='manual', counter=3)

    stats = upsert_rows(row_model, {1: _payload(1, 'a', 1)}, FIELDS,
                        scope='test', prune=True)

    assert stats['deleted'] == 1
    assert set(row_model.objects.values_list('row_id', flat=True)) == {1, 3}


def test_prune_with_empty_incoming_clears_table_and_ledger(row_model):
    upsert_rows(row_model, {1: _payload(1), 2: _payload(2)}, FIELDS, scope='test')

    stats = upsert_rows(row_model, {}, FIELDS, scope='test', prune=True)

    assert stats['deleted'] == 2
    assert row_model.objects.count() == 0
    assert ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row').count() == 0


def test_prune_should_remove_ledger_entries_of_deleted_rows(row_model):
    """台账若残留已删行，后续可能按旧归属误删手工重建的同 id 行。"""
    upsert_rows(row_model, {1: _payload(1), 2: _payload(2)}, FIELDS, scope='test')
    upsert_rows(row_model, {1: _payload(1)}, FIELDS, scope='test', prune=True)

    owned = set(ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row'
    ).values_list('row_id', flat=True))
    assert owned == {1}  # 被 prune 删除的 2 已同步移出台账


def test_prune_does_not_delete_manual_row_recreated_with_same_pk(row_model):
    """真实场景回归：prune 后手工用同主键重建，再 prune 不能删它。"""
    upsert_rows(row_model, {1: _payload(1), 2: _payload(2)}, FIELDS, scope='test')
    upsert_rows(row_model, {1: _payload(1)}, FIELDS, scope='test', prune=True)
    assert not ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row', row_id=2).exists()

    row_model.objects.create(row_id=2, label='manual', counter=2)

    stats = upsert_rows(row_model, {1: _payload(1)}, FIELDS, scope='test', prune=True)

    assert stats['deleted'] == 0
    assert stats['created'] == 0
    assert stats['updated'] == 0
    assert row_model.objects.get(row_id=2).label == 'manual'
    assert set(ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row'
    ).values_list('row_id', flat=True)) == {1}


def test_prune_cleans_zombie_ledger_rows_without_target(row_model):
    """台账有登记、目标行早已不存在时，清理应成功且不报错。"""
    upsert_rows(row_model, {1: _payload(1)}, FIELDS, scope='test')
    ETLManagedRow.objects.create(
        scope='test', table_name='test_etl_upsert_row', row_id=5)
    assert not row_model.objects.filter(row_id=5).exists()

    stats = upsert_rows(row_model, {1: _payload(1)}, FIELDS, scope='test', prune=True)

    assert stats['deleted'] == 0  # 僵尸台账没有真实行可删
    assert set(ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row'
    ).values_list('row_id', flat=True)) == {1}


def test_prune_cleans_ledger_atomically_with_failed_write(row_model):
    """台账清理与写入同事务：写入失败时已删的行与台账都要回滚。"""
    upsert_rows(row_model, {1: _payload(1, 'a', 1), 2: _payload(2, 'b', 2)},
                FIELDS, scope='test')

    incoming = {1: _payload(1, 'a', 1), 3: {'row_id': 3, 'label': 'bad'}}
    with pytest.raises(IntegrityError):
        upsert_rows(row_model, incoming, FIELDS, scope='test', prune=True, batch_size=1)

    assert row_model.objects.get(row_id=2).label == 'b'
    assert set(ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row'
    ).values_list('row_id', flat=True)) == {1, 2}


def test_dry_run_counts_without_writing(row_model):
    upsert_rows(row_model, {1: _payload(1, 'a', 1), 2: _payload(2, 'b', 2)},
                FIELDS, scope='test')

    stats = upsert_rows(row_model, {1: _payload(1, 'b', 1), 3: _payload(3, 'c', 3)},
                        FIELDS, scope='test', prune=True, dry_run=True)

    assert stats['created'] == 1
    assert stats['updated'] == 1
    assert stats['deleted'] == 1  # 台账内、已不在源库的 row 2
    assert row_model.objects.count() == 2
    assert row_model.objects.get(row_id=1).label == 'a'
    assert ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row').count() == 2  # dry-run 不写台账


def test_midway_failure_rolls_back_the_whole_batch(row_model):
    """第 2 条因 NOT NULL 失败时，第 1 条也不能留下。"""
    incoming = {
        1: _payload(1, 'ok', 1),
        2: {'row_id': 2, 'label': 'bad'},  # counter 缺失 -> NULL
    }
    with pytest.raises(IntegrityError):
        upsert_rows(row_model, incoming, FIELDS, scope='test', batch_size=1)

    assert row_model.objects.count() == 0
    assert ETLManagedRow.objects.filter(
        scope='test', table_name='test_etl_upsert_row').count() == 0


def test_rebuild_derived_dry_run_and_actual(row_model):
    row_model.objects.create(row_id=9, label='old', counter=9)

    dry = rebuild_derived(row_model, [
        row_model(row_id=1, label='a', counter=1),
        row_model(row_id=2, label='b', counter=2),
    ], dry_run=True)
    assert dry['deleted'] == 1
    assert dry['created'] == 2
    assert row_model.objects.count() == 1

    stats = rebuild_derived(row_model, [
        row_model(row_id=1, label='a', counter=1),
        row_model(row_id=2, label='b', counter=2),
    ])
    assert stats['deleted'] == 1
    assert stats['created'] == 2
    assert set(row_model.objects.values_list('row_id', flat=True)) == {1, 2}
