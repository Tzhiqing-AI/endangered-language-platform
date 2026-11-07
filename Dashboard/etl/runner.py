"""ETL 执行器：运行日志、备份、事务、dry-run、resume 与缓存失效。"""
from __future__ import annotations

import gzip
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime

from django.core.management import call_command
from django.db import transaction
from django.utils import timezone

from Dashboard.models import ETLRun, ETLStep
from Dashboard.etl.config import BACKUP_DIR
from Dashboard.etl.report import STATUS_MODELS
from Dashboard.etl.steps import (
    StepContext, apply_names, backfill_language_fields, build_dataset,
    invalidate_caches, rebuild_elcat, train_models,
)


@dataclass
class Stage:
    name: str
    func: object


def rebuild_stages():
    return [Stage('rebuild', rebuild_elcat)]


def enrich_stages(only=None):
    stages = []
    if only in (None, 'names'):
        stages.append(Stage('names', apply_names))
    if only in (None, 'backfill'):
        stages.append(Stage('backfill', backfill_language_fields))
    return stages


def ml_stages():
    return [Stage('dataset', build_dataset), Stage('train', train_models)]


def pipeline_stages(skip_ml=False):
    stages = rebuild_stages() + enrich_stages()
    if not skip_ml:
        stages += ml_stages()
    return stages


BACKUP_LABELS = [f'Dashboard.{model.__name__}' for model in STATUS_MODELS]
BACKUP_LABELS.append('Dashboard.LanguageLocation')


class EtlRunner:
    def __init__(self, command, *, dry_run=False, resume=False, source='elcat',
                 options=None, stdout=None):
        self.command = command
        self.dry_run = dry_run
        self.resume = resume
        self.source = source
        self.options = options or {}
        self.stdout = stdout or (lambda msg: print(msg))

    def run(self, stages):
        completed, previous = self._resume_state()
        run = None
        if not self.dry_run:
            run = ETLRun.objects.create(
                command=self.command, source=self.source,
                options=json.dumps({'dry_run': self.dry_run,
                                    'resume': self.resume,
                                    'resumed_from': previous.id if previous else None,
                                    **self.options}, ensure_ascii=False),
            )
        ctx = StepContext(run_id=run.id if run else None, dry_run=self.dry_run)
        backup_path = ''
        started = time.time()
        current_stage = 'init'
        try:
            if self.command in ('rebuild', 'pipeline') and not self.dry_run:
                current_stage = 'backup'
                self.stdout('创建重建前备份...')
                backup_path = self._backup(run)
                if run is not None:
                    run.backup_path = backup_path
                    run.save(update_fields=['backup_path'])
                self.stdout(f'备份完成: {backup_path}')

            for stage in stages:
                current_stage = stage.name
                if stage.name in completed:
                    self.stdout(f'[resume] 跳过已完成阶段: {stage.name}')
                    if not self.dry_run:
                        _write_step(run, stage.name, 'skipped', 0,
                                    f'resumed from run #{previous.id}')
                    continue
                self.stdout(f'[stage] {stage.name} ...')
                step_start = time.time()
                with transaction.atomic():
                    result = stage.func(ctx)
                elapsed = round(time.time() - step_start, 1)
                rows = _rows_of(result)
                self.stdout(f'[stage] {stage.name} 完成，影响 {rows} 行，{elapsed}s')
                if not self.dry_run:
                    _write_step(run, stage.name, 'success', rows, f'{elapsed}s')

            if not self.dry_run:
                run.status = 'success'
                run.stats = json.dumps(ctx.stats, ensure_ascii=False, default=str)
                run.finished_at = timezone.now()
                run.save(update_fields=['status', 'stats', 'finished_at'])
                invalidate_caches()
            self.stdout(f'\nETL {self.command} 完成，耗时 {round(time.time() - started, 1)}s')
            return {'run_id': run.id if run else None, 'backup': backup_path,
                    'stats': ctx.stats}
        except Exception as exc:
            if not self.dry_run and run is not None:
                _write_step(run, current_stage, 'failed', 0, str(exc))
                run.status = 'failed'
                run.error = str(exc)
                run.finished_at = timezone.now()
                run.save(update_fields=['status', 'error', 'finished_at'])
            self.stdout(f'\nETL {self.command} 失败：{exc}')
            raise

    def _resume_state(self):
        if not self.resume:
            return set(), None
        last = ETLRun.objects.filter(
            command=self.command, status__in=('running', 'failed')
        ).exclude(dry_run=True).order_by('-id').first()
        if last is None:
            self.stdout('没有可恢复的未完成运行，按全新运行开始。')
            return set(), None
        completed = set(last.steps.filter(status='success').values_list('name', flat=True))
        self.stdout(f'恢复运行 #{last.id}，已完成阶段: {sorted(completed) or "无"}')
        return completed, last

    def _backup(self, run):
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw = BACKUP_DIR / f'before_rebuild_{run.id}_{stamp}.json'
        call_command('dumpdata', *BACKUP_LABELS, output=str(raw), verbosity=0)
        gz = raw.with_suffix('.json.gz')
        with raw.open('rb') as src, gzip.open(gz, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        raw.unlink()
        return str(gz)


def _rows_of(result):
    if isinstance(result, dict):
        if 'changed' in result:
            return int(result['changed'])
        if 'updated' in result:
            return int(result['updated'])
        total = 0
        for value in result.values():
            # 只统计真正的写表统计项；source_rows_not_in_target 等指标不是写入行数
            if isinstance(value, dict) and 'table' in value:
                total += int(value.get('created', 0)) + int(value.get('updated', 0))
        return total
    return 0


def _write_step(run, name, status, rows, message):
    if run is None:
        return
    ETLStep.objects.update_or_create(
        run=run, name=name,
        defaults={'status': status, 'rows_affected': rows, 'message': message[:2000],
                  'finished_at': timezone.now()},
    )
