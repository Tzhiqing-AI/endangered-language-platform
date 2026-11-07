"""统一 ETL 入口：python manage.py etl <action>。

标准顺序已在命令内固化：rebuild -> enrich(names -> backfill) -> ml(dataset -> train)。
"""
from __future__ import annotations

import json
import os

from django.core.management.base import BaseCommand, CommandError

from Dashboard.etl.config import CSV_DEPRECATION_REASON
from Dashboard.etl.report import collect_status, format_status, run_verify
from Dashboard.etl.runner import (
    EtlRunner, enrich_stages, ml_stages, pipeline_stages, rebuild_stages,
)

DESTRUCTIVE_HINT = (
    '拒绝执行：重建会修改 language_v2 的多张表。\n'
    '请同时满足：环境变量 ALLOW_DESTRUCTIVE_DB_REBUILD=1 且追加 --confirm。\n'
    '只想预览请使用 --dry-run（只读、不写库）。'
)


class Command(BaseCommand):
    help = (
        'ELCat ETL 统一入口。标准顺序：rebuild -> enrich -> ml。\n'
        'CSV 路径已废弃（csv_output 目录已删除）：该历史导出相对 elcat.sql '
        '存在系统性丢失，数据源统一为 elcat_source，请使用 --source elcat（默认）。'
    )

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest='action', required=True)

        p_status = sub.add_parser('status', help='各表行数、关键字段填充率、重复名称、测试记录')
        p_status.add_argument('--json', action='store_true', help='输出 JSON')
        p_status.add_argument('--no-source', action='store_true', help='不连接 elcat_source')

        sub.add_parser('verify', help='源/目标断言与数据质量检查（只读）')

        p_rebuild = sub.add_parser('rebuild', help='从源库重建规范化表')
        p_rebuild.add_argument('--source', choices=['elcat', 'csv'], default='elcat')
        p_rebuild.add_argument('--confirm', action='store_true', help='确认执行破坏性重建')
        p_rebuild.add_argument('--dry-run', action='store_true', help='只校验不写库')
        p_rebuild.add_argument('--resume', action='store_true', help='从上次失败阶段继续')

        p_enrich = sub.add_parser('enrich', help='应用人工名称 + 回填展示字段')
        p_enrich.add_argument('--only', choices=['names', 'backfill'], default=None)
        p_enrich.add_argument('--dry-run', action='store_true')
        p_enrich.add_argument('--resume', action='store_true')

        p_ml = sub.add_parser('ml', help='构建数据集并训练模型（离线）')
        p_ml.add_argument('--dry-run', action='store_true')
        p_ml.add_argument('--resume', action='store_true')

        p_pipe = sub.add_parser('pipeline', help='按标准顺序执行 rebuild -> enrich -> ml')
        p_pipe.add_argument('--confirm', action='store_true')
        p_pipe.add_argument('--dry-run', action='store_true')
        p_pipe.add_argument('--resume', action='store_true')
        p_pipe.add_argument('--skip-ml', action='store_true', help='跳过数据集与训练')

    def handle(self, *args, **options):
        action = options['action']
        if action == 'status':
            payload = collect_status(include_source=not options['no_source'])
            if options['json']:
                self.stdout.write(json.dumps(payload, ensure_ascii=False, default=str,
                                             indent=2))
            else:
                self.stdout.write(format_status(payload))
            return

        if action == 'verify':
            ok, checks, payload = run_verify()
            for item in checks:
                flag = 'PASS' if item['ok'] else 'FAIL'
                self.stdout.write(f'[{flag}] {item["name"]}: {item["detail"]}')
            if payload.get('source_available'):
                self.stdout.write(f'\n源库已连接；对照 {len(payload["source_comparison"])} 张表。')
            else:
                self.stdout.write(f'\n源库不可用：{payload.get("source_error", "未连接")}')
            if not ok:
                raise CommandError('verify 未通过')
            self.stdout.write('\nverify 通过。')
            return

        if action == 'rebuild':
            if options['source'] == 'csv':
                raise CommandError(CSV_DEPRECATION_REASON)
            if not options['dry_run']:
                self._require_destructive(options['confirm'])
            runner = EtlRunner('rebuild', dry_run=options['dry_run'],
                               resume=options['resume'], source='elcat',
                               options={'source': options['source']},
                               stdout=self.stdout.write)
            runner.run(rebuild_stages())
            return

        if action == 'enrich':
            runner = EtlRunner('enrich', dry_run=options['dry_run'],
                               resume=options['resume'],
                               options={'only': options['only']},
                               stdout=self.stdout.write)
            runner.run(enrich_stages(options['only']))
            return

        if action == 'ml':
            runner = EtlRunner('ml', dry_run=options['dry_run'],
                               resume=options['resume'],
                               stdout=self.stdout.write)
            runner.run(ml_stages())
            return

        if action == 'pipeline':
            if not options['dry_run']:
                self._require_destructive(options['confirm'])
            runner = EtlRunner('pipeline', dry_run=options['dry_run'],
                               resume=options['resume'],
                               options={'skip_ml': options['skip_ml']},
                               stdout=self.stdout.write)
            runner.run(pipeline_stages(skip_ml=options['skip_ml']))
            return

        raise CommandError(f'未知 action: {action}')

    def _require_destructive(self, confirmed):
        if os.getenv('ALLOW_DESTRUCTIVE_DB_REBUILD') != '1' or not confirmed:
            raise CommandError(DESTRUCTIVE_HINT)
