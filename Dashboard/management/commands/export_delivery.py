"""生成毕业设计交付版数据库并导出 structure.sql / data.sql。

设计原则：
1. 默认 dry-run，只读源库并打印计划，不创建/修改任何数据库；
2. 加 --apply 才创建独立的交付库（默认 language_v2_delivery），
   从源库 dump 恢复后只在交付库上脱敏，源库始终只读；
3. 不打印任何真实邮箱、密码哈希、数据库连接串。
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from getpass import getpass
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connections, transaction
from django.db.models import Q

from Community.models import Like, Message, Post, Reply, Report
from Dashboard.models import Language
from User.models import User

SITE_USERNAMES = ('wyy', 'fan', 'sspu', 'htd')
EXAMPLE_DOMAIN = 'example.invalid'
DB_NAME_RE = re.compile(r'^[A-Za-z0-9_]+$')


class Command(BaseCommand):
    help = (
        '生成毕业设计交付版数据：从源库只读 dump，恢复到独立交付库，'
        '脱敏后导出 structure.sql 与 data.sql。默认 dry-run。'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply', action='store_true',
            help='真正执行：创建/重建交付库、脱敏、导出。默认只打印计划。',
        )
        parser.add_argument(
            '--no-input', action='store_true',
            help='不进行交互确认；必须与 --apply 且 --demo-password 同时使用。',
        )
        parser.add_argument(
            '--delivery-db', default='language_v2_delivery',
            help='独立交付库名，默认 language_v2_delivery；不得与源库同名。',
        )
        parser.add_argument(
            '--output-dir', default=None,
            help='导出目录，默认 <工程上级目录>/delivery。',
        )
        parser.add_argument(
            '--contributor-email-mode', choices=('mask', 'keep-authorized'),
            default='mask',
            help=(
                '贡献者邮箱模式：mask=保守模式，全部替换为示例地址（默认）；'
                'keep-authorized=仅保留 --authorized-users-file 中列出的用户名，其余替换。'
            ),
        )
        parser.add_argument(
            '--authorized-users-file', default=None,
            help=(
                'keep-authorized 模式必填。UTF-8 文本，每行一个已获再分发授权的贡献者用户名；'
                '空行和 # 开头的注释忽略。文件中不要写邮箱。'
            ),
        )
        parser.add_argument(
            '--demo-password', default=None,
            help=(
                '本站 4 个账号统一演示口令。不推荐写在命令行（会进入 shell 历史）；'
                '省略时脚本用 getpass 交互输入，不回显、不打印。'
            ),
        )
        parser.add_argument(
            '--clear-private-messages', action='store_true',
            help='清空 community_message.content（保留行和关联），默认保留。',
        )
        parser.add_argument(
            '--clear-reports', action='store_true',
            help='清空 community_report.description，默认保留。',
        )
        parser.add_argument(
            '--recreate-delivery', action='store_true',
            help='交付库已存在时先 DROP 再重建；只影响交付库，不影响源库。',
        )
        parser.add_argument(
            '--drop-delivery-db', action='store_true',
            help='导出并校验成功后删除交付库（导出文件保留）。',
        )
        parser.add_argument(
            '--mysqldump', default=None,
            help='mysqldump 可执行文件路径；默认查 PATH，Windows 回退到 MySQL 8.0。',
        )
        parser.add_argument(
            '--mysql', default=None,
            help='mysql 可执行文件路径；默认查 PATH，Windows 回退到 MySQL 8.0。',
        )

    def handle(self, *args, **options):
        db = settings.DATABASES.get('default', {})
        if not str(db.get('ENGINE', '')).endswith('mysql'):
            raise CommandError('该命令只支持 MySQL 后端。')

        source_db = str(db.get('NAME') or '')
        delivery_db = str(options['delivery_db'] or '')
        if not source_db:
            raise CommandError('settings.DATABASES["default"]["NAME"] 为空。')
        if not DB_NAME_RE.match(delivery_db):
            raise CommandError('--delivery-db 只能是字母、数字、下划线。')
        if source_db.lower() == delivery_db.lower():
            raise CommandError('交付库名不能与源库同名；脚本不会直接操作源库。')

        output_dir = Path(
            options['output_dir'] or (Path(settings.BASE_DIR).parent / 'delivery')
        ).resolve()

        authorized = self._load_authorized_users(
            options['contributor_email_mode'], options['authorized_users_file']
        )
        plan = self._collect_plan(source_db, delivery_db, output_dir,
                                  options['contributor_email_mode'], authorized,
                                  options['clear_private_messages'],
                                  options['clear_reports'])

        self._print_plan(plan)
        if not options['apply']:
            self.stdout.write(self.style.WARNING(
                '\nDRY-RUN：未创建交付库、未修改任何数据、未导出文件。'
                '\n确认无误后加 --apply 真正执行。'
            ))
            return

        self._confirm(plan, options['no_input'], options['recreate_delivery'])
        demo_password = self._resolve_demo_password(
            options['demo_password'], options['no_input']
        )
        binaries = self._resolve_binaries(options['mysqldump'], options['mysql'])

        self._run_apply(plan, demo_password, binaries,
                        recreate=options['recreate_delivery'],
                        drop_after=options['drop_delivery_db'])

    # ------------------------------------------------------------------
    # 计划与只读核对
    # ------------------------------------------------------------------
    def _load_authorized_users(self, mode: str, path: str | None) -> set[str]:
        if mode == 'mask':
            if path:
                raise CommandError('--authorized-users-file 仅用于 keep-authorized 模式。')
            return set()

        if not path:
            raise CommandError(
                'keep-authorized 模式必须提供 --authorized-users-file；'
                '文件每行一个已获授权的贡献者用户名。'
            )
        file_path = Path(path)
        if not file_path.is_file():
            raise CommandError('--authorized-users-file 文件不存在。')

        names = set()
        try:
            for line in file_path.read_text(encoding='utf-8-sig').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                names.add(line)
        except OSError as exc:
            raise CommandError(f'读取授权名单失败：{exc.__class__.__name__}') from exc

        if not names:
            raise CommandError('授权名单为空；请核对文件内容。')
        return names

    def _collect_plan(self, source_db, delivery_db, output_dir, mode,
                      authorized, clear_messages, clear_reports):
        try:
            site_users = list(
                User.objects.filter(username__in=SITE_USERNAMES)
                .order_by('username')
                .values('username', 'admin', 'expert', 'is_active')
            )
            contributors = User.objects.exclude(username__in=SITE_USERNAMES)
            contributor_count = contributors.count()
            contributor_with_email = contributors.exclude(
                Q(email__isnull=True) | Q(email='')
            ).count()
            contributors_with_marker = contributors.filter(
                password__startswith='!'
            ).count()
            messages = Message.objects.count()
            reports = Report.objects.count()
            reports_with_description = Report.objects.exclude(
                Q(description__isnull=True) | Q(description='')
            ).count()
        except Exception as exc:
            raise CommandError(
                f'只读核对失败：{exc.__class__.__name__}；'
                '请确认源库可连接且已执行 migrate。'
            ) from exc

        missing = sorted(set(SITE_USERNAMES) - {u['username'] for u in site_users})
        if missing:
            raise CommandError(
                '本站账号未找齐：' + '、'.join(missing) +
                '；请先在源库确认这 4 个账号存在。'
            )

        contributor_names = set(
            contributors.values_list('username', flat=True)
        )
        unknown_authorized = authorized - contributor_names
        if unknown_authorized:
            raise CommandError(
                f'授权名单中有 {len(unknown_authorized)} 个用户名不在贡献者记录中；'
                '请核对名单，脚本不打印具体用户名。'
            )

        if mode == 'mask':
            mask_count = contributor_count
            keep_count = 0
            keep_with_email = 0
        else:
            mask_count = contributor_count - len(authorized)
            keep_count = len(authorized)
            keep_with_email = contributors.filter(
                username__in=authorized
            ).exclude(Q(email__isnull=True) | Q(email='')).count()

        return {
            'source_db': source_db,
            'delivery_db': delivery_db,
            'output_dir': output_dir,
            'site_users': site_users,
            'contributor_count': contributor_count,
            'contributor_with_email': contributor_with_email,
            'contributors_with_marker': contributors_with_marker,
            'mode': mode,
            'authorized': authorized,
            'mask_count': mask_count,
            'keep_count': keep_count,
            'keep_with_email': keep_with_email,
            'messages': messages,
            'reports': reports,
            'reports_with_description': reports_with_description,
            'clear_messages': clear_messages,
            'clear_reports': clear_reports,
        }

    def _print_plan(self, plan):
        self.stdout.write('交付版数据生成计划（源库只读，所有写入只发生在交付库）')
        self.stdout.write(f'  源库（只读 dump）：{plan["source_db"]}')
        self.stdout.write(f'  交付库（将创建/重建）：{plan["delivery_db"]}')
        self.stdout.write(f'  导出目录：{plan["output_dir"]}')
        self.stdout.write('')
        self.stdout.write(f'  本站账号：{len(plan["site_users"])} 个')
        for user in plan['site_users']:
            role = []
            if user['admin']:
                role.append('管理员')
            if user['expert']:
                role.append('专家')
            if not role:
                role.append('普通用户')
            self.stdout.write(
                f'    - {user["username"]}（{"+".join(role)}，'
                f'{"启用" if user["is_active"] else "未启用"}）：'
                '保留用户名/角色，邮箱改为示例地址，密码用统一演示口令'
            )
        self.stdout.write('')
        self.stdout.write(
            f'  ELCat 贡献者：{plan["contributor_count"]} 条；'
            f'其中原有邮箱 {plan["contributor_with_email"]} 条，'
            f'不可用密码标记 {plan["contributors_with_marker"]} 条（保持不变）'
        )
        if plan['mode'] == 'mask':
            self.stdout.write(
                '  邮箱模式：mask（保守，默认）——全部贡献者邮箱替换为示例地址；'
                f'计划替换 {plan["mask_count"]} 条，保留 0 条'
            )
        else:
            self.stdout.write(
                '  邮箱模式：keep-authorized——仅保留授权名单内的贡献者邮箱；'
                f'计划保留 {plan["keep_count"]} 条（其中有邮箱 {plan["keep_with_email"]} 条），'
                f'其余 {plan["mask_count"]} 条替换为示例地址'
            )
        self.stdout.write('')
        self.stdout.write(
            f'  私信/通知：community_message {plan["messages"]} 条；'
            + ('将清空 content' if plan['clear_messages'] else '保留 content')
        )
        self.stdout.write(
            f'  举报：community_report {plan["reports"]} 条（'
            f'有 description {plan["reports_with_description"]} 条）；'
            + ('将清空 description' if plan['clear_reports'] else '保留 description')
        )
        self.stdout.write('')
        self.stdout.write('  将导出：structure.sql（仅结构）、data.sql（仅数据）')
        self.stdout.write('  校验：表数量、关键表行数、文件大小、SHA256')

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def _confirm(self, plan, no_input, recreate):
        self.stderr.write(self.style.WARNING(
            '\n警告：即将创建/写入交付库，并修改 user 表（邮箱、密码）。'
            f'\n交付库：{plan["delivery_db"]}；源库只读，不会被修改。'
        ))
        if recreate:
            self.stderr.write(self.style.WARNING(
                f'--recreate-delivery 会先删除已有的 {plan["delivery_db"]}。'
            ))
        if no_input:
            return

        answer = input('确认执行请输入 APPLY：').strip()
        if answer != 'APPLY':
            raise CommandError('已取消，未写入任何数据。')

    def _resolve_demo_password(self, supplied, no_input):
        if supplied:
            password = supplied
        else:
            if no_input:
                raise CommandError(
                    '--no-input 模式下必须提供 --demo-password；'
                    '或去掉 --no-input 用交互输入。'
                )
            password = getpass('演示口令（不会回显，请与纸质说明一致）：')
        if not password or len(password) < 6:
            raise CommandError('演示口令不能为空且至少 6 位。')
        return password

    def _resolve_binaries(self, mysqldump_arg, mysql_arg):
        def resolve(explicit, name):
            if explicit:
                path = Path(explicit)
                if not path.is_file():
                    raise CommandError(f'{name} 路径不存在。')
                return str(path)
            found = shutil.which(name)
            if found:
                return found
            fallback = Path(
                r'C:\Program Files\MySQL\MySQL Server 8.0\bin'
            ) / f'{name}.exe'
            if fallback.is_file():
                return str(fallback)
            raise CommandError(
                f'找不到 {name}；请用 --{name} 指定完整路径（Windows 默认路径：'
                r'C:\Program Files\MySQL\MySQL Server 8.0\bin）。'
            )

        return {
            'mysqldump': resolve(mysqldump_arg or '', 'mysqldump'),
            'mysql': resolve(mysql_arg or '', 'mysql'),
        }

    def _mysql_connection_args(self):
        db = settings.DATABASES['default']
        host = str(db.get('HOST') or '127.0.0.1')
        port = str(db.get('PORT') or '3306')
        user = str(db.get('USER') or 'root')
        return ['--protocol=tcp', f'--host={host}', f'--port={port}',
                f'--user={user}']

    def _mysql_env(self):
        db = settings.DATABASES['default']
        env = dict(os.environ)
        password = db.get('PASSWORD')
        if password:
            # 不放在命令行参数里，避免出现在进程列表/历史记录。
            env['MYSQL_PWD'] = str(password)
        return env

    def _run_apply(self, plan, demo_password, binaries, recreate, drop_after):
        source_db = plan['source_db']
        delivery_db = plan['delivery_db']
        output_dir = plan['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)

        self._create_delivery_db(delivery_db, recreate)
        alias = None
        try:
            with tempfile.TemporaryDirectory(prefix='delivery_export_') as tmp:
                source_dump = Path(tmp) / 'source.sql'
                self._dump_source(binaries['mysqldump'], source_db, source_dump)
                self._restore_dump(binaries['mysql'], delivery_db, source_dump)

            alias = self._register_alias(delivery_db)
            self._sanitize(alias, delivery_db, plan, demo_password)

            structure_path = output_dir / 'structure.sql'
            data_path = output_dir / 'data.sql'
            self._dump_structure(binaries['mysqldump'], delivery_db, structure_path)
            self._dump_data(binaries['mysqldump'], delivery_db, data_path)

            self._verify(alias, delivery_db, plan, structure_path, data_path)
        finally:
            if alias:
                connections[alias].close()

        if drop_after:
            self._drop_delivery_db(delivery_db)
            self.stdout.write(self.style.SUCCESS(
                f'\n已完成；交付库 {delivery_db} 已删除，导出文件保留在 {output_dir}'
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f'\n已完成；交付库 {delivery_db} 保留用于复核；'
                f'导出文件在 {output_dir}'
            ))

    def _create_delivery_db(self, delivery_db, recreate):
        with connections['default'].cursor() as cursor:
            cursor.execute(
                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                "WHERE SCHEMA_NAME=%s", [delivery_db],
            )
            exists = cursor.fetchone() is not None
            if exists and not recreate:
                raise CommandError(
                    f'交付库 {delivery_db} 已存在；'
                    '请换名，或显式加 --recreate-delivery 删除重建。'
                )
            if exists and recreate:
                cursor.execute(f'DROP DATABASE `{delivery_db}`')
            cursor.execute(
                f'CREATE DATABASE `{delivery_db}` '
                'CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci'
            )
        self.stdout.write(f'已创建交付库：{delivery_db}')

    def _dump_source(self, mysqldump, source_db, path):
        args = [
            mysqldump, *self._mysql_connection_args(),
            '--single-transaction', '--quick', '--routines', '--triggers',
            '--events', '--no-tablespaces', '--set-gtid-purged=OFF',
            '--default-character-set=utf8mb4', f'--result-file={path}',
            source_db,
        ]
        self._run(args, '源库只读 dump')

    def _restore_dump(self, mysql, delivery_db, path):
        args = [mysql, *self._mysql_connection_args(),
                '--default-character-set=utf8mb4', delivery_db]
        with open(path, 'rb') as handle:
            self._run(args, '恢复交付库', stdin=handle)
        self.stdout.write('已将源库 dump 恢复到交付库（源库未被写入）')

    def _dump_structure(self, mysqldump, delivery_db, path):
        args = [
            mysqldump, *self._mysql_connection_args(),
            '--single-transaction', '--quick', '--routines', '--triggers',
            '--events', '--no-tablespaces', '--set-gtid-purged=OFF',
            '--default-character-set=utf8mb4', '--no-data',
            f'--result-file={path}', delivery_db,
        ]
        self._run(args, '导出 structure.sql')

    def _dump_data(self, mysqldump, delivery_db, path):
        args = [
            mysqldump, *self._mysql_connection_args(),
            '--single-transaction', '--quick', '--no-create-info',
            '--skip-triggers', '--skip-add-locks', '--complete-insert',
            '--no-tablespaces', '--set-gtid-purged=OFF',
            '--default-character-set=utf8mb4',
            f'--result-file={path}', delivery_db,
        ]
        self._run(args, '导出 data.sql')

    def _run(self, args, action, stdin=None):
        try:
            proc = subprocess.run(
                args, stdin=stdin, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=self._mysql_env(), check=False,
            )
        except OSError as exc:
            raise CommandError(
                f'{action} 失败：无法启动 MySQL 客户端（{exc.__class__.__name__}）'
            ) from exc
        if proc.returncode != 0:
            raise CommandError(
                f'{action} 失败，退出码 {proc.returncode}；'
                '请检查本机 MySQL 客户端权限。错误详情含连接信息，已省略。'
            )

    def _register_alias(self, delivery_db):
        alias = 'delivery_sanitize'
        config = dict(settings.DATABASES['default'])
        config['NAME'] = delivery_db
        connections.databases[alias] = config
        connections[alias].close()
        return alias

    def _sanitize(self, alias, delivery_db, plan, demo_password):
        with transaction.atomic(using=alias):
            site_users = list(
                User.objects.using(alias)
                .filter(username__in=SITE_USERNAMES)
                .order_by('username')
            )
            if len(site_users) != len(SITE_USERNAMES):
                raise CommandError('交付库中的本站账号数量与源库不一致，已回滚。')

            for user in site_users:
                user.email = f'{user.username}@{EXAMPLE_DOMAIN}'
                # 用 User.set_password 正常生成哈希，不写任何哈希常量。
                user.set_password(demo_password)
                user.save(using=alias, update_fields=['email', 'password'])

            contributors = list(
                User.objects.using(alias)
                .exclude(username__in=SITE_USERNAMES)
                .order_by('username')
            )
            if plan['mode'] == 'mask':
                mask_users = contributors
            else:
                mask_users = [
                    user for user in contributors
                    if user.username not in plan['authorized']
                ]

            for index, user in enumerate(mask_users, start=1):
                user.email = f'elcat-{index:04d}@{EXAMPLE_DOMAIN}'
            if mask_users:
                User.objects.using(alias).bulk_update(
                    mask_users, ['email'], batch_size=500
                )

            if plan['clear_messages']:
                Message.objects.using(alias).update(content='')
            if plan['clear_reports']:
                Report.objects.using(alias).update(description='')

        self.stdout.write(
            f'已脱敏 {len(site_users)} 个本站账号、'
            f'{len(mask_users)} 个贡献者邮箱（交付库 {delivery_db}）'
        )

    def _verify(self, alias, delivery_db, plan, structure_path, data_path):
        with connections[alias].cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=%s", [delivery_db],
            )
            table_count = cursor.fetchone()[0]

        counts = {
            'user': User.objects.using(alias).count(),
            'language': Language.objects.using(alias).count(),
            'community_post': Post.objects.using(alias).count(),
            'community_reply': Reply.objects.using(alias).count(),
            'community_like': Like.objects.using(alias).count(),
            'community_message': Message.objects.using(alias).count(),
            'community_report': Report.objects.using(alias).count(),
        }
        non_example_emails = (
            User.objects.using(alias)
            .exclude(Q(email__isnull=True) | Q(email=''))
            .exclude(email__endswith=f'@{EXAMPLE_DOMAIN}')
            .count()
        )
        site_pbkdf2 = User.objects.using(alias).filter(
            username__in=SITE_USERNAMES,
            password__startswith='pbkdf2_sha256$',
        ).count()
        contributor_markers = User.objects.using(alias).exclude(
            username__in=SITE_USERNAMES
        ).filter(password__startswith='!').count()

        if plan['mode'] == 'mask' and non_example_emails:
            raise CommandError(
                f'校验失败：仍有 {non_example_emails} 条非示例邮箱；未完成导出。'
            )
        if site_pbkdf2 != len(SITE_USERNAMES):
            raise CommandError(
                f'校验失败：本站账号可用哈希数为 {site_pbkdf2}，'
                f'预期 {len(SITE_USERNAMES)}。'
            )
        if table_count != 62:
            raise CommandError(
                f'校验失败：交付库表数量为 {table_count}，预期 62。'
            )

        self.stdout.write('')
        self.stdout.write('校验信息：')
        self.stdout.write(f'  表数量：{table_count}')
        self.stdout.write(
            '  关键表行数：' + '，'.join(
                f'{name}={value}' for name, value in counts.items()
            )
        )
        self.stdout.write(
            f'  邮箱：非示例地址 {non_example_emails} 条；'
            f'本站账号 PBKDF2 哈希 {site_pbkdf2} 条；'
            f'贡献者不可用标记 {contributor_markers} 条'
        )
        self.stdout.write(
            f'  消息/举报：'
            f'community_message={counts["community_message"]}，'
            f'community_report={counts["community_report"]}'
        )
        for label, path in (
            ('structure.sql', structure_path),
            ('data.sql', data_path),
        ):
            size = path.stat().st_size
            self.stdout.write(
                f'  {label}：{size} bytes（{size / 1024 / 1024:.2f} MB），'
                f'SHA256={self._sha256(path)}'
            )

        if plan['mode'] == 'keep-authorized':
            self.stdout.write(
                f'  授权模式：计划保留 {plan["keep_count"]} 条（其中有邮箱 '
                f'{plan["keep_with_email"]} 条）；实际保留非示例邮箱 '
                f'{non_example_emails} 条'
            )

    def _sha256(self, path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _drop_delivery_db(self, delivery_db):
        with connections['default'].cursor() as cursor:
            cursor.execute(f'DROP DATABASE `{delivery_db}`')
