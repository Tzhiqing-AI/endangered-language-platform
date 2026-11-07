"""已停用：csv_output（目录已删除）相对 elcat.sql 存在系统性数据丢失。

保留本文件仅用于说明历史路径；任何执行都会立即退出，不再导入 CSV。
请改用 ``python manage.py etl rebuild --source elcat``。
"""
import sys

MESSAGE = """
import_csv_data.py 已停用。

原因：逐表比对确认 csv_output（历史 CSV 导出，目录已删除）相对 elcat.sql
存在系统性丢失与结构损坏：
  - 多 INSERT 表只保留最后一条 INSERT（auth_user 1437/17813、django_session 5765/6653416）
  - 文本字段换行/引号未转义，出现 57,936 列的畸形记录
  - language_context 仅 65/14704、language_codes 1027/3669、language_sample 1273/6805

请改用数据库路径：
  python manage.py etl rebuild --source elcat --confirm
  python manage.py etl status
"""


def main():
    print(MESSAGE)
    return 2


if __name__ == '__main__':
    sys.exit(main())
