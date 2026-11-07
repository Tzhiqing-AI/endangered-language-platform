FROM python:3.12-slim

# 依赖集合选择：默认只装 Web 运行时。
# 需要容器内执行 ETL 时改用：
#   docker build --build-arg REQUIREMENTS_FILE=requirements-etl.txt .
# ML 训练依赖不进镜像（模型必须离线训练，镜像只加载 ml/artifacts/ 产物）。
ARG REQUIREMENTS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-etl.txt requirements-ml.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

COPY . .

# 说明：CSV 导入路径已废弃（csv_output 目录已删除，该历史导出相对 elcat.sql 存在系统性数据丢失），
# 容器内 ETL 统一从 elcat_source 源库读取，构建上下文只包含 language/。

# collectstatic 仅需要可用的 SECRET_KEY 占位，不写入运行环境（compose 会覆盖）
RUN DJANGO_SECRET_KEY=build-only-placeholder python manage.py collectstatic --noinput --settings=language.settings

EXPOSE 8000

CMD ["gunicorn", "language.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120"]
