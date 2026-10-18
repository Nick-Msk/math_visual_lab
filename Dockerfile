# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

WORKDIR /app

# Устанавливаем uv (быстрый менеджер пакетов, как и локально)
RUN pip install --no-cache-dir uv

# ШАГ 1: Копируем ТОЛЬКО файлы, описывающие зависимости.
# Этот слой будет закэширован и не изменится при правках в app.py.
COPY pyproject.toml README.md ./

# ШАГ 2: Ставим зависимости через uv.
# Используем --mount=type=cache, чтобы uv кэшировал скачанные пакеты между сборками.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip compile pyproject.toml -o /tmp/requirements.txt && \
    uv pip install --system --no-cache-dir -r /tmp/requirements.txt

# ШАГ 3: Теперь копируем весь остальной исходный код.
# Этот слой будет меняться при каждой правке, но установка зависимостей (ШАГ 2) уже закэширована.
COPY app.py ./

EXPOSE 8050

CMD ["python", "app.py"]

