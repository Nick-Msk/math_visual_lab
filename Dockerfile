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
    uv sync --no-install-project

# ШАГ 3: Теперь копируем весь остальной исходный код.
# Этот слой будет меняться при каждой правке, но установка зависимостей (ШАГ 2) уже закэширована.
COPY . .

# ШАГ 4: Устанавливаем сам проект (без повторной установки зависимостей).
# Это быстро, так как все пакеты уже на месте.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

EXPOSE 8050

CMD ["python", "app.py"]

