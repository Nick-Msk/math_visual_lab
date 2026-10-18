FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости
COPY pyproject.toml .

RUN pip install --no-cache-dir

# Копируем исходный код
COPY app.py .

# Открываем порт
EXPOSE 8050

# Запускаем приложение
CMD ["python", "app.py"]

