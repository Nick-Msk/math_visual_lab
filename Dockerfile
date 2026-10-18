FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY app.py .

# Открываем порт
EXPOSE 8050

# Запускаем приложение
CMD ["python", "app.py"]

