FROM python:3.12-slim

WORKDIR /app

# Копируем все файлы проекта в контейнер
# (Благодаря .dockerignore, папка .venv сюда не попадет)
COPY . .

# Устанавливаем зависимости и сам проект (точка в конце обязательна!)
RUN pip install --no-cache-dir .

EXPOSE 8050

# Запускаем приложение
CMD ["python", "app.py"]

