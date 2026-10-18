#!/usr/bin/env bash
set -e

echo "🐳 Сборка и запуск контейнера (жду готовности сервера)..."
docker compose up --build -d --wait

echo "🌐 Открытие браузера..."
if command -v open >/dev/null 2>&1; then
    open http://localhost:8050
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:8050
elif command -v start >/dev/null 2>&1; then
    start http://localhost:8050
else
    echo "Откройте вручную: http://localhost:8050"
fi

echo ""
echo "✅ Math Visual Lab запущен: http://localhost:8050"
echo "   Логи: docker compose logs -f"
echo "   Стоп: docker compose down"

