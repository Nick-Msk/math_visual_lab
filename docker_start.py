import subprocess
import sys

def main():
    print("🐳 Сборка Docker-образа...")
    subprocess.run(["docker", "build", "-t", "math_visual_lab", "."], check=True)

    print("\n🚀 Запуск контейнера на порту 8050...")
    try:
        subprocess.run([
            "docker", "run",
            "-p", "8050:8050",
            "-e", "DOCKER_ENV=1",
            "--name", "math_lab",
            "math_visual_lab"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Остановка контейнера...")
        subprocess.run(["docker", "stop", "math_lab"])
        subprocess.run(["docker", "rm", "math_lab"])
        print("✅ Контейнер остановлен и удалён.")

