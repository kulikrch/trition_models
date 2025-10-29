"""
Скрипт для очистки временных файлов проекта
"""
import os
import shutil
import glob


def cleanup_project():
    """Очищает временные файлы проекта"""
    print("Cleaning up temporary files...")
    
    # Паттерны для удаления
    cleanup_patterns = [
        "tmp_rovodev_*",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
        "**/data/cifar-10-*",  # Кеш CIFAR-10
        "**/.DS_Store",
        "**/Thumbs.db"
    ]
    
    files_removed = 0
    dirs_removed = 0
    
    for pattern in cleanup_patterns:
        matches = glob.glob(pattern, recursive=True)
        
        for match in matches:
            try:
                if os.path.isfile(match):
                    os.remove(match)
                    files_removed += 1
                    print(f"Removed file: {match}")
                elif os.path.isdir(match):
                    shutil.rmtree(match)
                    dirs_removed += 1
                    print(f"Removed directory: {match}")
            except Exception as e:
                print(f"Error removing {match}: {e}")
    
    print(f"\nCleanup completed:")
    print(f"  Files removed: {files_removed}")
    print(f"  Directories removed: {dirs_removed}")


def cleanup_docker():
    """Очищает Docker ресурсы"""
    print("Cleaning up Docker resources...")
    
    commands = [
        "docker-compose down -v",
        "docker system prune -f",
        "docker volume prune -f"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Project cleanup script')
    parser.add_argument('--docker', action='store_true', help='Also cleanup Docker resources')
    parser.add_argument('--all', action='store_true', help='Cleanup everything including models')
    
    args = parser.parse_args()
    
    cleanup_project()
    
    if args.docker or args.all:
        cleanup_docker()
    
    if args.all:
        print("\nRemoving models...")
        if os.path.exists("models"):
            shutil.rmtree("models")
            print("Models directory removed")
        
        if os.path.exists("data"):
            shutil.rmtree("data")
            print("Data directory removed")
    
    print("\n✓ Cleanup completed!")