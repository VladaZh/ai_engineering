import os
import sys
import uvicorn
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Запуск FastAPI-сервиса через uvicorn."""

    host = "0.0.0.0"
    port = 8000
    reload = "true"
    log_level = "info"

    uvicorn.run(
        "src.service.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
