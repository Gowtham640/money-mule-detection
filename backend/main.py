"""Single entry point: ``python -m backend.main`` starts the API, the
WebSocket server, the transaction spawner and the detection pipeline."""

import uvicorn

from backend import settings


def main():
    uvicorn.run(
        "backend.api:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="warning",
        ws="websockets-sansio",
    )


if __name__ == "__main__":
    main()
