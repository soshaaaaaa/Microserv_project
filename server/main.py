import sys
from pathlib import Path

from fastapi import FastAPI

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from server.api.routes import router

app = FastAPI(
    title="NLP-микросервис - выполнили: Сошина Арина, Протопопова Дарья, Романова Амелия",
    description="Простой сервис для обработки текста и базовых NLP-признаков, ссылка на Github - https://github.com/soshaaaaaa/Microserv_project",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True)
    
