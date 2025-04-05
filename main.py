from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from rag_model import PDFAssistant
import tempfile
import os
import numpy as np
import logging
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Assistant API",
    description="API для анализа PDF документов и ответов на вопросы по их содержимому",
    version="1.0.0"
)

# Настройка CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на список разрешенных доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем экземпляр ассистента с обработкой ошибок
try:
    assistant = PDFAssistant()
    logger.info("PDFAssistant успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка при инициализации PDFAssistant: {str(e)}")
    raise

# Состояние загрузки
upload_status = {
    "in_progress": False,
    "completed": False,
    "error": None,
    "progress": 0,
    "start_time": None,
    "end_time": None
}

@app.get("/status")
async def get_status():
    """Получение статуса загрузки и обработки документа"""
    processing_time = None
    if upload_status["end_time"] and upload_status["start_time"]:
        processing_time = upload_status["end_time"] - upload_status["start_time"]
    
    return {
        **upload_status,
        "chunks": len(assistant.chunks) if hasattr(assistant, "chunks") else 0,
        "processing_time": processing_time
    }

def process_pdf_background(file_path: str):
    """Фоновая обработка PDF"""
    global upload_status
    upload_status["in_progress"] = True
    upload_status["completed"] = False
    upload_status["error"] = None
    upload_status["start_time"] = time.time()

    try:
        num_chunks = assistant.process_pdf(file_path)
        upload_status["completed"] = True
        logger.info(f"PDF обработан успешно, создано {num_chunks} чанков")
    except Exception as e:
        upload_status["error"] = str(e)
        logger.error(f"Ошибка при обработке PDF: {str(e)}")
    finally:
        upload_status["in_progress"] = False
        upload_status["end_time"] = time.time()
        os.unlink(file_path)  # Удаление временного файла

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Загрузка и обработка PDF документа в фоновом режиме"""
    global upload_status
    
    # Проверка формата файла
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Принимаются только PDF файлы")
    
    # Если уже идет обработка, отклоняем запрос
    if upload_status["in_progress"]:
        raise HTTPException(400, "В данный момент уже обрабатывается другой документ")
    
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        
        # Проверка размера файла (увеличиваем до 100MB)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(400, "Файл слишком большой (макс. 100MB)")
        
        tmp.write(content)
        tmp_path = tmp.name
    
    # Запускаем обработку в фоновом режиме
    background_tasks.add_task(process_pdf_background, tmp_path)
    
    return {
        "status": "Обработка PDF запущена",
        "filename": file.filename
    }

@app.post("/ask")
async def ask_question(question: str):
    """Получение ответа на вопрос по документу"""
    if not assistant.chunks:
        raise HTTPException(400, "Сначала загрузите PDF документ!")

    if not question or len(question.strip()) < 3:
        raise HTTPException(400, "Пожалуйста, задайте корректный вопрос (минимум 3 символа)")

    try:
        start_time = time.time()
        result = assistant.ask(question)
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": result["answer"],
            "confidence": round(result["score"], 3),
            "context": result["context"],
            "pages": result["pages"],
            "processing_time_ms": round(processing_time * 1000)
        }
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        raise HTTPException(500, f"Ошибка при генерации ответа: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok", "models_loaded": True}