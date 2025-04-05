# PDF Assistant Backend

Серверная часть приложения для анализа PDF документов с использованием RAG (Retrieval Augmented Generation) подхода.

## Возможности

- Загрузка и обработка PDF документов
- Извлечение текста из документов
- Семантический поиск по содержимому
- Генерация ответов на вопросы по документам

## Технологии

- FastAPI для API
- Transformers (Microsoft Phi-3-mini-4k-instruct) для генерации текста
- SentenceTransformers (multilingual-e5-large) для эмбеддингов
- FAISS для быстрого векторного поиска
- pdfplumber для извлечения текста из PDF

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/ВАШ_ЛОГИН/TenderHack.git
cd TenderHack/backend
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
```

3. Активируйте виртуальное окружение:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск

```bash
uvicorn main:app --reload
```

API будет доступно по адресу: http://localhost:8000

## Использование API

### Загрузка PDF
```bash
curl -X POST "http://localhost:8000/upload" -H "Content-Type: multipart/form-data" -F "file=@путь_к_файлу.pdf"
```

### Получение статуса загрузки
```bash
curl -X GET "http://localhost:8000/status"
```

### Задание вопроса
```bash
curl -X POST "http://localhost:8000/ask" -d '{"question": "Ваш вопрос по документу?"}'
```

## Примечания для развертывания

- Модели будут скачаны автоматически при первом запуске
- Для работы требуется минимум 8 ГБ ОЗУ
- Для ускорения работы рекомендуется использовать GPU 