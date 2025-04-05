from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import torch
from tqdm import tqdm
import os
import gc
import nltk
import re
import numpy as np


class PDFAssistant:
    def __init__(self):
        # Загружаем NLTK для более качественного разбиения текста
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
        
        print("Инициализация моделей...")
        
        # Используем более мощную модель для эмбеддингов
        self.embedder = SentenceTransformer(
            'intfloat/multilingual-e5-large',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_folder='./models'
        )
        
        # Более мощная открытая генеративная модель
        model_name = "microsoft/Phi-3-mini-4k-instruct"  # Хорошее соотношение качество/ресурсы
        
        print(f"Загрузка генеративной модели {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir='./models'
        )
        
        # Загружаем модель в обычном режиме для работы на CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir='./models',
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Очистка памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
        # Настройка параметров
        self.chunk_size = 1000      # Размер чанка в символах
        self.chunk_overlap = 200    # Перекрытие чанков для сохранения контекста
        
        # Улучшенные промпты для различных типов запросов
        self.system_prompt = """Ты - профессиональный помощник по работе с документами. Твоя задача - давать точные, конкретные ответы
на вопросы пользователей, основываясь исключительно на предоставленном контексте.

Правила:
1. Отвечай только на основе предоставленного контекста
2. Если в контексте нет ответа на вопрос, честно признай это
3. Не придумывай информацию
4. Структурируй ответы для лучшего понимания
5. Используй маркированные списки, когда это уместно
6. Приводи конкретные цитаты из документа, если это необходимо

ВАЖНО: Твои ответы должны быть максимально полезными и точными."""

        self.qa_template = """{system}

Контекст:
{context}

Вопрос: {question}

Ответ:"""

    def preprocess_text(self, text):
        """Предобработка текста с улучшенной очисткой"""
        # Удаление избыточных пробелов и переносов
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов, сохраняя структуру предложений
        text = re.sub(r'[^\w\s.,;:!?\(\)«»\-—–]', '', text)
        return text.strip()

    def create_semantic_chunks(self, text):
        """Создание семантически значимых чанков"""
        # Разбиваем на параграфы
        paragraphs = text.split('\n\n')
        
        # Удаляем пустые параграфы и предобрабатываем текст
        paragraphs = [self.preprocess_text(p) for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_page = 1
        
        for paragraph in paragraphs:
            # Если текущий чанк + новый параграф не превышает лимит
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + " "
            else:
                # Добавляем текущий чанк в список, если он не пустой
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": current_page
                    })
                    
                # Начинаем новый чанк с перекрытием
                if len(paragraph) > self.chunk_size:
                    # Если параграф слишком длинный, разбиваем на предложения
                    sentences = nltk.sent_tokenize(paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append({
                                    "text": current_chunk.strip(),
                                    "page": current_page
                                })
                            current_chunk = sentence + " "
                else:
                    current_chunk = paragraph + " "
                    
        # Добавляем последний чанк, если он не пустой
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "page": current_page
            })
            
        return chunks

    def process_pdf(self, file_path: str):
        """Улучшенная обработка PDF с прогресс-баром и метаданными"""
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Извлечение текста из PDF ({total_pages} страниц)...")
                
                all_chunks = []
                
                for page_num, page in enumerate(tqdm(pdf.pages, desc="Обработка страниц")):
                    text = page.extract_text() or ""
                    if text.strip():
                        page_chunks = self.create_semantic_chunks(text)
                        for chunk in page_chunks:
                            chunk["page"] = page_num + 1
                        all_chunks.extend(page_chunks)
            
            # Обновляем данные
            self.chunks = [chunk["text"] for chunk in all_chunks]
            self.metadata = [{"page": chunk["page"]} for chunk in all_chunks]
            
            print(f"Создано {len(self.chunks)} смысловых чанков из документа")
            
            # Специальный префикс для E5
            prefixed_chunks = [f'passage: {chunk}' for chunk in self.chunks]

            # Генерация эмбеддингов
            print(f"Создание эмбеддингов для {len(prefixed_chunks)} чанков...")
            
            # Обрабатываем партиями для экономии памяти
            batch_size = 8
            all_embeddings = []
            
            for i in tqdm(range(0, len(prefixed_chunks), batch_size), desc="Создание эмбеддингов"):
                batch = prefixed_chunks[i:i+batch_size]
                with torch.no_grad():
                    embeddings = self.embedder.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    ).cpu().numpy()
                    all_embeddings.append(embeddings)
            
            # Объединяем все эмбеддинги
            embeddings = np.vstack(all_embeddings)
            
            # Создание индекса HNSW для быстрого поиска
            try:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 соседа для HNSW
                self.index.add(embeddings)
                print("PDF успешно обработан и проиндексирован.")
            except Exception as e:
                print(f"Ошибка при создании индекса FAISS: {str(e)}")
                raise
            
            return len(self.chunks)
        except Exception as e:
            print(f"Ошибка при обработке PDF: {str(e)}")
            raise

    def ask(self, question: str, top_k=5):
        """Улучшенная генерация ответов с контекстным окном"""
        if not self.chunks or self.index is None:
            return {
                "answer": "Пожалуйста, загрузите документ перед тем, как задавать вопросы.",
                "score": 0.0,
                "context": "",
                "pages": []
            }
        
        # Формируем запрос с префиксом для E5
        prefixed_question = f'query: {question}'

        # Поиск по индексу
        query_embed = self.embedder.encode(prefixed_question)
        distances, indices = self.index.search(query_embed.reshape(1, -1), top_k)
        
        # Формирование контекста из найденных чанков
        context_chunks = []
        relevant_pages = set()
        
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunks):
                context_chunks.append(self.chunks[idx])
                relevant_pages.add(self.metadata[idx]["page"])
        
        if not context_chunks:
            return {
                "answer": "К сожалению, в документе не найдено релевантной информации по вашему вопросу.",
                "score": 0.0,
                "context": "Контекст отсутствует.",
                "pages": []
            }
        
        # Объединяем чанки в контекст
        context = "\n\n".join(context_chunks)
        
        # Формируем промпт для генеративной модели
        prompt = self.qa_template.format(
            system=self.system_prompt,
            context=context[:3500],  # Ограничиваем контекст для экономии токенов
            question=question
        )
        
        # Токенизируем промпт
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодируем ответ и удаляем промпт из ответа
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ модели
        response_start = full_output.find("Ответ:") + 6
        response = full_output[response_start:].strip() if response_start > 6 else "Не удалось сгенерировать ответ"
        
        # Вычисляем оценку уверенности на основе расстояний к топ-чанкам
        mean_distance = float(np.mean(distances[0]))
        confidence = float(1.0 / (1.0 + mean_distance))
        confidence = min(max(confidence, 0.0), 1.0)  # Нормализуем
        
        return {
            "answer": response,
            "score": confidence,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "pages": sorted(list(relevant_pages))
        }