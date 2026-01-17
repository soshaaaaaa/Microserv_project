# NLP-микросервис на FastAPI

Простой учебный проект для обработки текстов.

## Возможности
- Bag of Words (numpy)
- TF-IDF (numpy)
- LSA (sklearn)
- word2vec (упрощенный вариант через SVD, т.к. в sklearn нет Word2Vec)
- NLTK: токенизация, стоп-слова, стемминг, лемматизация, POS, NER, n-граммы, частотный анализ, корпуса

## Структура
```
Microserv_project/
├── client/
│   ├── corpus.txt
│   └── send_corpus.py
├── server/
│   ├── api/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   └── main.py
└── requirements.txt
```

## Установка
Используйте окружение `D:\microserv\.venv`.
```
D:\microserv\.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск сервера
Из папки `D:\microserv\Microserv_project`:
```
uvicorn server.main:app --reload
```
Или проще:
```
python server/main.py
```
Первый запрос к NLTK может занять время на скачивание моделей.

## Примеры клиента
Клиент работает через простые подсказки в терминале.
```
python client/send_corpus.py
```

Пример для NLTK:
1) Endpoint: `text_nltk/tokenize`
2) Text: `Hello, world!`

## API эндпоинты
- `/` — информация о сервисе
- `/tf-idf` — TF-IDF (numpy)
- `/bag-of-words` — Bag of Words (numpy)
- `/lsa` — LSA (sklearn)
- `/word2vec` - word2vec (SVD-приближение на sklearn)
- `/text_nltk/tokenize`
- `/text_nltk/sent_tokenize`
- `/text_nltk/stopwords`
- `/text_nltk/stem`
- `/text_nltk/lemmatize`
- `/text_nltk/pos`
- `/text_nltk/ner`
- `/text_nltk/freq`
- `/text_nltk/ngrams`
- `/text_nltk/pipeline`
- `/text_nltk/corpora`
