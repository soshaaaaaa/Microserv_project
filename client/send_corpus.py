import requests


def read_corpus(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def main():
    base_url = "http://127.0.0.1:8000"
    corpus = read_corpus("client/corpus.txt")
    text = (
        "I was born in the year 1632, in the city of York, of a good family, though not "
        "of that country; my father being a foreigner of Bremen, who settled first at Hull. "
        "He got a good estate by merchandise, and leaving off his trade, lived afterwards "
        "at York, from whence he had married my mother, whose relations were named Robinson."
    )

    endpoints = [
        "bag-of-words",
        "tf-idf",
        "lsa",
        "word2vec",
        "text_nltk/tokenize",
        "text_nltk/stem",
        "text_nltk/lemmatize",
        "text_nltk/pos",
        "text_nltk/ner",
    ]

    for endpoint in endpoints:
        if endpoint.startswith("text_nltk"):
            payload = {"text": text}
        else:
            payload = {"texts": corpus}

        response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=30)
        print(endpoint, response.status_code)
        print(response.text)


main()
