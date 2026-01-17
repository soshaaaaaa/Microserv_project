import requests


def read_corpus(path):
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def main():
    base_url = input("Base URL [http://127.0.0.1:8000]: ").strip() or "http://127.0.0.1:8000"
    endpoint = input("Endpoint (например tf-idf): ").strip() or "tf-idf"
    endpoint = endpoint.lstrip("/")

    if endpoint.startswith("text_nltk"):
        if endpoint == "text_nltk/corpora":
            payload = {}
        else:
            text = input("Text: ").strip()
            payload = {"text": text}
    else:
        file_path = input("File [client/corpus.txt]: ").strip() or "client/corpus.txt"
        payload = {"texts": read_corpus(file_path)}

    response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=30)
    print(response.status_code)
    print(response.text)


main()
