class DocHash:
    def __init__(self, doc_url, doc_hash):
        self.url = doc_url
        self.hash = doc_hash

    def __str__(self) -> str:
        return f"DocHash(url={self.url}, hash={self.hash})"
