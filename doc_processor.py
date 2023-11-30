import multiprocessing

from tqdm import tqdm
import nltk
from datasketch import MinHash

from models import DocHash


def create_minhash(doc):
    m = MinHash()
    for d in nltk.word_tokenize(doc):
        m.update(d.encode("utf8"))
    return m


def calculate_hash(doc) -> DocHash:
    hash = create_minhash(doc["text"])

    return DocHash(doc_url=doc["url"], doc_hash=hash)


class DocProcessor:
    def __init__(self, docs_queue):
        self.docs_queue = docs_queue

    def create_document_hashes(self):
        docHashes = []

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = []

            while not self.docs_queue.empty():
                doc = self.docs_queue.get()
                result = pool.apply_async(calculate_hash, args=(doc,))
                results.append(result)
                self.docs_queue.task_done()

            # Wait for all tasks to complete
            for result in tqdm(results):
                docHash = result.get()

                docHashes.append(docHash)

        return docHashes
