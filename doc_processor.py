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
    def create_document_hashes(self, docs_queue):
        docHashes = []

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = []

            progress = tqdm(total=docs_queue.qsize(), desc="Setting up processes")

            while not docs_queue.empty():
                doc = docs_queue.get()
                result = pool.apply_async(calculate_hash, args=(doc,))
                results.append(result)
                docs_queue.task_done()
                progress.update(1)

            # Wait for all tasks to complete
            for result in tqdm(results, desc="Calculating Hashes"):
                docHash = result.get()

                docHashes.append(docHash)

        return docHashes
