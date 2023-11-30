import argparse
import hashlib
import json
import os
from multiprocessing import Pool, cpu_count
from queue import Queue
from threading import Thread
import nltk
import jieba
import pandas as pd
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class DocumentProcessor:
    def __init__(self, docs_queue, clean_docs_queue, num_perm, threshold):
        self.docs_queue = docs_queue
        self.clean_docs_queue = clean_docs_queue
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.duplicates = set()
        self.doc_count = 0
        self.removed_docs = 0

    def process_doc_worker(self, doc):
        minhash = create_minhash(doc['text'])
        self.lsh.insert(f"doc_{hash(doc)}", minhash)
        result = self.lsh.query(minhash)

        is_duplicate = len(result) > 1 and any(r in self.duplicates for r in result)
        is_repetitive = process_document(doc)[1] if not is_duplicate else True

        return doc, is_duplicate, is_repetitive

    def process_result(self, result):
        doc, is_duplicate, is_repetitive = result

        if is_duplicate:
            self.duplicates.add(f"doc_{hash(doc)}")

        if is_duplicate or is_repetitive:
            self.removed_docs += 1
        else:
            self.clean_docs_queue.put(doc)

        self.doc_count += 1
        self.docs_queue.task_done()

        if self.doc_count % 200 == 0:
            print(f"Total processed: {self.doc_count}, Total removed: {self.removed_docs}")

    def run(self):
        with Pool(cpu_count()) as pool:
            while True:
                doc = self.docs_queue.get()
                if doc is None:
                    self.clean_docs_queue.put(None)
                    self.docs_queue.task_done()
                    break

                pool.apply_async(self.process_doc_worker, args=(doc,), callback=self.process_result)

            pool.close()
            pool.join()

def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing JSONL files")
    parser.add_argument("--output", default="./", help="Output directory")
    return parser.parse_args()

def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def is_repetition_removal(text, duplicate_line_fraction=0.5, duplicate_line_character_faction=0.3):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    line_count = 0
    dup_line = 0
    dup_line_chars = 0
    visit_lines = {}
    for line in text.split("\n"):
        line_hash = hash_text(line)
        if line_hash in visit_lines:
            dup_line += 1
            dup_line_chars += len(line)
        visit_lines[line_hash] = True
        line_count += 1

    if float(dup_line) / line_count > duplicate_line_fraction:
        return True

    if float(dup_line_chars) / len(text) > duplicate_line_character_faction:
        return True

    top_ngram_character_fractions = [
        (2, 0.20),
        (3, 0.17),
        (4, 0.15),
    ]
    for ngram, threshold in top_ngram_character_fractions:
        try:
            word_list = list(jieba.cut(text))
            bgs = nltk.ngrams(word_list, ngram)
            fdist = nltk.FreqDist(bgs)
            for word_list, repeat in fdist.items():
                char_count = sum([len(word) for word in word_list])
                if char_count * (repeat - 1) / len(text) > threshold:
                    return True
        except Exception as e:
            print(f"Error in ngram analysis: {e}")

    duplicate_ngram_character_fractions = [
        (5, 0.27),
        (6, 0.25),
        (7, 0.23),
        (8, 0.20),
        (9, 0.18),
        (10, 0.15),
    ]
    for ngram, threshold in duplicate_ngram_character_fractions:
        try:
            fdist = {}
            word_list = list(jieba.cut(text))
            mark = [0] * len(word_list)
            for i in range(len(word_list) - ngram + 1):
                bag = tuple(word_list[i: i + ngram])
                if bag in fdist:
                    for j in range(i, i + ngram):
                        mark[j] = len(word_list[j])
                    fdist[bag] += 1
                else:
                    fdist[bag] = 1

            if sum(mark) / float(len(text)) > threshold:
                return True
        except Exception as e:
            print(f"Error in duplicate ngram analysis: {e}")

    return False

def process_document(doc):
    if not isinstance(doc, dict) or "text" not in doc:
        raise ValueError("Document must be a dictionary with a 'text' key")
    return doc, is_repetition_removal(doc["text"])

def read_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")

def write_parquet(data, output_dir, base_file_name, rows_per_file):
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory {output_dir}")

    file_path = os.path.join(output_dir, f"{base_file_name}.parquet")
    try:
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
    except Exception as e:
        print(f"Error writing parquet file {file_path}: {e}")

def create_minhash(doc):
    m = MinHash()
    for d in nltk.word_tokenize(doc):
        m.update(d.encode('utf8'))
    return m

def process_document_batch(docs, lsh, duplicates):
    results = []
    for doc in docs:
        minhash = create_minhash(doc['text'])
        lsh.insert(f"doc_{doc['id']}", minhash)  # Assuming each doc has a unique 'id'
        query_result = lsh.query(minhash)

        is_duplicate = len(query_result) > 1
        is_repetitive = False
        processed_doc = None

        if is_duplicate:
            duplicates.update(query_result)
        elif f"doc_{doc['id']}" not in duplicates:
            processed_doc, is_repetitive = process_document(doc)

        results.append((doc['id'], is_duplicate, is_repetitive, processed_doc))
    return results

def run_concurrent_processing(docs_queue, clean_docs_queue, lsh, duplicates):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        batch = []
        batch_size = 10  

        while True:
            doc = docs_queue.get()
            if doc is None:
                if batch:
                    futures.append(executor.submit(process_document_batch, batch, lsh, duplicates))
                break

            batch.append(doc)
            if len(batch) >= batch_size:
                futures.append(executor.submit(process_document_batch, batch, lsh, duplicates))
                batch = []

        for future in as_completed(futures):
            for doc_id, is_duplicate, is_repetitive, processed_doc in future.result():
                if is_duplicate or is_repetitive:
                    removed_docs += 1
                else:
                    if processed_doc is not None:
                        clean_docs_queue.put(processed_doc)

                docs_queue.task_done()


def writer_thread(clean_docs_queue, output_dir, batch_size=400000):
    batch = []
    batch_count = 0

    while True:
        clean_doc = clean_docs_queue.get()
        if clean_doc is not None:
            batch.append(clean_doc)

        if clean_doc is None:               
            print("Finishing writing")
            if batch:
                with tqdm(total=len(batch), desc=f"Writing Batch {batch_count}") as pbar:
                    write_parquet(batch, output_dir, f"clean_docs_batch_{batch_count}", len(batch))
                    pbar.update(len(batch))
            break

        if len(batch) >= batch_size:
            with tqdm(total=len(batch), desc=f"Writing Batch {batch_count}") as pbar:
                write_parquet(batch, output_dir, f"clean_docs_batch_{batch_count}", len(batch))
                pbar.update(len(batch))
            batch = []
            batch_count += 1

        clean_docs_queue.task_done()

def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist")

    os.makedirs(args.output, exist_ok=True)

    docs_queue = Queue(maxsize=20000000)
    clean_docs_queue = Queue()

    print("Dedup thread has started")
    dedup_thread = Thread(target=run_concurrent_processing, args=(docs_queue, clean_docs_queue))
    dedup_thread.start()

    print("Writer thread has started")
    writer_thread_instance = Thread(target=writer_thread, args=(clean_docs_queue, args.output, 400000))
    writer_thread_instance.start()

    for file_name in tqdm(os.listdir(args.input_dir), desc="Reading JSONL Files"):
        if file_name.endswith('.json'):
            file_path = os.path.join(args.input_dir, file_name)
            for doc in tqdm(read_jsonl(file_path), desc=f"Processing {file_name}"):
                docs_queue.put(doc)

    docs_queue.put(None)
    docs_queue.join()
    clean_docs_queue.put(None)
    clean_docs_queue.join()

    dedup_thread.join()
    writer_thread_instance.join()

    print(f"Processing complete. Check the output directory for results.")

if __name__ == "__main__":
    try:
        nltk.download('punkt')
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    main()
