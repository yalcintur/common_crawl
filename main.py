import argparse
import json
import os
from queue import Queue
import nltk
import time

from tqdm import tqdm
from datasketch import MinHashLSH
from datetime import datetime

from doc_processor import DocProcessor
from models import DocHash
from typing import List


def parse_args():
    parser = argparse.ArgumentParser("Filter out bad docs.")
    parser.add_argument(
        "--input_dir", required=True, help="Input directory containing JSONL files"
    )
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--lines_per_file", default=50000, help="Lines per file")

    return parser.parse_args()


def read_jsonl(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                yield json.loads(line)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")


def makedirsifnotexists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def all_keys(doc_hashes: List[DocHash]):
    for doc_hash in doc_hashes:
        yield doc_hash.url


def deduplicate(doc_hashes: List[DocHash]):
    lsh = MinHashLSH(threshold=0.6, num_perm=128)

    for doc_hash in doc_hashes:
        try:
            lsh.insert(doc_hash.url, doc_hash.hash)
        except ValueError as e:
            pass
        except Exception as e:
            print(f"Error inserting {doc_hash.url}: {e}")

        result = lsh.query(doc_hash.hash)

        if not (len(result) > 1):
            yield doc_hash.url


def list_jsonl_files(dir_path):
    file_paths = []

    for file_name in tqdm(os.listdir(dir_path), desc="Gathering JSONL Files"):
        if file_name.endswith(".json"):  ## Change this to .jsonl
            file_path = os.path.join(dir_path, file_name)
            file_paths.append(file_path)

    return file_paths


def write_to_file(file_path, data):
    start_time = time.time()
    with open(file_path, "w", encoding="utf-8") as f:
        for url in data:
            f.write(url + "\n")

    print(f"Saved {file_path} in {time.time() - start_time} seconds.")


def deduplicate_doc_hashes(
    doc_hashes: List[DocHash], lines_per_file: int, target_dir: str
):
    temp = []

    for deduplicated_url in tqdm(deduplicate(doc_hashes)):
        if deduplicated_url is not None:
            temp.append(deduplicated_url)

        if len(temp) >= lines_per_file:
            ## Write it line by line

            human_readable_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            write_to_file(
                os.path.join(
                    target_dir, f"deduplicated_urls_{human_readable_time}.txt"
                ),
                temp,
            )

            temp = []

    if len(temp) > 0:
        human_readable_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        write_to_file(
            os.path.join(target_dir, f"deduplicated_urls_{human_readable_time}.txt"),
            temp,
        )


def calculate_batch_doc_hashes(file_paths: List[str], lines_per_file: int):
    docs_queue = Queue()

    for file_path in tqdm(file_paths, desc="Reading JSONL Files"):
        for doc in read_jsonl(file_path):
            docs_queue.put(doc)

    print(f"Total number of files: {docs_queue.qsize()}")

    doc_processor = DocProcessor(docs_queue)

    doc_hashes = doc_processor.create_document_hashes()

    del docs_queue
    del doc_processor

    return doc_hashes


def main(args):
    TARGET_DIR = args.output
    LINES_PER_FILE = args.lines_per_file
    SOURCE_DIR = args.input_dir
    BATCH_COUNT = 4

    makedirsifnotexists(TARGET_DIR)

    file_paths = list_jsonl_files(SOURCE_DIR)

    filePathBacthQueue = Queue()

    for i in range(BATCH_COUNT):
        filePathBacthQueue.put(file_paths[i::BATCH_COUNT])

    # Clean file_paths from memory
    del file_paths

    doc_hashes = []

    for i in range(BATCH_COUNT):
        print(f"Starting batch {i}")

        doc_hashes.extend(
            calculate_batch_doc_hashes(filePathBacthQueue.get(), LINES_PER_FILE)
        )

    # Clean filePathBacthQueue from memory
    del filePathBacthQueue

    print("All hashes calculated. Starting deduplication.")

    deduplicate_doc_hashes(doc_hashes, LINES_PER_FILE, TARGET_DIR)

    print("Done.")


if __name__ == "__main__":
    try:
        nltk.download("punkt")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

    args = parse_args()

    main(args)
